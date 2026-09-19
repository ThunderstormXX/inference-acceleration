"""Transformers/PyTorch MPS inference from the same MLX affine checkpoint.

MLX checkpoints do not load directly through ``from_pretrained``. This adapter
keeps their packed linear weights and uses Transformers' native MetalLinear.
The embedding alone is dequantized to BF16, in bounded CPU chunks. The model
architecture, attention, recurrent cache and forward pass remain Transformers.
"""

from __future__ import annotations

import json
import os
import platform
from contextlib import ExitStack
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter
from typing import Any


class TransformersBackend:
    _NORM_SUFFIXES = (".input_layernorm.weight", ".post_attention_layernorm.weight",
                      "model.norm.weight", ".q_norm.weight", ".k_norm.weight")

    def __init__(self, model_path: str, prefill_step_size: int = 512,
                 kv_bits: int | None = None) -> None:
        if not isinstance(prefill_step_size, int) or isinstance(prefill_step_size, bool) or prefill_step_size < 1:
            raise ValueError("prefill_step_size must be a positive integer")
        if kv_bits is not None:
            raise ValueError("Transformers MPS adapter does not implement quantized KV cache")
        self.model_path = str(Path(model_path).expanduser().resolve())
        self.prefill_step_size = prefill_step_size
        self.kv_bits = kv_bits
        self._model: Any = None
        self._torch: Any = None
        self._tokenizer: Any = None
        self._load_details: dict[str, Any] = {}

    @property
    def tokenizer(self) -> Any:
        if self._tokenizer is None:
            raise RuntimeError("Call load() before accessing the tokenizer")
        return self._tokenizer

    @staticmethod
    def _text_key(key: str) -> str | None:
        if key.startswith(("vision_tower.", "model.visual.")) or "mtp." in key:
            return None
        if key.startswith("language_model."):
            return key[len("language_model."):]
        if key.startswith("model.language_model."):
            return "model." + key[len("model.language_model."):]
        return key

    @classmethod
    def _restore_mlx_layout(cls, name: str, value: Any, sanitized: bool) -> Any:
        """Invert MLX's layout and zero-centered norm conversion, once only."""
        if not sanitized:
            return value
        if name.endswith("conv1d.weight"):
            return value.transpose(1, 2).contiguous()
        if name.endswith(cls._NORM_SUFFIXES):
            # MLX uses w*x; HF uses (1+w)*x. FP32 subtraction preserves the
            # effective BF16 norm weights without another rounding to BF16.
            return value.float() - 1.0
        return value

    def load(self) -> TransformersBackend:
        if self._model is not None:
            return self
        import torch
        from safetensors import safe_open
        from transformers import AutoTokenizer, Qwen3_5ForCausalLM, Qwen3_5TextConfig
        from transformers.integrations import metal_quantization
        from transformers.integrations.metal_quantization import MetalLinear, _get_metal_kernel
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

        if not torch.backends.mps.is_available():
            raise RuntimeError("Transformers Metal backend requires an available MPS device")
        root = Path(self.model_path)
        config_data = json.loads((root / "config.json").read_text())
        quant = config_data.get("quantization", {})
        if config_data.get("model_type") != "qwen3_5" or quant.get("mode", "affine") != "affine":
            raise ValueError("Adapter supports only Qwen3.5 MLX affine checkpoints")
        bits, group_size = int(quant["bits"]), int(quant["group_size"])
        if bits not in (2, 4, 8):
            raise ValueError("Transformers Metal requires 2-, 4-, or 8-bit affine weights")
        shards = sorted(root.glob("*.safetensors"))
        if not shards:
            raise FileNotFoundError(f"No safetensors shards in {root}")
        # The published native binary embeds Metal4 shaders, incompatible with
        # macOS15. Compile the same upstream templates on this host instead.
        kernel_details = {"implementation": "Hugging Face precompiled native Metal kernel"}
        if int(platform.mac_ver()[0].split('.')[0]) < 26:
            from .metal_runtime import RuntimeAffineKernel
            kernel = RuntimeAffineKernel(torch, bits=bits, group_size=group_size)
            kernel.compile()
            metal_quantization._metal_kernel = kernel
            kernel_details = kernel.source_metadata
        else:
            _get_metal_kernel()
        config = Qwen3_5TextConfig(**config_data["text_config"])
        config._attn_implementation = "sdpa"
        old_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            with torch.device("meta"):
                model = Qwen3_5ForCausalLM(config)
        finally:
            torch.set_default_dtype(old_dtype)
        # Nonpersistent RoPE buffers are computed rather than checkpoint-loaded.
        model.model.rotary_emb = Qwen3_5TextRotaryEmbedding(config)

        with ExitStack() as stack:
            tensors: dict[str, tuple[Any, str]] = {}
            for shard in shards:
                handle = stack.enter_context(safe_open(str(shard), framework="pt", device="cpu"))
                for original in handle.keys():
                    key = self._text_key(original)
                    if key is not None:
                        if key in tensors:
                            raise ValueError(f"Duplicate checkpoint tensor: {key}")
                        tensors[key] = (handle, original)

            def read(key: str) -> Any:
                handle, original = tensors[key]
                return handle.get_tensor(original)

            conv_keys = [key for key in tensors if key.endswith("conv1d.weight")]
            if not conv_keys:
                raise ValueError("Expected Qwen3.5 recurrent convolution weights")
            shapes = [tensors[key][0].get_slice(tensors[key][1]).get_shape() for key in conv_keys]
            sanitized = all(shape[-1] == 1 for shape in shapes)
            if not sanitized and not all(shape[1] == 1 for shape in shapes):
                raise ValueError("Mixed or unknown convolution layouts in checkpoint")
            converted = 0
            for name, module in list(model.named_modules()):
                if isinstance(module, torch.nn.Linear) and f"{name}.scales" in tensors:
                    with torch.device("meta"):
                        replacement = MetalLinear(module.in_features, module.out_features,
                                                  bias=module.bias is not None,
                                                  bits=bits, group_size=group_size)
                    model.set_submodule(name, replacement)
                    converted += 1
            used: set[str] = set()
            for name, parameter in list(model.named_parameters()):
                source = name[:-len("qbiases")] + "biases" if name.endswith(".qbiases") else name
                if source not in tensors:
                    raise ValueError(f"Missing checkpoint tensor required by Transformers: {source}")
                value = read(source)
                used.add(source)
                if name == "model.embed_tokens.weight" and f"model.embed_tokens.scales" in tensors:
                    scales = read("model.embed_tokens.scales")
                    biases = read("model.embed_tokens.biases")
                    used.update(("model.embed_tokens.scales", "model.embed_tokens.biases"))
                    value = self._dequantize_embedding(value, scales, biases, bits, group_size, torch)
                else:
                    value = self._restore_mlx_layout(name, value, sanitized)
                if tuple(value.shape) != tuple(parameter.shape):
                    raise ValueError(f"Shape mismatch for {name}: {tuple(value.shape)} != {tuple(parameter.shape)}")
                module_name, attr = name.rsplit(".", 1)
                model.get_submodule(module_name)._parameters[attr] = torch.nn.Parameter(
                    value.to("mps"), requires_grad=False)
            unused = sorted(set(tensors) - used)
            if unused:
                raise ValueError(f"Unconsumed text checkpoint tensors: {unused[:10]}")
        model.to("mps")
        model.eval()
        torch.mps.synchronize()
        self._model, self._torch = model, torch
        self._tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True, trust_remote_code=False)
        self._load_details = {
            "linear_layers_native_metal": converted,
            "weight_bits": bits,
            "weight_group_size": group_size,
            "embedding_storage": "BF16 dequantized from original affine weights",
            "mlx_norm_and_conv_layout_reversed": sanitized,
            "metal_kernel": kernel_details,
        }
        return self

    @staticmethod
    def _dequantize_embedding(packed: Any, scales: Any, biases: Any,
                              bits: int, group_size: int, torch: Any) -> Any:
        rows, packed_width = packed.shape
        count = 32 // bits
        width = packed_width * count
        result = torch.empty((rows, width), dtype=torch.bfloat16)
        shifts = torch.arange(count, dtype=torch.int32) * bits
        for start in range(0, rows, 1024):
            end = min(start + 1024, rows)
            values = ((packed[start:end].to(torch.int32)[..., None] >> shifts) & ((1 << bits) - 1))
            values = values.reshape(end - start, width // group_size, group_size).float()
            values = values * scales[start:end].float()[..., None] + biases[start:end].float()[..., None]
            result[start:end] = values.reshape(end - start, width).to(torch.bfloat16)
        return result

    def measure(self, prompt_tokens: list[int], max_new_tokens: int = 128) -> dict[str, Any]:
        if self._model is None:
            raise RuntimeError("Call load() before measure()")
        if not prompt_tokens or any(isinstance(t, bool) or not isinstance(t, int) or t < 0 for t in prompt_tokens):
            raise ValueError("prompt_tokens must be nonempty nonnegative integer token IDs")
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        torch = self._torch
        prompt = torch.tensor([prompt_tokens], dtype=torch.long, device="mps")
        cache = None
        torch.mps.synchronize()
        torch.mps.empty_cache()
        sampled_memory = torch.mps.current_allocated_memory()
        with torch.inference_mode():
            started = perf_counter()
            # Match the MLX path: cache-only chunks and one final prompt token.
            # Avoid a discarded vocabulary projection at each chunk boundary.
            for start in range(0, len(prompt_tokens) - 1, self.prefill_step_size):
                end = min(start + self.prefill_step_size, len(prompt_tokens) - 1)
                output = self._model.model(input_ids=prompt[:, start:end],
                                           past_key_values=cache, use_cache=True)
                cache = output.past_key_values
            output = self._model(input_ids=prompt[:, -1:], past_key_values=cache,
                                 use_cache=True, logits_to_keep=1)
            cache = output.past_key_values
            token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
            torch.mps.synchronize()
            first_token = int(token.item())
            prefill_seconds = perf_counter() - started
            sampled_memory = max(sampled_memory, torch.mps.current_allocated_memory())
            generated = [first_token]
            decode_seconds = 0.0
            if max_new_tokens > 1:
                decode_tensors = []
                started = perf_counter()
                for _ in range(max_new_tokens - 1):
                    output = self._model(input_ids=token, past_key_values=cache,
                                         use_cache=True, logits_to_keep=1)
                    cache = output.past_key_values
                    token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
                    # Keep token feedback on MPS instead of synchronizing the
                    # CPU for each token; collect all IDs at the phase boundary.
                    decode_tensors.append(token)
                    sampled_memory = max(sampled_memory, torch.mps.current_allocated_memory())
                torch.mps.synchronize()
                generated.extend(torch.cat(decode_tensors, dim=-1).cpu().flatten().tolist())
                decode_seconds = perf_counter() - started
        return {
            "prompt_tokens": len(prompt_tokens), "generated_tokens": len(generated),
            "decode_tokens": len(generated) - 1,
            "prefill_seconds": prefill_seconds, "decode_seconds": decode_seconds,
            "generated_token_ids": generated,
            "output_text": self.tokenizer.decode(generated, skip_special_tokens=False),
            "peak_memory_gb": sampled_memory / 1_000_000_000,
            "timing_method": "perf_counter + torch.mps.synchronize; first token in prefill",
        }

    def metadata(self) -> dict[str, Any]:
        packages = {}
        for package in ("torch", "transformers", "kernels", "safetensors"):
            try:
                packages[package] = version(package)
            except PackageNotFoundError:
                packages[package] = None
        return {"backend": "transformers_mps_metal", "model_path": self.model_path,
                "packages": packages, "batch_size": 1, "sampling": "greedy; ignore EOS",
                "prefill_step_size": self.prefill_step_size, "prefix_cache": False,
                "memory_measurement": "max sampled live MPS tensor allocation; lower bound on peak",
                "mps_cpu_fallback_enabled": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1",
                "checkpoint_adapter": "MLX affine to native Transformers MetalLinear; text-only",
                **self._load_details}
