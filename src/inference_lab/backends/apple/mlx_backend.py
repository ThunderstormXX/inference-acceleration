"""Phase-separated, batch-one text inference on Apple Metal with MLX-LM.

The first sampled token belongs to prefill. A request producing N tokens thus
executes N - 1 decode forwards. EOS is intentionally ignored for a reproducible
fixed-length speed workload; this is not an end-to-end chat-serving benchmark.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter
from typing import Any

from .mlx_memory import RecommendedWiredMemory


class MLXBackend:
    """Load a local MLX checkpoint and benchmark independent prompt requests.

    Qwen3.5 contains attention and recurrent layers. MLX-LM's model-specific
    cache factory creates both kinds of state; a new cache is used per request.
    Optional KV quantization affects attention caches, not recurrent states.
    """

    def __init__(
        self,
        model_path: str,
        prefill_step_size: int = 512,
        kv_bits: int | None = None,
        wired_memory: bool = False,
    ) -> None:
        if isinstance(prefill_step_size, bool) or not isinstance(prefill_step_size, int):
            raise ValueError("prefill_step_size must be a positive integer")
        if prefill_step_size < 1:
            raise ValueError("prefill_step_size must be a positive integer")
        if kv_bits is not None and (
            isinstance(kv_bits, bool) or not isinstance(kv_bits, int) or kv_bits not in (4, 8)
        ):
            raise ValueError("kv_bits must be None, 4, or 8")
        if type(wired_memory) is not bool:
            raise ValueError("wired_memory must be a boolean")
        self.model_path = str(Path(model_path).expanduser().resolve())
        self.prefill_step_size = prefill_step_size
        self.kv_bits = kv_bits
        self.wired_memory = wired_memory
        self.trace_generation = False
        self._mx: Any = None
        self._model: Any = None
        self._tokenizer: Any = None
        self._make_prompt_cache: Any = None
        self._model_config: dict[str, Any] = {}

    @property
    def tokenizer(self) -> Any:
        if self._tokenizer is None:
            raise RuntimeError("Call load() before accessing the tokenizer")
        return self._tokenizer

    def load(self) -> MLXBackend:
        """Fully materialize model weights before any inference measurements."""
        if self._model is not None:
            return self
        if not Path(self.model_path).is_dir():
            raise FileNotFoundError(f"Model directory does not exist: {self.model_path}")
        try:
            import mlx.core as mx
            from mlx_lm import load
            from mlx_lm.models.cache import make_prompt_cache
        except ImportError as exc:
            raise RuntimeError(
                "MLX-LM is unavailable. Run the project's environment setup script."
            ) from exc
        if not mx.metal.is_available():
            raise RuntimeError("This backend requires Apple Silicon with Metal available")
        mx.set_default_device(mx.gpu)
        model, tokenizer, config = load(
            self.model_path,
            tokenizer_config={"trust_remote_code": False, "local_files_only": True},
            lazy=False,
            return_config=True,
        )
        model.eval()
        mx.eval(model.parameters())
        mx.synchronize()
        self._mx = mx
        self._model = model
        self._tokenizer = tokenizer
        self._make_prompt_cache = make_prompt_cache
        self._model_config = config
        return self

    def _quantize_cache(self, cache: list[Any]) -> None:
        if self.kv_bits is not None:
            for index, layer_cache in enumerate(cache):
                if hasattr(layer_cache, "to_quantized"):
                    cache[index] = layer_cache.to_quantized(group_size=64, bits=self.kv_bits)

    def _next_token(self, input_tokens: Any, cache: list[Any]) -> Any:
        logits = self._model(input_tokens[None], cache=cache)
        self._quantize_cache(cache)
        # Greedy argmax on logits equals argmax on log probabilities; omitting
        # the unused log-softmax avoids additional vocabulary-wide work.
        return self._mx.argmax(logits[:, -1, :], axis=-1)

    def measure(
        self, prompt_tokens: list[int], max_new_tokens: int = 128
    ) -> dict[str, Any]:
        if self._model is None:
            raise RuntimeError("Call load() before measure()")
        if self.wired_memory:
            with RecommendedWiredMemory(self._mx):
                return self._measure(prompt_tokens, max_new_tokens)
        return self._measure(prompt_tokens, max_new_tokens)

    def _measure(
        self, prompt_tokens: list[int], max_new_tokens: int = 128
    ) -> dict[str, Any]:
        """Measure synchronized prefill and exactly N - 1 decode forwards.

        Input tokenization and final text decoding are outside the timed regions.
        Prefill includes first-token greedy sampling. Decode includes Python
        orchestration and token transfer, with GPU work pipelined one step ahead.
        No decode work is submitted until the prefill boundary has synchronized.
        """
        if self._model is None:
            raise RuntimeError("Call load() before measure()")
        if not prompt_tokens:
            raise ValueError("prompt_tokens must contain at least one token")
        if any(
            isinstance(token, bool) or not isinstance(token, int) or token < 0
            for token in prompt_tokens
        ):
            raise ValueError("prompt_tokens must contain nonnegative integer token IDs")
        if (
            isinstance(max_new_tokens, bool)
            or not isinstance(max_new_tokens, int)
            or max_new_tokens < 1
        ):
            raise ValueError("max_new_tokens must be a positive integer")

        mx = self._mx
        cache = self._make_prompt_cache(self._model)
        prompt = mx.array(prompt_tokens, dtype=mx.uint32)
        mx.eval(prompt)
        mx.synchronize()
        mx.clear_cache()
        mx.reset_peak_memory()

        recorder = None
        if self.trace_generation:
            from inference_lab.visualization.recording import GenerationRecorder
            recorder = GenerationRecorder(self, clock=perf_counter)
        prefill_start = perf_counter()
        if recorder is not None:
            recorder.start_at(prefill_start)
        # Only cache state is evaluated for these chunks. This lets MLX discard
        # the unused vocabulary projection for all but the final prompt token.
        position = 0
        while position < len(prompt_tokens) - 1:
            end = min(position + self.prefill_step_size, len(prompt_tokens) - 1)
            self._model(prompt[position:end][None], cache=cache)
            self._quantize_cache(cache)
            mx.eval([entry.state for entry in cache])
            mx.clear_cache()
            position = end
        token = self._next_token(prompt[-1:], cache)
        mx.eval(token, [entry.state for entry in cache])
        mx.synchronize()
        first_token_id = int(token.item())
        if recorder is not None:
            recorder.commit([first_token_id])
        prefill_seconds = perf_counter() - prefill_start

        generated = [first_token_id]
        decode_seconds = 0.0
        if max_new_tokens > 1:
            decode_start = prefill_start + prefill_seconds if recorder is not None else perf_counter()
            for step in range(max_new_tokens - 1):
                next_token = self._next_token(token, cache)
                mx.async_eval(next_token)
                # Reading the previous token, after submitting its successor,
                # overlaps CPU scheduling with device execution. The first token
                # has already been recorded at the prefill boundary.
                if step:
                    generated.append(int(token.item()))
                    if recorder is not None:
                        recorder.commit([generated[-1]])
                token = next_token
                if (step + 1) % 256 == 0:
                    mx.clear_cache()
            generated.append(int(token.item()))
            if recorder is not None:
                recorder.commit([generated[-1]])
            mx.eval([entry.state for entry in cache])
            mx.synchronize()
            decode_seconds = perf_counter() - decode_start

        peak_memory_gb = float(mx.get_peak_memory()) / 1_000_000_000
        output_text = self.tokenizer.decode(generated, skip_special_tokens=False)
        result = {
            "prompt_tokens": len(prompt_tokens),
            "generated_tokens": len(generated),
            "decode_tokens": len(generated) - 1,
            "prefill_seconds": prefill_seconds,
            "decode_seconds": decode_seconds,
            "generated_token_ids": generated,
            "output_text": output_text,
            "peak_memory_gb": peak_memory_gb,
            "timing_method": "perf_counter + MLX phase synchronization; first token in prefill",
        }

        if recorder is not None:
            result["generation_trace"] = recorder.finish_measurement(result)
        return result

    def metadata(self) -> dict[str, Any]:
        """Describe the actual execution policy alongside benchmark results."""
        packages = {}
        for package in ("mlx", "mlx-lm"):
            try:
                packages[package] = version(package)
            except PackageNotFoundError:
                packages[package] = None
        quantization = self._model_config.get("quantization") or {}
        return {
            "framework": "mlx-lm",
            "trace_generation": self.trace_generation,
            "generation_timing_policy": (
                "instrumented; host token-availability events included in phase times; no display enrichment"
                if self.trace_generation else "uninstrumented generation"
            ),
            "versions": packages,
            "device": "metal",
            "device_info": dict(self._mx.device_info()) if self._mx is not None else None,
            "model_path": self.model_path,
            "model_type": self._model_config.get("model_type"),
            "weight_bits": quantization.get("bits"),
            "weight_group_size": quantization.get("group_size"),
            "prefill_step_size": self.prefill_step_size,
            "wired_memory": self.wired_memory,
            "wired_memory_policy": (
                "recommended working-set size per request; previous MLX limit restored in finally"
                if self.wired_memory else "default MLX allocator policy"
            ),
            "kv_bits": self.kv_bits,
            "kv_group_size": 64 if self.kv_bits is not None else None,
            "kv_quantization_scope": "attention caches only; recurrent state unchanged",
            "sampling": "greedy",
            "ignore_eos": True,
            "batch_size": 1,
            "fresh_cache_per_request": True,
            "first_generated_token_phase": "prefill",
            "text_only": True,
            "peak_memory_unit": "decimal GB, MLX allocator peak per request",
        }
