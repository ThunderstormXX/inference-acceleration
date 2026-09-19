"""Official vLLM + vllm-metal, driven one synchronous engine step at a time."""
import os
from importlib.metadata import version
from pathlib import Path
from time import perf_counter


class VLLMBackend:
    def __init__(self, model_path: str, prefill_step_size: int = 512, kv_bits=None):
        if kv_bits is not None:
            raise ValueError("This vLLM adapter does not enable quantized KV cache")
        self.model_path = str(Path(model_path).resolve())
        self.prefill_step_size = prefill_step_size
        self.engine = None
        self.counter = 0

    def load(self):
        # In-process execution and disabled async scheduling are both required:
        # vLLM otherwise queues the next forward before returning the first
        # token, even without a background engine process.
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        os.environ.setdefault("VLLM_METAL_MEMORY_FRACTION", "auto")
        os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
        os.environ.setdefault("DO_NOT_TRACK", "1")
        from vllm import LLM, SamplingParams
        from vllm.sampling_params import RequestOutputKind
        from transformers import AutoTokenizer
        import mlx.core as mx

        self._mx = mx
        self._SamplingParams = SamplingParams
        self._output_kind = RequestOutputKind.CUMULATIVE
        self.llm = LLM(
            model=self.model_path, tokenizer=self.model_path,
            max_model_len=4096, max_num_seqs=1,
            gpu_memory_utilization=0.65,
            max_num_batched_tokens=self.prefill_step_size,
            enable_prefix_caching=False, enable_chunked_prefill=True,
            async_scheduling=False,
            enforce_eager=True, trust_remote_code=False,
            disable_log_stats=True, seed=0,
            limit_mm_per_prompt={"image": 0, "video": 0},
        )
        self.engine = self.llm.llm_engine
        self._validate_synchronous_engine()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
        mx.synchronize()
        return self

    def _validate_synchronous_engine(self):
        client = getattr(self.engine, "engine_core", None)
        core = getattr(client, "engine_core", None)
        if core is None:
            raise RuntimeError("Phase timing requires vLLM's in-process engine core")
        if getattr(core, "async_scheduling", None) is not False:
            raise RuntimeError("Phase timing requires vLLM async scheduling to be disabled")
        if not hasattr(core, "batch_queue") or core.batch_queue is not None:
            raise RuntimeError("Phase timing requires vLLM's forward batch queue to be disabled")

    def measure(self, prompt_tokens: list[int], max_new_tokens: int = 128):
        if self.engine is None:
            raise RuntimeError("Call load() first")
        if len(prompt_tokens) + max_new_tokens > 4096:
            raise ValueError("Prompt plus output exceeds the vLLM context limit of 4096")
        self.counter += 1
        params = self._SamplingParams(temperature=0.0, max_tokens=max_new_tokens,
                                      ignore_eos=True,
                                      detokenize=False, output_kind=self._output_kind)
        self._mx.synchronize()
        self._mx.reset_peak_memory()
        started = perf_counter()
        self.engine.add_request(str(self.counter), {"prompt_token_ids": prompt_tokens}, params)
        first_at = None
        output = None
        while self.engine.has_unfinished_requests():
            for result in self.engine.step():
                if result.outputs and result.outputs[0].token_ids:
                    if first_at is None:
                        if len(result.outputs[0].token_ids) != 1:
                            raise RuntimeError("First engine output contains multiple tokens; phase split would be invalid")
                        self._mx.synchronize()
                        first_at = perf_counter()
                    output = result
        self._mx.synchronize()
        finished = perf_counter()
        if output is None or first_at is None:
            raise RuntimeError("vLLM emitted no tokens")
        tokens = list(output.outputs[0].token_ids)
        if len(tokens) != max_new_tokens:
            raise RuntimeError(f"Expected {max_new_tokens} output tokens, received {len(tokens)}")
        if output.num_cached_tokens:
            raise RuntimeError("Unexpected prefix-cache hit in cold-request benchmark")
        return {
            "prompt_tokens": len(prompt_tokens), "generated_tokens": len(tokens),
            "decode_tokens": len(tokens) - 1,
            "prefill_seconds": first_at - started, "decode_seconds": finished - first_at,
            "generated_token_ids": tokens,
            "output_text": self.tokenizer.decode(tokens, skip_special_tokens=False),
            "peak_memory_gb": self._mx.get_peak_memory() / 1e9,
            "timing_method": "synchronous in-process vLLM engine steps + Metal synchronization; includes scheduler, first token in prefill",
        }

    def metadata(self):
        return {
            "framework": "vllm", "plugin": "vllm-metal", "device": "metal",
            "versions": {name: version(name) for name in ("vllm", "vllm-metal", "mlx", "mlx-lm", "mlx-vlm")},
            "batch_size": 1, "max_model_len": 4096,
            "gpu_memory_utilization": 0.65,
            "prefill_step_size": self.prefill_step_size,
            "prefix_caching": False, "multiprocessing": False,
            "async_scheduling": False, "forward_batch_queue": False,
            "ignore_eos": True, "sampling": "greedy", "enforce_eager": True,
            "timing_scope": "engine latency including synchronous scheduler; no server/network",
            "peak_memory_unit": "decimal GB, MLX allocator",
        }
