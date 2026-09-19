"""Structural interface: adapters can use different optional environments."""
from typing import Protocol, Any


class InferenceBackend(Protocol):
    tokenizer: Any

    def load(self) -> None: ...

    def measure(self, prompt_tokens: list[int], max_new_tokens: int) -> dict: ...

    def metadata(self) -> dict: ...


def create_backend(config):
    kwargs = dict(model_path=config.model_path, prefill_step_size=config.prefill_step_size,
                  kv_bits=config.kv_bits)
    if config.wired_memory and config.backend not in ("mlx", "mlx-vlm"):
        raise ValueError("--wired-memory is supported only by MLX and MLX-VLM")
    if config.backend == "mlx":
        from .apple.mlx_backend import MLXBackend
        return MLXBackend(**kwargs, wired_memory=config.wired_memory)
    if config.backend == "transformers":
        from .apple.transformers.backend import TransformersBackend
        return TransformersBackend(**kwargs)
    if config.backend == "vllm":
        from .apple.vllm_backend import VLLMBackend
        return VLLMBackend(**kwargs)
    if config.backend == "mlx-vlm":
        from .apple.mlx_vlm_backend import MLXVLMBackend
        return MLXVLMBackend(**kwargs, wired_memory=config.wired_memory)
    raise ValueError(f"Unknown backend: {config.backend}")
