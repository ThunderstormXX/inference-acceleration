"""Shared benchmark configuration, independent of inference frameworks."""
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class BenchmarkConfig:
    backend: str
    model_path: str = str(ROOT / "models/qwen3.5-9b-mlx-4bit")
    dataset_path: str = str(ROOT / "data/raw/deepscaler-100.jsonl")
    count: int = 100
    max_new_tokens: int = 128
    max_prompt_tokens: int = 2048
    prefill_step_size: int = 512
    warmup: int = 1
    prompt_mode: str = "problem"
    chain_prefix_tokens: int = 512
    kv_bits: int | None = None
    label: str = ""
    user_initiated: bool = False
    wired_memory: bool = False

    def __post_init__(self):
        if type(self.user_initiated) is not bool:
            raise ValueError("user_initiated must be a boolean")
        if type(self.wired_memory) is not bool:
            raise ValueError("wired_memory must be a boolean")
        if self.wired_memory and self.backend not in ("mlx", "mlx-vlm", "mlx-dflash", "mlx-mtp"):
            raise ValueError("--wired-memory is supported only by MLX, MLX-VLM, MLX-DFlash and MLX-MTP")
        if not 1 <= self.count <= 100:
            raise ValueError("count must be between 1 and 100")
        if self.max_new_tokens < 2:
            raise ValueError("At least 2 output tokens are needed to measure decode")
        if self.max_prompt_tokens < 1 or self.prefill_step_size < 1 or self.warmup < 0:
            raise ValueError("Invalid prompt size, prefill step size or warmup")
        if self.prompt_mode not in ("problem", "chain-prefix"):
            raise ValueError("Unknown prompt mode")
        if self.chain_prefix_tokens < 1:
            raise ValueError("chain_prefix_tokens must be positive")
