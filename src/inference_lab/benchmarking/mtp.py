"""Compare native MTP against the same MLX-VLM target and environment."""
from __future__ import annotations

import argparse
from dataclasses import dataclass

from inference_lab.core.config import ROOT, BenchmarkConfig
from .runner import BenchmarkRunner


@dataclass(frozen=True)
class MTPConfig(BenchmarkConfig):
    draft_path: str = str(ROOT / "models/qwen3.5-9b-mtp-4bit")
    block_size: int = 3

    def __post_init__(self):
        super().__post_init__()
        if self.backend != "mlx-mtp":
            raise ValueError("MTPConfig requires backend mlx-mtp")
        if type(self.block_size) is not int or not 2 <= self.block_size <= 5:
            raise ValueError("Native MTP experiments use block_size 2..5")
        if self.kv_bits is not None:
            raise ValueError("KV-cache quantization is not part of this MTP protocol")
        if not self.wired_memory:
            raise ValueError("The paired MTP protocol requires wired_memory=True")


def create_mtp_backend(config):
    from inference_lab.backends.apple.speculative.mtp_backend import MTPBackend
    return MTPBackend(model_path=config.model_path, draft_path=config.draft_path,
                      block_size=config.block_size, prefill_step_size=config.prefill_step_size,
                      wired_memory=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("baseline", "mtp"), default="mtp")
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--block-size", type=int, choices=range(2, 6), default=MTPConfig.block_size)
    parser.add_argument("--prefill-step-size", type=int, default=512)
    parser.add_argument("--model-path", default=BenchmarkConfig.model_path)
    parser.add_argument("--draft-path", default=MTPConfig.draft_path)
    parser.add_argument("--dataset-path", default=BenchmarkConfig.dataset_path)
    parser.add_argument("--label")
    args = parser.parse_args()
    label = args.label or ("mtp-baseline" if args.mode == "baseline" else f"mtp-k{args.block_size}")
    if not label or not all(char.isalnum() or char in "-_" for char in label):
        parser.error("label may contain letters, digits, - and _ only")
    common = dict(count=args.count, max_new_tokens=args.max_new_tokens, warmup=args.warmup,
                  model_path=args.model_path, dataset_path=args.dataset_path,
                  prefill_step_size=args.prefill_step_size, label=label, wired_memory=True)
    if args.mode == "baseline":
        BenchmarkRunner(BenchmarkConfig(backend="mlx-vlm", **common)).run()
    else:
        config = MTPConfig(backend="mlx-mtp", draft_path=args.draft_path,
                           block_size=args.block_size, **common)
        BenchmarkRunner(config, create_mtp_backend).run()
    return 0
