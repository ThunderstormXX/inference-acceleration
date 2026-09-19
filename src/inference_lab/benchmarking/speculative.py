"""Phase-separated, fixed-length DFlash measurements using the shared runner."""
from __future__ import annotations

import argparse
from dataclasses import dataclass

from inference_lab.core.config import ROOT, BenchmarkConfig
from .runner import BenchmarkRunner


@dataclass(frozen=True)
class SpeculativeConfig(BenchmarkConfig):
    draft_path: str = str(ROOT / "models/qwen3.5-9b-dflash")
    draft_bits: int = 4
    block_size: int = 3

    def __post_init__(self):
        super().__post_init__()
        if self.backend != "mlx-dflash":
            raise ValueError("SpeculativeConfig requires backend mlx-dflash")
        if type(self.draft_bits) is not int or self.draft_bits not in (4, 8):
            raise ValueError("draft_bits must be 4 or 8")
        if type(self.block_size) is not int or not 2 <= self.block_size <= 5:
            raise ValueError("Quantized DFlash experiments use block_size 2..5")
        if self.kv_bits is not None:
            raise ValueError("KV-cache quantization is not part of this DFlash protocol")
        if not self.wired_memory:
            raise ValueError("The upstream DFlash generator always uses wired memory")


def create_speculative_backend(config):
    from inference_lab.backends.apple.speculative.dflash_backend import DFlashBackend
    return DFlashBackend(model_path=config.model_path, draft_path=config.draft_path,
                        draft_bits=config.draft_bits, block_size=config.block_size,
                        prefill_step_size=config.prefill_step_size, wired_memory=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("baseline", "dflash"), default="dflash")
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--block-size", type=int, choices=range(2, 6), default=3)
    parser.add_argument("--draft-bits", type=int, choices=(4, 8), default=4)
    parser.add_argument("--model-path", default=BenchmarkConfig.model_path)
    parser.add_argument("--draft-path", default=SpeculativeConfig.draft_path)
    parser.add_argument("--dataset-path", default=BenchmarkConfig.dataset_path)
    parser.add_argument("--label")
    args = parser.parse_args()
    label = args.label or ("spec-baseline" if args.mode == "baseline"
                           else f"dflash-k{args.block_size}-q{args.draft_bits}")
    if not label or not all(c.isalnum() or c in "-_" for c in label):
        parser.error("label may contain letters, digits, - and _ only")
    common = dict(count=args.count, max_new_tokens=args.max_new_tokens, warmup=args.warmup,
                  model_path=args.model_path, dataset_path=args.dataset_path,
                  label=label, wired_memory=True)
    if args.mode == "baseline":
        BenchmarkRunner(BenchmarkConfig(backend="mlx", **common)).run()
    else:
        config = SpeculativeConfig(backend="mlx-dflash", draft_path=args.draft_path,
                                   draft_bits=args.draft_bits, block_size=args.block_size, **common)
        BenchmarkRunner(config, create_speculative_backend).run()
    return 0
