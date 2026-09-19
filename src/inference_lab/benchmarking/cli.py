import argparse
import json
from pathlib import Path
from dataclasses import fields
from inference_lab.core.config import ROOT, BenchmarkConfig
from .runner import BenchmarkRunner


def main(backend: str):
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--config", type=Path, default=ROOT / "configs/benchmark.json")
    initial, _ = preliminary.parse_known_args()
    allowed = {field.name for field in fields(BenchmarkConfig)} - {"backend"}
    values = json.loads(initial.config.read_text())
    fixed = {"batch_size": 1, "sampling": "greedy", "ignore_eos": True,
             "prefix_cache": False}
    for name, expected in fixed.items():
        if name in values and (type(values[name]) is not type(expected) or values[name] != expected):
            preliminary.error(f"This measurement protocol requires {name}={expected!r}")
    unknown = set(values) - allowed - set(fixed) - {"backend"}
    if unknown:
        preliminary.error(f"Unknown configuration fields: {', '.join(sorted(unknown))}")
    if "backend" in values and values["backend"] != backend:
        preliminary.error(f"This launcher selects backend={backend!r}")
    defaults = BenchmarkConfig(backend, **{k: v for k, v in values.items() if k in allowed})
    parser = argparse.ArgumentParser(description=f"Benchmark {backend}: separate prefill and decode")
    parser.add_argument("--config", type=Path, default=initial.config)
    parser.add_argument("--model-path", default=defaults.model_path)
    parser.add_argument("--dataset-path", default=defaults.dataset_path)
    parser.add_argument("--count", type=int, default=defaults.count)
    parser.add_argument("--max-new-tokens", type=int, default=defaults.max_new_tokens)
    parser.add_argument("--max-prompt-tokens", type=int, default=defaults.max_prompt_tokens)
    parser.add_argument("--prefill-step-size", type=int, default=defaults.prefill_step_size)
    parser.add_argument("--warmup", type=int, default=defaults.warmup)
    parser.add_argument("--prompt-mode", choices=["problem", "chain-prefix"], default=defaults.prompt_mode)
    parser.add_argument("--chain-prefix-tokens", type=int, default=defaults.chain_prefix_tokens)
    parser.add_argument("--kv-bits", type=int, choices=[4, 8], default=defaults.kv_bits)
    parser.add_argument("--label", default=defaults.label)
    parser.add_argument("--user-initiated", action=argparse.BooleanOptionalAction,
                        default=defaults.user_initiated,
                        help="Declare this finite run as a user-initiated macOS activity")
    parser.add_argument("--wired-memory", action=argparse.BooleanOptionalAction,
                        default=defaults.wired_memory,
                        help="Use MLX's recommended wired-memory limit per request (MLX/MLX-VLM only)")
    args = parser.parse_args()
    del args.config
    if args.label and not all(c.isalnum() or c in "-_" for c in args.label):
        parser.error("label may contain letters, digits, - and _ only")
    BenchmarkRunner(BenchmarkConfig(backend=backend, **vars(args))).run()
