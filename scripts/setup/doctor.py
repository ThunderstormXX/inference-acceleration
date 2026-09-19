"""Verify actual imports/device availability in each isolated environment."""
import argparse
import json
from pathlib import Path
import subprocess
import os

ROOT = Path(__file__).resolve().parents[2]


class EnvironmentDoctor:
    def run(self):
        checks = {
            "mlx": (".venv", "import mlx.core as mx; from mlx_lm import load; from mlx_vlm.utils import load_model; print('Metal available:', mx.metal.is_available()); print(mx.device_info())"),
            "transformers": (".venv-transformers", "import torch; from transformers import Qwen3_5ForCausalLM; from transformers.integrations.metal_quantization import _get_metal_kernel; print('MPS available:', torch.backends.mps.is_available()); print('Native kernel import:', bool(_get_metal_kernel()), '(shader execution requires the benchmark smoke test)')"),
            "vllm": (".venv-vllm", "from vllm import LLM, SamplingParams; import vllm_metal; import mlx.core as mx; print('Metal available:', mx.metal.is_available()); print('Plugin:', vllm_metal.register())"),
        }
        records = {}
        for name, (directory, code) in checks.items():
            python = ROOT / directory / "bin/python"
            print(f"Checking {name}: {python}", flush=True)
            try:
                result = subprocess.run([str(python), "-c", code], cwd=ROOT,
                                        capture_output=True, text=True, timeout=120)
                records[name] = {"status": "ok" if result.returncode == 0 else "failed",
                                 "returncode": result.returncode,
                                 "stdout": result.stdout, "stderr": result.stderr}
                print(result.stdout + result.stderr, flush=True)
            except (OSError, subprocess.TimeoutExpired) as error:
                records[name] = {"status": "failed", "error": str(error)}
        output = ROOT / "artifacts/reports/environment-check.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
        return int(any(r["status"] != "ok" for r in records.values()))


if __name__ == "__main__":
    raise SystemExit(EnvironmentDoctor().run())
