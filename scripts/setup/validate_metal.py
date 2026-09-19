"""Check quantized Torch Metal matrix products against independent CPU math."""

from __future__ import annotations

import fcntl
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# setup/transformers.py must not shadow the installed Transformers package.
sys.path[:] = [str(ROOT / "src"), *[entry for entry in sys.path
                                  if Path(entry or ".").resolve() != Path(__file__).resolve().parent]]

import torch

from inference_lab.backends.apple.transformers.metal_runtime import RuntimeAffineKernel


class MetalKernelValidator:
    """Numerical GPU validation with the project-wide execution lock."""

    def run(self) -> None:
        artifacts = ROOT / "artifacts"
        artifacts.mkdir(exist_ok=True)
        with (artifacts / "gpu.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            kernel = RuntimeAffineKernel(torch)
            kernel.compile()
            results = []
            for dtype in (torch.float32, torch.float16, torch.bfloat16):
                for m, n, k in ((1, 32, 64), (1, 35, 128), (7, 35, 128), (33, 128, 128)):
                    generator = torch.Generator().manual_seed(19)
                    quantized = torch.randint(0, 16, (n, k), generator=generator, dtype=torch.int64)
                    packed = sum(quantized[:, i::8] << (4 * i) for i in range(8)).to(torch.uint32)
                    scales = (torch.rand((n, k // 64), generator=generator) * .1).to(dtype)
                    biases = (torch.rand((n, k // 64), generator=generator) * .2 - .1).to(dtype)
                    inputs = torch.randn((m, k), generator=generator).to(dtype)
                    dequantized = (quantized.reshape(n, -1, 64).float() * scales.float()[..., None]
                                   + biases.float()[..., None]).reshape(n, k)
                    expected = inputs.float() @ dequantized.T
                    actual = kernel.affine_qmm_t(inputs.to("mps"), packed.to("mps"),
                                                 scales.to("mps"), biases.to("mps"), 64, 4).cpu().float()
                    tolerances = {torch.float32: (1e-5, 1e-5), torch.float16: (.005, .015),
                                  torch.bfloat16: (.03, .12)}
                    rtol, atol = tolerances[dtype]
                    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
                    result = {"dtype": str(dtype), "M": m, "N": n, "K": k,
                              "max_absolute_error": (actual - expected).abs().max().item(),
                              "rtol": rtol, "atol": atol, "passed": True}
                    results.append(result)
                    print(json.dumps(result), flush=True)
            output = artifacts / "metal-kernel-validation.json"
            output.write_text(json.dumps({"created_at": datetime.now(timezone.utc).isoformat(),
                                         "torch": torch.__version__, "kernel": kernel.source_metadata,
                                         "cases": results, "passed": True}, indent=2) + "\n")
            print(f"Passed {len(results)} cases. Results: {output}")

if __name__ == "__main__":
    MetalKernelValidator().run()
