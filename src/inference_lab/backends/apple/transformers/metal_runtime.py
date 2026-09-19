"""Compile the upstream MLX affine shaders through PyTorch's MPS runtime.

The Hub's precompiled library uses Metal 4.0 and cannot execute on macOS 15.
This wrapper compiles the unchanged quantization shader templates from the
installed, pinned MLX headers with the host OS compiler. No MLX arrays, runtime
or GPU operations are used: inputs, output allocation and scheduling are Torch.
"""

from __future__ import annotations

import hashlib
from importlib.metadata import distribution, version
from pathlib import Path
from typing import Any


class RuntimeAffineKernel:
    """The affine_qmm_t interface used by Transformers MetalLinear."""

    def __init__(self, torch: Any, bits: int = 4, group_size: int = 64) -> None:
        if bits != 4 or group_size != 64:
            raise ValueError("The runtime shader adapter is validated for 4-bit/group64 only")
        self.torch = torch
        self.bits, self.group_size = bits, group_size
        self.library: Any = None
        self.source, self.source_metadata = self.build_source(bits, group_size)

    @staticmethod
    def build_source(bits: int = 4, group_size: int = 64) -> tuple[str, dict[str, Any]]:
        from torch.utils._cpp_embed_headers import _embed_headers

        try:
            package = distribution("mlx")
        except Exception as exc:
            raise RuntimeError("MLX headers are missing; rerun scripts/setup/transformers.sh") from exc
        include = Path(package.locate_file("mlx/include"))
        if not (include / "mlx/backend/metal/kernels/quantized.h").is_file():
            raise FileNotFoundError(f"MLX shader headers are missing at {include}")
        lines = [
            '#include "mlx/backend/metal/kernels/utils.h"\n',
            '#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"\n',
            '#include "mlx/backend/metal/kernels/quantized_utils.h"\n',
            '#include "mlx/backend/metal/kernels/quantized.h"\n',
        ]
        for suffix, scalar in (("fp32", "float"), ("fp16", "float16_t"), ("bf16", "bfloat16_t")):
            lines.append(f'instantiate_kernel("lab_qmv_{suffix}", affine_qmv, {scalar}, {group_size}, {bits}, false)\n')
            for aligned in ("true", "false"):
                lines.append(f'instantiate_kernel("lab_qmm_{suffix}_{aligned}", affine_qmm_t, {scalar}, {group_size}, {bits}, {aligned}, false)\n')
        source = _embed_headers(lines, [include], set())
        return source, {
            "implementation": "torch.mps.compile_shader; upstream MLX affine qmv/qmm templates",
            "source_package": f"mlx=={version('mlx')}",
            "source_url": f"https://github.com/ml-explore/mlx/tree/v{version('mlx')}/mlx/backend/metal/kernels",
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "source_bytes": len(source.encode()),
            "no_mlx_runtime": True,
        }

    def compile(self) -> None:
        if self.library is None:
            self.library = self.torch.mps.compile_shader(self.source)

    def affine_qmm_t(self, inputs: Any, weight: Any, scales: Any, biases: Any,
                     group_size: int, bits: int) -> Any:
        if (bits, group_size) != (self.bits, self.group_size):
            raise ValueError("Quantization parameters differ from the compiled shader")
        torch = self.torch
        if inputs.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise TypeError(f"Unsupported affine input dtype: {inputs.dtype}")
        self.compile()
        x = inputs.contiguous()
        k, n = x.shape[-1], weight.shape[0]
        m = x.numel() // k
        if k % 64 or n < 8:
            raise ValueError("Affine shader expects K divisible by 64 and N >= 8")
        output = torch.empty((*x.shape[:-1], n), dtype=x.dtype, device=x.device)
        suffix = {torch.float32: "fp32", torch.float16: "fp16", torch.bfloat16: "bf16"}[x.dtype]
        if m == 1:
            kernel = getattr(self.library, f"lab_qmv_{suffix}")
            # Batch addressing is compiled out; zeros fill unused ABI slots.
            kernel(weight, scales, biases, x, output, k, n,
                   0, 0, 0, 0, 0, 0, 0, 0,
                   threads=(64, (n + 7) // 8, 1), group_size=(64, 1, 1))
        else:
            aligned = "true" if n % 32 == 0 else "false"
            kernel = getattr(self.library, f"lab_qmm_{suffix}_{aligned}")
            kernel(weight, scales, biases, x, output, k, n, m,
                   0, 0, 0, 0, 0, 0, 0, 0,
                   threads=(((n + 31) // 32) * 128, (m + 31) // 32, 1),
                   group_size=(128, 1, 1))
        return output
