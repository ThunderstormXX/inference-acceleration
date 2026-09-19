"""Install isolated Transformers, Torch and versioned affine shader sources."""

import argparse
import os
import platform
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYTHON_VERSION = "3.13.5"


class TransformersEnvironmentSetup:
    """Keep Torch 2.10's native kernel ABI separate from other runtimes."""

    def __init__(self, prewarm_kernel=True):
        self.prewarm_kernel = prewarm_kernel
        self.env = dict(os.environ, UV_CACHE_DIR=str(ROOT / ".cache/uv"),
                        UV_PYTHON_INSTALL_DIR=str(ROOT / ".cache/python"),
                        UV_HTTP_TIMEOUT="300", UV_CONCURRENT_DOWNLOADS="4",
                        HF_HOME=str(ROOT / ".cache/huggingface"),
                        TORCH_HOME=str(ROOT / ".cache/torch"), PYTHONUNBUFFERED="1",
                        PYTHONPATH=str(ROOT / "src"))

    def run(self):
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            raise RuntimeError("The native Metal kernel requires Apple Silicon macOS")
        uv = shutil.which("uv")
        if not uv:
            raise RuntimeError("uv is required to install the isolated Transformers environment")
        directory = ROOT / ".venv-transformers"
        python = directory / "bin/python"
        lock = ROOT / "requirements-transformers.lock"
        if not lock.is_file():
            raise RuntimeError(f"Required version lock is missing: {lock}; restore it before setup")

        def run(args):
            subprocess.run(args, env=self.env, cwd=ROOT, check=True)

        if not python.exists():
            run([uv, "venv", "--python", PYTHON_VERSION, str(directory)])
        actual_python = subprocess.check_output(
            [str(python), "-c", "import platform; print(platform.python_version())"],
            env=self.env, cwd=ROOT, text=True,
        ).strip()
        if actual_python != PYTHON_VERSION:
            raise RuntimeError(
                f"{directory} uses Python {actual_python}; this lock requires {PYTHON_VERSION}. "
                "Create the environment with the required interpreter before retrying."
            )
        run([uv, "pip", "sync", "--python", str(python), str(lock)])
        run([uv, "pip", "check", "--python", str(python)])
        print(f"Transformers environment ready: {python}", flush=True)
        if self.prewarm_kernel:
            # CPU-only preparation: compilation/dispatch belongs to the
            # explicit GPU validation task, protected by the shared lock.
            if int(platform.mac_ver()[0].split('.')[0]) < 26:
                run([str(python), "-c", (
                    "from inference_lab.backends.apple.transformers.metal_runtime import RuntimeAffineKernel; "
                    "source, metadata = RuntimeAffineKernel.build_source(); "
                    "print('Runtime shader source prepared:', metadata); "
                    "print('GPU validation: bash scripts/setup/validate_metal.sh')"
                )])
            else:
                run([str(python), "-c", (
                    "import torch; "
                    "from transformers.integrations.metal_quantization import _get_metal_kernel; "
                    "kernel = _get_metal_kernel(); "
                    "assert callable(kernel.affine_qmm_t); "
                    "print('Native Metal library imported; GPU dispatch has not been tested')"
                )])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-kernel-prewarm", action="store_true",
                        help="Install packages without preparing shader sources or importing the native library")
    arguments = parser.parse_args()
    TransformersEnvironmentSetup(prewarm_kernel=not arguments.skip_kernel_prewarm).run()
