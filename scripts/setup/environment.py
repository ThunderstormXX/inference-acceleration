"""Install isolated environments; all caches and managed runtimes stay local."""
import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class EnvironmentSetup:
    def __init__(self, backend):
        self.backend = backend
        self.env = dict(os.environ, UV_CACHE_DIR=str(ROOT / ".cache/uv"),
                        UV_PYTHON_INSTALL_DIR=str(ROOT / ".cache/python"),
                        UV_HTTP_TIMEOUT="300", UV_CONCURRENT_DOWNLOADS="4")

    def run(self):
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            raise RuntimeError("This project targets macOS on Apple Silicon")
        uv = shutil.which("uv")
        if not uv:
            raise RuntimeError("Install uv first: https://docs.astral.sh/uv/getting-started/installation/")
        vllm = self.backend == "vllm"
        python_version = "3.12.8" if vllm else "3.13.5"
        directory = ROOT / (".venv-vllm" if vllm else ".venv")
        python = directory / "bin/python"
        lock = ROOT / ("requirements-vllm.lock" if vllm else "requirements-mac.lock")
        if not lock.is_file():
            raise RuntimeError(f"Required version lock is missing: {lock}; restore it before setup")
        def run(args):
            subprocess.run(args, env=self.env, cwd=ROOT, check=True)
        if not python.exists():
            run([uv, "venv", "--python", python_version, str(directory)])
        actual_python = subprocess.check_output(
            [str(python), "-c", "import platform; print(platform.python_version())"],
            env=self.env, cwd=ROOT, text=True,
        ).strip()
        if actual_python != python_version:
            raise RuntimeError(
                f"{directory} uses Python {actual_python}; this lock requires {python_version}. "
                "Create the environment with the required interpreter before retrying."
            )
        # Consume the checked-in freeze as the complete dependency set. Never
        # rewrite it from an environment that may contain unrelated packages.
        run([uv, "pip", "sync", "--python", str(python), str(lock)])
        run([uv, "pip", "check", "--python", str(python)])
        print(f"Environment ready: {python}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["mac", "vllm"], default="mac")
    EnvironmentSetup(parser.parse_args().backend).run()
