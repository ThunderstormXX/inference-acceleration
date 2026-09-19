"""Set up the official DFlash runtime without changing existing benchmark environments."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DFLASH_REVISION = "07ebd93db9f472af339b644bb70221ad8428328a"
PYTHON_VERSION = "3.13.5"


class SpeculativeEnvironmentSetup:
    """Initialize a lock explicitly once; subsequent runs only consume that lock."""

    def __init__(self, initialize_lock: bool = False):
        self.initialize_lock = initialize_lock
        self.directory = PROJECT_ROOT / ".venv-speculative"
        self.python = self.directory / "bin/python"
        self.lock = PROJECT_ROOT / "requirements-speculative.lock"
        self.env = dict(os.environ,
                        UV_CACHE_DIR=str(PROJECT_ROOT / ".cache/uv"),
                        UV_PYTHON_INSTALL_DIR=str(PROJECT_ROOT / ".cache/python"),
                        UV_HTTP_TIMEOUT="300", UV_CONCURRENT_DOWNLOADS="4",
                        HF_HOME=str(PROJECT_ROOT / ".cache/huggingface"),
                        PYTHONUNBUFFERED="1")

    def _run(self, command: list[str]) -> None:
        subprocess.run(command, cwd=PROJECT_ROOT, env=self.env, check=True)

    def _initialize_lock(self, uv: str) -> None:
        if self.lock.exists():
            return
        if not self.initialize_lock:
            raise RuntimeError("requirements-speculative.lock is missing; initialize explicitly with --initialize-lock")
        with tempfile.NamedTemporaryFile(dir=PROJECT_ROOT, prefix=".requirements-speculative-",
                                         suffix=".lock", delete=False) as stream:
            temporary = Path(stream.name)
        try:
            self._run([uv, "pip", "compile", "--python", str(self.python),
                       "--output-file", str(temporary),
                       str(PROJECT_ROOT / "requirements-speculative.in")])
            if DFLASH_REVISION not in temporary.read_text():
                raise RuntimeError("Compiled lock does not preserve the official DFlash commit")
            temporary.replace(self.lock)
        finally:
            temporary.unlink(missing_ok=True)

    def run(self) -> dict:
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            raise RuntimeError("The speculative runtime targets Apple Silicon macOS")
        uv = shutil.which("uv")
        if not uv:
            raise RuntimeError("Install uv before running the speculative environment setup")
        if not self.lock.exists() and not self.initialize_lock:
            raise RuntimeError("requirements-speculative.lock is missing; initialize explicitly with --initialize-lock")
        if not self.python.exists():
            self._run([uv, "venv", "--python", PYTHON_VERSION, str(self.directory)])
        version = subprocess.check_output(
            [str(self.python), "-c", "import platform; print(platform.python_version())"],
            cwd=PROJECT_ROOT, env=self.env, text=True,
        ).strip()
        if version != PYTHON_VERSION:
            raise RuntimeError(f"{self.directory} uses Python {version}; expected {PYTHON_VERSION}")
        self._initialize_lock(uv)
        if DFLASH_REVISION not in self.lock.read_text():
            raise RuntimeError("The speculative lock does not contain the required official DFlash commit")
        self._run([uv, "pip", "sync", "--python", str(self.python), str(self.lock)])
        self._run([uv, "pip", "check", "--python", str(self.python)])
        # Read wheel metadata only: importing dflash.model_mlx would initialize
        # the MLX runtime, which belongs to the separate GPU benchmark task.
        probe = """
import json
from importlib.metadata import distribution, version
runtime = distribution('dflash')
direct = json.loads(runtime.read_text('direct_url.json'))
print(json.dumps({'packages': {name: version(name) for name in ('dflash', 'mlx', 'mlx-lm', 'transformers', 'huggingface-hub')}, 'dflash_source': direct}))
"""
        metadata = json.loads(subprocess.check_output(
            [str(self.python), "-c", probe], cwd=PROJECT_ROOT, env=self.env, text=True,
        ))
        if metadata["dflash_source"].get("vcs_info", {}).get("commit_id") != DFLASH_REVISION:
            raise RuntimeError("Installed DFlash package metadata has the wrong Git commit")
        metadata.update(python=version, environment=str(self.directory),
                        verified_at_utc=datetime.now(timezone.utc).isoformat(),
                        lock_path=str(self.lock), gpu_runtime_imported=False)
        report_dir = PROJECT_ROOT / "artifacts/reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        report = report_dir / f"speculative-environment-{stamp}.json"
        report.write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"Speculative environment ready: {self.python}", flush=True)
        print(f"Verified official DFlash commit: {DFLASH_REVISION}", flush=True)
        print(f"Environment metadata: {report}", flush=True)
        return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initialize-lock", action="store_true",
                        help="Explicitly compile the initial lock only if it does not exist; existing locks are never rewritten")
    SpeculativeEnvironmentSetup(parser.parse_args().initialize_lock).run()


if __name__ == "__main__":
    main()
