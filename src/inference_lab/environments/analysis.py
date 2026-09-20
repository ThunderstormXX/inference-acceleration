"""Isolated, locked CPU environment for scientific benchmark plots."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[3]

class AnalysisEnvironmentSetup:
    def __init__(self, initialize_lock=False):
        self.initialize_lock = initialize_lock
    def run(self):
        uv = shutil.which("uv")
        if not uv:
            raise RuntimeError("Install uv before setting up the analysis environment")
        env = dict(os.environ, UV_CACHE_DIR=str(ROOT / ".cache/uv"))
        python = ROOT / ".venv-analysis/bin/python"
        lock = ROOT / "requirements-analysis.lock"
        if not lock.exists() and not self.initialize_lock:
            raise RuntimeError("Missing analysis lock; use --initialize-lock once")
        def run(args):
            subprocess.run([uv, *args], cwd=ROOT, env=env, check=True)
        if not python.exists():
            run(["venv", "--python", str(ROOT / ".venv/bin/python"), str(python.parents[1])])
        if not lock.exists():
            run(["pip", "compile", "--python", str(python), "requirements-analysis.in", "--no-header", "--output-file", str(lock)])
        run(["pip", "sync", "--python", str(python), str(lock)])
        run(["pip", "check", "--python", str(python)])
        print(f"Analysis environment: {python}")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initialize-lock", action="store_true")
    AnalysisEnvironmentSetup(parser.parse_args().initialize_lock).run()

if __name__ == "__main__":
    main()
