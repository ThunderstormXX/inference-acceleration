"""Repeat the first five benchmark inputs under the same finite-run policy."""

import argparse
from datetime import datetime, timezone
import subprocess

from inference_lab.core.config import ROOT
from inference_lab.core.io import write_json


class RecheckSuite:
    BACKENDS = ("transformers", "vllm", "mlx", "mlx_vlm")

    def __init__(self):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        self.log_directory = ROOT / "artifacts/logs" / f"{stamp}-recheck5"

    def _run_task(self, name: str, command: list[str]) -> int:
        log = self.log_directory / f"{name}.log"
        print(f"Running {name}; log: {log}", flush=True)
        with log.open("w") as output:
            process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in process.stdout:
                print(line, end="", flush=True)
                output.write(line)
                output.flush()
            return process.wait()

    def run(self) -> int:
        self.log_directory.mkdir(parents=True)
        manifest = {"label": "recheck5", "count": 5, "max_new_tokens": 128,
                    "backends": list(self.BACKENDS), "tasks": [], "status": "running"}
        manifest_path = self.log_directory / "manifest.json"
        write_json(manifest_path, manifest)
        for backend in self.BACKENDS:
            command = ["bash", str(ROOT / "scripts/benchmark/awake.sh"),
                       "--backend", backend, "--count", "5", "--max-new-tokens", "128",
                       "--prompt-mode", "problem", "--warmup", "1", "--label", "recheck5"]
            if backend in ("mlx", "mlx_vlm"):
                command += ["--wired-memory"]
            code = self._run_task(backend, command)
            manifest["tasks"].append({"task": backend, "command": command, "exit_code": code})
            write_json(manifest_path, manifest)
        for name in ("recheck", "compare"):
            command = ["bash", str(ROOT / "scripts/report" / f"{name}.sh")]
            code = self._run_task(f"report-{name}", command)
            manifest["tasks"].append({"task": f"report/{name}", "command": command, "exit_code": code})
            write_json(manifest_path, manifest)
        failed = any(task["exit_code"] for task in manifest["tasks"])
        manifest["status"] = "failed" if failed else "completed"
        write_json(manifest_path, manifest)
        print(f"Recheck {manifest['status']}: {manifest_path}", flush=True)
        return int(failed)


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    return RecheckSuite().run()
