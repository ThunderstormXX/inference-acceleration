"""Run the requested tasks sequentially, preserving logs and failed results."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


class ExperimentSuite:
    def __init__(self, skip_downloads: bool, benchmark_args: list[str], backends=None,
                 awake=False, mlx_wired_memory=False, wait_for_gpu=False):
        self.skip_downloads = skip_downloads
        self.benchmark_args = benchmark_args
        self.backends = backends or ["transformers", "vllm", "mlx", "mlx_vlm"]
        self.awake = awake
        self.mlx_wired_memory = mlx_wired_memory
        self.wait_for_gpu = wait_for_gpu

    def run(self):
        if self.wait_for_gpu:
            lock = ROOT / "artifacts/gpu.lock"
            lock.parent.mkdir(parents=True, exist_ok=True)
            print("Waiting for the current GPU benchmark to finish", flush=True)
            with lock.open("a") as stream:
                fcntl.flock(stream, fcntl.LOCK_EX)
        tasks = [] if self.skip_downloads else ["download/model", "download/dataset"]
        tasks += [f"benchmark/{backend}" for backend in self.backends] + ["report/compare"]
        failures = []
        for task in tasks:
            command = ["bash", str(ROOT / "scripts" / f"{task}.sh")]
            if task.startswith("benchmark/"):
                backend = task.split("/")[1]
                if self.awake:
                    command = ["bash", str(ROOT / "scripts/benchmark/awake.sh"),
                               "--backend", backend]
                command += self.benchmark_args
                if self.mlx_wired_memory and backend in ("mlx", "mlx_vlm"):
                    command += ["--wired-memory"]
            log = ROOT / "artifacts/logs" / f"suite-{task.replace('/', '-')}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            print(f"\nRunning {task}; log: {log}", flush=True)
            with log.open("w") as output:
                process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, text=True, bufsize=1)
                for line in process.stdout:
                    print(line, end="", flush=True)
                    output.write(line)
                    output.flush()
                code = process.wait()
            if code:
                failures.append(task)
                if task.startswith("download/"):
                    break
        if failures:
            print("Failed tasks: " + ", ".join(failures), file=sys.stderr)
            return 1
        return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-downloads", action="store_true")
    parser.add_argument("--backends", nargs="+", choices=["transformers", "vllm", "mlx", "mlx_vlm"])
    parser.add_argument("--awake", action="store_true", help="Temporary caffeinate -di per benchmark")
    parser.add_argument("--mlx-wired-memory", action="store_true", help="Enable wired memory only for MLX/MLX-VLM")
    parser.add_argument("--wait-for-gpu", action="store_true", help="Wait for an existing benchmark before starting")
    args, benchmark_args = parser.parse_known_args()
    raise SystemExit(ExperimentSuite(args.skip_downloads, benchmark_args, args.backends,
                                    args.awake, args.mlx_wired_memory, args.wait_for_gpu).run())
