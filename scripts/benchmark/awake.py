"""Run one benchmark with temporary macOS idle-sleep assertions."""
import argparse
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


class AwakeBenchmark:
    def run(self, backend: str, arguments: list[str]) -> int:
        environment = dict(os.environ, INFERENCE_CAFFEINATE_ASSERTIONS="di")
        # No -u: do not simulate user activity or explicitly wake the display.
        # caffeinate releases both assertions when the benchmark exits.
        return subprocess.call(
            ["/usr/bin/caffeinate", "-di", "bash",
             str(ROOT / "scripts/benchmark" / f"{backend}.sh"), *arguments],
            cwd=ROOT, env=environment,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True,
                        choices=["mlx", "mlx_vlm", "vllm", "transformers"])
    options, remaining = parser.parse_known_args()
    raise SystemExit(AwakeBenchmark().run(options.backend, remaining))
