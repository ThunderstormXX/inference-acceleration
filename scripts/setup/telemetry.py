"""Read-only host telemetry, without model execution or the GPU lock."""

import argparse
from pathlib import Path

from inference_lab.diagnostics.host import HostDiagnostics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--interval", type=float, default=5)
    args = parser.parse_args()
    HostDiagnostics().run(Path(__file__).resolve().parents[2] / "artifacts/reports",
                          samples=args.samples, interval=args.interval)
