"""Capture sequential real-time autoregressive and native MTP token events."""
from pathlib import Path
import sys

script_directory = Path(__file__).resolve().parent
sys.path[:] = [entry for entry in sys.path if Path(entry or ".").resolve() != script_directory]
sys.path.insert(0, str(script_directory.parents[1] / "src"))

from inference_lab.visualization.trace import main

if __name__ == "__main__":
    raise SystemExit(main())
