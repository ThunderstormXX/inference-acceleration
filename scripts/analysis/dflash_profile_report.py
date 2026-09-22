"""Render the saved DFlash profile without importing any inference runtime."""
from pathlib import Path
import sys

directory = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != directory]
sys.path.insert(0, str(directory.parents[1] / "src"))

from inference_lab.profiling.report import main

if __name__ == "__main__":
    raise SystemExit(main())
