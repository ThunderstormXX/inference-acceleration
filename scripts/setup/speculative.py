#!/usr/bin/env python3
"""Bootstrap the isolated official DFlash runtime with the host Python."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from inference_lab.environments.speculative import main


if __name__ == "__main__":
    main()
