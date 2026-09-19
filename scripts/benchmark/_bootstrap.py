"""Prevent task filenames such as transformers.py from shadowing libraries."""
from pathlib import Path
import sys


def configure_imports():
    directory = Path(__file__).resolve().parent
    sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != directory]
    sys.path.insert(0, str(directory.parents[1] / "src"))
