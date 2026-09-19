"""Run a fixed-length DFlash or matched autoregressive benchmark."""
from _bootstrap import configure_imports
configure_imports()

from inference_lab.benchmarking.speculative import main

if __name__ == "__main__":
    raise SystemExit(main())
