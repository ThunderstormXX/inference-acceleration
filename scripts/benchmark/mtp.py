"""Run native MTP or its matched MLX-VLM autoregressive baseline."""
from _bootstrap import configure_imports
configure_imports()

from inference_lab.benchmarking.mtp import main

if __name__ == "__main__":
    raise SystemExit(main())
