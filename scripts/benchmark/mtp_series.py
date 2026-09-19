"""Run or resume alternating baseline/MTP benchmark chunks."""
from _bootstrap import configure_imports
configure_imports()

from inference_lab.benchmarking.mtp_series import main

if __name__ == "__main__":
    raise SystemExit(main())
