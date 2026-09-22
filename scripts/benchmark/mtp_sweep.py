"""Run the bounded shared-target MTP depth sweep."""
from _bootstrap import configure_imports
configure_imports()

from inference_lab.optimizations.mtp_sweep import main

if __name__ == "__main__":
    raise SystemExit(main())
