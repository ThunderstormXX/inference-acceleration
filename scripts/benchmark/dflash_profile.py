"""Run bounded, paired DFlash phase/layer/operation diagnostics."""
from _bootstrap import configure_imports
configure_imports()

from inference_lab.profiling.runner import main

if __name__ == '__main__':
    raise SystemExit(main())
