"""Compare scoped kernels with the unchanged resident Qwen target."""
from _bootstrap import configure_imports
configure_imports()
from inference_lab.optimizations.kernel_sweep import main
if __name__ == "__main__":
    raise SystemExit(main())
