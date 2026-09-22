"""Run a bounded live causal n-gram decoding experiment."""
from _bootstrap import configure_imports
configure_imports()

from inference_lab.optimizations.ngram_sweep import main

if __name__ == "__main__":
    raise SystemExit(main())
