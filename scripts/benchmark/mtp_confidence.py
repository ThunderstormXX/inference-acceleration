from _bootstrap import configure_imports
configure_imports()
from inference_lab.experiments.confidence.runner import main
if __name__ == '__main__':
    raise SystemExit(main())
