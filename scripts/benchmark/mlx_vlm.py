from _bootstrap import configure_imports
configure_imports()

from inference_lab.benchmarking.cli import main

if __name__ == "__main__":
    main("mlx-vlm")
