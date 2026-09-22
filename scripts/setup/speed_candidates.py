"""Download the pinned ZMLX source without changing installed packages."""
from pathlib import Path
import subprocess
ROOT = Path(__file__).resolve().parents[2]
COMMIT = "d8cd0d88d4299ca6821d706f9d5b4a3520d688f5"
def main():
    path = ROOT / ".cache/external/ZMLX"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--no-checkout", "https://github.com/Hmbown/ZMLX.git", str(path)], check=True)
        subprocess.run(["git", "-C", str(path), "checkout", "--detach", COMMIT], check=True)
    head = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(path), "status", "--porcelain"], text=True).strip()
    if head != COMMIT or dirty:
        raise RuntimeError("Existing ZMLX checkout differs; preserve it and resolve before running")
    print(f"ZMLX source ready at {COMMIT}; installed environments unchanged")
if __name__ == "__main__":
    main()
