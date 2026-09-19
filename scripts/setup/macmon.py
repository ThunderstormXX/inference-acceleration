"""Fetch a pinned tiny monitor locally and read IOReport metrics without sudo."""

import argparse
import hashlib
import io
import json
import subprocess
import tarfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class MacmonSnapshot:
    VERSION = "0.8.2"
    URL = "https://github.com/vladkens/macmon/releases/download/v0.8.2/macmon-v0.8.2.tar.gz"
    SHA256 = "588d5bde79885ba36f693e5150911c10c3ad208a2e418a3f2aa827ac84a2d973"

    def run(self, samples=3, interval=1000, inspect_only=False):
        if not 1 <= samples <= 10 or not 100 <= interval <= 5000:
            raise ValueError("This bounded diagnostic supports 1..10 samples, 100..5000 ms")
        directory = ROOT / ".cache/tools" / f"macmon-{self.VERSION}"
        directory.mkdir(parents=True, exist_ok=True)
        archive = directory / "release.tar.gz"
        if not archive.exists():
            with urllib.request.urlopen(self.URL, timeout=30) as response:
                archive.write_bytes(response.read())
        data = archive.read_bytes()
        if hashlib.sha256(data).hexdigest() != self.SHA256:
            raise ValueError("Published GitHub release archive checksum mismatch")
        binary = directory / "macmon"
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as bundle:
            candidates = [item for item in bundle.getmembers()
                          if item.isfile() and Path(item.name).name == "macmon"]
            if len(candidates) != 1:
                raise ValueError(f"Expected one executable; archive: {bundle.getnames()}")
            # Write exactly one regular file: no symlinks, archive paths or scripts.
            binary.write_bytes(bundle.extractfile(candidates[0]).read())
            binary.chmod(0o755)
        if inspect_only:
            subprocess.run(["file", str(binary)], check=True)
            subprocess.run([str(binary), "pipe", "--help"], check=True)
            return
        started = datetime.now(timezone.utc)
        command = [str(binary), "pipe", "--samples", str(samples), "--interval", str(interval), "--soc-info"]
        result = subprocess.run(command, capture_output=True, text=True,
                                timeout=samples * interval / 1000 + 20)
        records = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
        report = {"started_utc": started.isoformat(), "version": self.VERSION,
                  "release_url": self.URL, "archive_sha256": self.SHA256,
                  "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
                  "command": command, "returncode": result.returncode,
                  "stderr": result.stderr, "samples": records, "no_sudo": True,
                  "gpu_workload_started": False,
                  "source": "macmon: private macOS IOReport/SMC telemetry; third-party interpretation"}
        output = ROOT / "artifacts/reports" / f"{started.strftime('%Y%m%dT%H%M%SZ')}-macmon.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        print(f"Results: {output}")
        result.check_returncode()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--interval", type=int, default=1000)
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args()
    MacmonSnapshot().run(args.samples, args.interval, args.inspect_only)
