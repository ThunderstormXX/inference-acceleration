"""Download exactly the first 100 complete training rows, not the full dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_ID = "ThunderstormXXL/deepscaler-teacher-sft-vllm-official-40k"
DATASET_REVISION = "6f0fadbf6b495cdea0e4bd83c4b64c7e36a992a3"
DATASET_SLICE_SHA256 = "359f60d6d3c6a2a93d5f4a1cc2e7f1ec13ac3e10a1f83b6ad5c21bab4e8a5f54"


@dataclass(frozen=True)
class DatasetDownloadConfig:
    repo_id: str = DATASET_ID
    revision: str = DATASET_REVISION
    config_name: str = "default"
    split: str = "train"
    count: int = 100
    output_path: Path = PROJECT_ROOT / "data/raw/deepscaler-100.jsonl"


class DatasetSliceDownloader:
    """Fetch a bounded Dataset Viewer slice with revision and integrity checks."""

    def __init__(self, config: DatasetDownloadConfig) -> None:
        self.config = config

    @staticmethod
    def _write_atomic(path: Path, payload: bytes) -> None:
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def _validate_rows(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        if payload.get("partial"):
            raise RuntimeError("Dataset Viewer reports a partial dataset conversion")
        rows = payload.get("rows", [])
        if len(rows) != self.config.count:
            raise RuntimeError(f"Requested {self.config.count} rows, received {len(rows)}")
        originals = []
        for expected_index, item in enumerate(rows):
            if item.get("row_idx") != expected_index:
                raise RuntimeError(f"Unexpected source row index at position {expected_index}")
            if item.get("truncated_cells"):
                raise RuntimeError(f"Dataset Viewer truncated row {expected_index}; refusing partial chains")
            row = item.get("row")
            if not isinstance(row, dict):
                raise RuntimeError(f"Invalid row {expected_index}")
            for field in ("problem", "response"):
                if not isinstance(row.get(field), str) or not row[field].strip():
                    raise RuntimeError(f"Row {expected_index} has no non-empty {field!r}")
            originals.append(row)
        return originals

    def run(self) -> dict[str, Any]:
        import requests
        from huggingface_hub import HfApi
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        if not 1 <= self.config.count <= 100:
            raise ValueError("count must be between 1 and 100; this task never downloads the full dataset")
        info = HfApi().dataset_info(self.config.repo_id, revision=self.config.revision)
        if not info.sha:
            raise RuntimeError("Hugging Face did not return a dataset commit SHA")
        with requests.Session() as session:
            session.mount("https://", HTTPAdapter(max_retries=Retry(
                total=4, backoff_factor=1, status_forcelist=(429, 500, 502, 503, 504)
            )))
            response = session.get(
                "https://datasets-server.huggingface.co/rows",
                params={
                    "dataset": self.config.repo_id,
                    "config": self.config.config_name,
                    "split": self.config.split,
                    "offset": 0,
                    "length": self.config.count,
                },
                timeout=(15, 120),
            )
            response.raise_for_status()
            viewer_revision = response.headers.get("x-revision")
            if viewer_revision != info.sha:
                raise RuntimeError(
                    f"Viewer revision {viewer_revision!r} differs from pinned repository revision {info.sha!r}. "
                    "The Viewer serves the current dataset revision; refusing to change the benchmark sample. "
                    "Keep the previously downloaded JSONL and manifest, or explicitly select a new --revision."
                )
            payload = response.json()
            rows = self._validate_rows(payload)
            source_url = response.url
            response_bytes = len(response.content)
        serialized = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8")
        checksum = hashlib.sha256(serialized).hexdigest()
        if (self.config.repo_id == DATASET_ID and info.sha == DATASET_REVISION
                and self.config.config_name == "default" and self.config.split == "train"
                and self.config.count == 100 and checksum != DATASET_SLICE_SHA256):
            raise RuntimeError("The pinned first-100-row slice has an unexpected SHA256; refusing dataset drift")
        output = self.config.output_path.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        manifest_path = output.with_suffix(".manifest.json")
        manifest = {
            "schema_version": 1,
            "repo_id": self.config.repo_id,
            "config": self.config.config_name,
            "split": self.config.split,
            "source_revision": info.sha,
            "requested_revision": self.config.revision,
            "viewer_revision": viewer_revision,
            "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
            "method": "Hugging Face Dataset Viewer /rows; no full dataset or parquet shard downloaded",
            "source_url": source_url,
            "offset": 0,
            "count": len(rows),
            "source_row_indices": list(range(len(rows))),
            "num_rows_total": payload.get("num_rows_total"),
            "features": payload.get("features"),
            "response_bytes": response_bytes,
            "jsonl_bytes": len(serialized),
            "sha256": checksum,
            "output_path": str(output),
        }
        manifest_path.unlink(missing_ok=True)
        self._write_atomic(output, serialized)
        self._write_atomic(manifest_path, (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
        print(f"Dataset ready: exactly {len(rows)} complete {self.config.split} rows -> {output}", flush=True)
        print(f"Downloaded {response_bytes:,} bytes; SHA256 {manifest['sha256']}", flush=True)
        print(f"Manifest: {manifest_path}", flush=True)
        return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DATASET_ID)
    parser.add_argument("--revision", default=DATASET_REVISION)
    parser.add_argument("--config-name", default="default")
    parser.add_argument("--split", default="train")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--output-path", type=Path, default=DatasetDownloadConfig.output_path)
    args = parser.parse_args()
    DatasetSliceDownloader(DatasetDownloadConfig(**vars(args))).run()


if __name__ == "__main__":
    main()
