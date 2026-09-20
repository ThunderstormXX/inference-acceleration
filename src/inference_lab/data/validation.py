"""Acquire bounded, disjoint calibration/heldout rows from the pinned teacher dataset."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from .download import DATASET_ID, DATASET_REVISION, DATASET_SLICE_SHA256, DatasetSliceDownloader, PROJECT_ROOT


DEFAULT_SPLIT_SHA256 = {
    "calibration": "36b26ede7a373a8abe2f8238ea691a18a0fcd382c21a1cf916593066d53569c2",
    "heldout": "561a68c8f9330fff1cad93c119b766b0d23d4513f7e91ccba96180168d2bb7a4",
}


@dataclass(frozen=True)
class ValidationDownloadConfig:
    offset: int = 100
    count: int = 10
    test_count: int = 1
    output_dir: Path = PROJECT_ROOT / "artifacts/data/validation"
    existing_path: Path = PROJECT_ROOT / "data/raw/deepscaler-100.jsonl"

    def __post_init__(self):
        if type(self.offset) is not int or self.offset < 100:
            raise ValueError("offset must be an integer >= 100, disjoint from existing indices 0..99")
        if type(self.count) is not int or self.count < 1 or type(self.test_count) is not int or self.test_count < 1:
            raise ValueError("count and test_count must be positive integers")
        if self.count + self.test_count > 100:
            raise ValueError("At most 100 new rows may be requested; full dataset downloads are forbidden")


class ValidationSplitDownloader:
    """Preserve full source fields and add source_row_index for split consumers."""

    def __init__(self, config: ValidationDownloadConfig):
        self.config = config

    @staticmethod
    def _sha(payload):
        return hashlib.sha256(payload).hexdigest()

    def _indices(self):
        c = self.config
        return {"calibration": list(range(c.offset, c.offset + c.count)),
                "heldout": list(range(c.offset + c.count, c.offset + c.count + c.test_count))}

    def _existing(self):
        path = self.config.existing_path.resolve()
        manifest_path = path.with_suffix(".manifest.json")
        data, saved = path.read_bytes(), manifest_path.read_bytes()
        manifest = json.loads(saved)
        checksum = self._sha(data)
        expected = list(range(100))
        if (checksum != DATASET_SLICE_SHA256 or manifest.get("sha256") != checksum
                or manifest.get("repo_id") != DATASET_ID or manifest.get("source_revision") != DATASET_REVISION
                or manifest.get("source_row_indices") != expected or manifest.get("count") != 100
                or manifest.get("config") != "default" or manifest.get("split") != "train"):
            raise ValueError("Existing 100-row dataset/manifest does not match its pinned source; refusing unverified disjointness")
        rows = [json.loads(line) for line in data.splitlines()]
        if len(rows) != 100:
            raise ValueError("Existing source must contain exactly 100 rows")
        return rows, {"path": str(path), "sha256": checksum,
                      "manifest_path": str(manifest_path), "manifest_sha256": self._sha(saved),
                      "source_row_indices": expected}

    def _validate_rows(self, payload):
        expected_indices = self._indices()["calibration"] + self._indices()["heldout"]
        if payload.get("partial"):
            raise RuntimeError("Dataset Viewer conversion is partial; refusing incomplete chains")
        items = payload.get("rows")
        if not isinstance(items, list) or len(items) != len(expected_indices):
            raise RuntimeError(f"Expected exactly {len(expected_indices)} new rows")
        total = payload.get("num_rows_total")
        if type(total) is not int or total < self.config.offset + len(expected_indices):
            raise RuntimeError("Requested source indices exceed the reported dataset length")
        rows = []
        for index, item in zip(expected_indices, items):
            if not isinstance(item, dict) or type(item.get("row_idx")) is not int or item["row_idx"] != index:
                raise RuntimeError(f"Unexpected source row index; expected {index}")
            if item.get("truncated_cells"):
                raise RuntimeError(f"Dataset Viewer truncated row {index}; refusing partial chains")
            row = item.get("row")
            if not isinstance(row, dict) or "source_row_index" in row:
                raise RuntimeError(f"Invalid or conflicting source metadata at row {index}")
            for field in ("problem", "response"):
                if not isinstance(row.get(field), str) or not row[field].strip():
                    raise RuntimeError(f"Source row {index} lacks a complete nonempty {field}")
            rows.append({**row, "source_row_index": index})
        return rows

    def _check_default_digest(self, split, checksum):
        if (self.config.offset, self.config.count, self.config.test_count) == (100, 10, 1):
            if checksum != DEFAULT_SPLIT_SHA256[split]:
                raise RuntimeError(f"Pinned default {split} content SHA256 drift; refusing changed source rows")

    def _reuse(self, output, existing_evidence):
        path = output / "manifest.json"
        if not path.exists():
            if output.exists() and any(output.iterdir()):
                raise ValueError("Nonempty output directory has no completed validation manifest; choose a new directory")
            return None
        manifest = json.loads(path.read_text())
        if (manifest.get("status") != "completed" or manifest.get("source_revision") != DATASET_REVISION
                or manifest.get("repo_id") != DATASET_ID or manifest.get("config") != "default" or manifest.get("split") != "train"
                or manifest.get("requested_revision") != DATASET_REVISION or manifest.get("viewer_revision") != DATASET_REVISION
                or manifest.get("request") != {
                    "offset": self.config.offset, "length": self.config.count + self.config.test_count}
                or manifest.get("existing100") != existing_evidence):
            raise ValueError("Existing validation bundle differs from requested pinned split or source evidence")
        for split, indices in self._indices().items():
            evidence = manifest.get("splits", {}).get(split, {})
            path = output / f"{split}.jsonl"
            content = path.read_bytes()
            lines = content.splitlines(keepends=True)
            rows = [json.loads(line) for line in lines]
            self._check_default_digest(split, self._sha(content))
            if (evidence.get("source_row_indices") != indices or evidence.get("count") != len(indices)
                    or evidence.get("path") != str(path)
                    or evidence.get("sha256") != self._sha(content) or evidence.get("jsonl_bytes") != len(content)
                    or evidence.get("row_sha256") != [self._sha(line) for line in lines]
                    or [row.get("source_row_index") for row in rows] != indices):
                raise ValueError(f"Saved validation {split} failed hash/index integrity checks")
        print(f"Validation data already verified: {path.parent}", flush=True)
        return manifest

    def run(self):
        output = self.config.output_dir.resolve()
        existing_rows, existing_evidence = self._existing()
        # New artifacts must never write alongside or over the original benchmark input.
        if output == self.config.existing_path.resolve().parent or self.config.existing_path.resolve().is_relative_to(output):
            raise ValueError("Validation output must use a separate directory from the existing 100-row dataset")
        indices = self._indices()
        occupied = set(existing_evidence["source_row_indices"])
        if occupied.intersection(indices["calibration"] + indices["heldout"]) or set(indices["calibration"]).intersection(indices["heldout"]):
            raise ValueError("Validation splits overlap each other or the existing benchmark")
        reused = self._reuse(output, existing_evidence)
        if reused is not None:
            return reused
        import requests
        from huggingface_hub import HfApi
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        info = HfApi().dataset_info(DATASET_ID, revision=DATASET_REVISION)
        if info.sha != DATASET_REVISION:
            raise RuntimeError("Hugging Face returned an unexpected pinned dataset revision")
        length = self.config.count + self.config.test_count
        with requests.Session() as session:
            session.mount("https://", HTTPAdapter(max_retries=Retry(total=4, backoff_factor=1,
                                                                   status_forcelist=(429, 500, 502, 503, 504))))
            response = session.get("https://datasets-server.huggingface.co/rows",
                                   params={"dataset": DATASET_ID, "config": "default", "split": "train",
                                           "offset": self.config.offset, "length": length}, timeout=(15, 120))
            response.raise_for_status()
            viewer_revision = response.headers.get("x-revision")
            if viewer_revision != DATASET_REVISION:
                raise RuntimeError(f"Viewer revision {viewer_revision!r} differs from pinned {DATASET_REVISION}; refusing drift")
            payload = response.json()
            rows = self._validate_rows(payload)
            source_url, response_bytes = response.url, len(response.content)
        records = {"calibration": rows[:self.config.count], "heldout": rows[self.config.count:]}
        files, splits = {}, {}
        for split, values in records.items():
            lines = [(json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8") for row in values]
            content = b"".join(lines)
            self._check_default_digest(split, self._sha(content))
            files[f"{split}.jsonl"] = content
            splits[split] = {"path": str(output / f"{split}.jsonl"), "count": len(values),
                             "source_row_indices": indices[split], "jsonl_bytes": len(content),
                             "sha256": self._sha(content), "row_sha256": [self._sha(line) for line in lines]}
        old_problems = {str(row.get("problem", "")).strip() for row in existing_rows}
        calibration_problems = {row["problem"].strip() for row in records["calibration"]}
        manifest = {"schema_version": 1, "status": "completed", "repo_id": DATASET_ID,
                    "config": "default", "split": "train", "source_revision": DATASET_REVISION,
                    "requested_revision": DATASET_REVISION, "viewer_revision": viewer_revision,
                    "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
                    "method": "One bounded Dataset Viewer /rows request; no complete dataset/parquet shard downloaded",
                    "source_url": source_url, "request": {"offset": self.config.offset, "length": length},
                    "response_bytes": response_bytes, "num_rows_total": payload["num_rows_total"],
                    "features": payload.get("features"), "splits": splits, "existing100": existing_evidence,
                    "row_schema": "All original HF row fields plus source_row_index; row_sha256 hashes each UTF-8 JSONL line including newline",
                    "disjointness": {"scope": "source row indices", "existing_source_indices": sorted(occupied),
                                     "calibration_vs_existing": True, "heldout_vs_existing": True,
                                     "calibration_vs_heldout": True},
                    "problem_text_overlap": {"calibration_with_existing_indices": [row["source_row_index"] for row in records["calibration"] if row["problem"].strip() in old_problems],
                                             "heldout_with_existing_indices": [row["source_row_index"] for row in records["heldout"] if row["problem"].strip() in old_problems],
                                             "heldout_with_calibration_indices": [row["source_row_index"] for row in records["heldout"] if row["problem"].strip() in calibration_problems]}}
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".validation-", dir=output.parent))
        try:
            for name, content in files.items():
                DatasetSliceDownloader._write_atomic(temporary / name, content)
            DatasetSliceDownloader._write_atomic(temporary / "manifest.json", (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode())
            os.replace(temporary, output)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        print(f"Validation ready: calibration={indices['calibration']}, heldout={indices['heldout']} -> {output}", flush=True)
        print(f"Downloaded exactly {length} new rows ({response_bytes:,} response bytes), revision {DATASET_REVISION}", flush=True)
        print(f"Manifest: {output / 'manifest.json'}", flush=True)
        return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offset", type=int, default=100)
    parser.add_argument("--count", type=int, default=10, help="Calibration rows")
    parser.add_argument("--test-count", type=int, default=1, help="Heldout rows immediately after calibration")
    parser.add_argument("--output-dir", type=Path, default=ValidationDownloadConfig.output_dir)
    ValidationSplitDownloader(ValidationDownloadConfig(**vars(parser.parse_args()))).run()


if __name__ == "__main__":
    main()
