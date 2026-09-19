"""Reject silently incomplete Dataset Viewer slices before benchmarking them."""

import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from inference_lab.data.download import DATASET_ID, DATASET_REVISION, DatasetDownloadConfig, DatasetSliceDownloader


class DatasetSliceValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.downloader = DatasetSliceDownloader(DatasetDownloadConfig(count=2))
        self.payload = {
            "partial": False,
            "rows": [
                {"row_idx": index, "row": {"problem": "Question?", "response": "Full teacher chain", "source": "preserved"}, "truncated_cells": []}
                for index in range(2)
            ],
        }

    def test_preserves_original_fields_and_teacher_chain(self) -> None:
        rows = self.downloader._validate_rows(self.payload)
        self.assertEqual(rows, [item["row"] for item in self.payload["rows"]])

    def test_rejects_incomplete_slice(self) -> None:
        self.payload["rows"].pop()
        with self.assertRaisesRegex(RuntimeError, "received 1"):
            self.downloader._validate_rows(self.payload)

    def test_rejects_truncated_teacher_chain(self) -> None:
        self.payload["rows"][0]["truncated_cells"] = ["response"]
        with self.assertRaisesRegex(RuntimeError, "truncated row"):
            self.downloader._validate_rows(self.payload)

    def test_rejects_reordered_source_rows(self) -> None:
        self.payload["rows"].reverse()
        with self.assertRaisesRegex(RuntimeError, "source row index"):
            self.downloader._validate_rows(self.payload)

    def test_rejects_empty_teacher_chain(self) -> None:
        self.payload["rows"][0]["row"]["response"] = "  "
        with self.assertRaisesRegex(RuntimeError, "response"):
            self.downloader._validate_rows(self.payload)

    def test_revision_mismatch_preserves_existing_slice(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "slice.jsonl"
            output.write_text("previous verified sample\n")
            downloader = DatasetSliceDownloader(DatasetDownloadConfig(output_path=output))
            with patch("huggingface_hub.HfApi") as api, patch("requests.Session") as session:
                api.return_value.dataset_info.return_value = SimpleNamespace(sha=DATASET_REVISION)
                response = session.return_value.__enter__.return_value.get.return_value
                response.headers = {"x-revision": "a-new-dataset-commit"}
                with self.assertRaisesRegex(RuntimeError, "refusing to change"):
                    downloader.run()
                api.return_value.dataset_info.assert_called_once_with(DATASET_ID, revision=DATASET_REVISION)
            self.assertEqual(output.read_text(), "previous verified sample\n")

    def test_checksum_drift_preserves_existing_slice(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "slice.jsonl"
            output.write_text("previous verified sample\n")
            downloader = DatasetSliceDownloader(DatasetDownloadConfig(output_path=output))
            with patch("huggingface_hub.HfApi") as api, patch("requests.Session") as session:
                api.return_value.dataset_info.return_value = SimpleNamespace(sha=DATASET_REVISION)
                response = session.return_value.__enter__.return_value.get.return_value
                response.headers = {"x-revision": DATASET_REVISION}
                response.url = "https://datasets-server.huggingface.co/rows"
                response.content = b"stub response"
                response.json.return_value = {
                    "partial": False,
                    "rows": [
                        {"row_idx": index, "row": {"problem": "Wrong question", "response": "Wrong chain"}, "truncated_cells": []}
                        for index in range(100)
                    ],
                }
                with self.assertRaisesRegex(RuntimeError, "unexpected SHA256"):
                    downloader.run()
            self.assertEqual(output.read_text(), "previous verified sample\n")


if __name__ == "__main__":
    unittest.main()
