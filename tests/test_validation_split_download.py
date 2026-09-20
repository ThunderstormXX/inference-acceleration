"""Bounded calibration/heldout acquisition must preserve source identity and old data."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from inference_lab.data import validation


@pytest.fixture
def source(tmp_path, monkeypatch):
    existing = tmp_path / "old/deepscaler-100.jsonl"
    existing.parent.mkdir()
    data = "".join(json.dumps({"problem": f"original {i}", "response": f"teacher {i}"}) + "\n" for i in range(100)).encode()
    existing.write_bytes(data)
    checksum = sha256(data).hexdigest()
    monkeypatch.setattr(validation, "DATASET_SLICE_SHA256", checksum)
    existing.with_suffix(".manifest.json").write_text(json.dumps({
        "sha256": checksum, "repo_id": validation.DATASET_ID, "source_revision": validation.DATASET_REVISION,
        "config": "default", "split": "train", "source_row_indices": list(range(100)), "count": 100,
    }))
    config = validation.ValidationDownloadConfig(offset=100, count=2, test_count=1,
                                                 output_dir=tmp_path / "new", existing_path=existing)
    payload = {"partial": False, "num_rows_total": 39494,
               "rows": [{"row_idx": i, "truncated_cells": [],
                         "row": {"problem": f"new problem {i}", "response": f"full chain {i}",
                                 "source": "unchanged", "extra": {"arbitrary": [1, 2, 3]}}}
                        for i in range(100, 103)]}
    calls = []
    response = SimpleNamespace(headers={"x-revision": validation.DATASET_REVISION},
                               url="https://datasets-server.huggingface.co/rows?offset=100&length=3",
                               content=json.dumps(payload).encode(), json=lambda: deepcopy(payload),
                               raise_for_status=lambda: None)
    class Session:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def mount(self, *args, **kwargs): pass
        def get(self, url, **kwargs):
            calls.append((url, kwargs))
            return response
    import requests
    import huggingface_hub
    monkeypatch.setattr(requests, "Session", Session)
    info = SimpleNamespace(sha=validation.DATASET_REVISION)
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: SimpleNamespace(dataset_info=lambda *a, **k: info))
    return config, payload, calls, response, info


def test_one_bounded_request_preserves_rows_split_indices_and_existing_files(source):
    config, payload, calls, _, _ = source
    before = {p:p.read_bytes() for p in config.existing_path.parent.iterdir()}
    manifest = validation.ValidationSplitDownloader(config).run()
    assert len(calls) == 1
    assert calls[0][0] == "https://datasets-server.huggingface.co/rows"
    assert calls[0][1]["params"] == {"dataset": validation.DATASET_ID, "config": "default", "split": "train", "offset": 100, "length": 3}
    assert manifest["request"] == {"offset": 100, "length": 3}
    assert manifest["splits"]["calibration"]["source_row_indices"] == [100, 101]
    assert manifest["splits"]["heldout"]["source_row_indices"] == [102]
    for split, positions in (("calibration", [0, 1]), ("heldout", [2])):
        evidence = manifest["splits"][split]
        path = Path(evidence["path"])
        lines = path.read_bytes().splitlines(keepends=True)
        assert evidence["sha256"] == sha256(path.read_bytes()).hexdigest()
        assert evidence["row_sha256"] == [sha256(line).hexdigest() for line in lines]
        assert [json.loads(line) for line in lines] == [{**payload["rows"][i]["row"], "source_row_index": 100+i} for i in positions]
        assert not set(evidence["source_row_indices"]).intersection(range(100))
    assert all(path.read_bytes() == content for path, content in before.items())


def test_verified_rerun_is_offline_and_does_not_rewrite_bundle(source):
    config, _, calls, _, _ = source
    downloader = validation.ValidationSplitDownloader(config)
    first = downloader.run()
    original = {p:p.read_bytes() for p in config.output_dir.iterdir()}
    assert downloader.run() == first
    assert len(calls) == 1
    assert all(path.read_bytes() == content for path, content in original.items())


@pytest.mark.parametrize("corrupt", [
    lambda payload: payload.update(partial=True),
    lambda payload: payload["rows"].pop(),
    lambda payload: payload["rows"].reverse(),
    lambda payload: payload["rows"][0].update(truncated_cells=["response"]),
    lambda payload: payload["rows"][0]["row"].update(response=" "),
    lambda payload: payload["rows"][0]["row"].update(source_row_index=0),
    lambda payload: payload.update(num_rows_total=101),
])
def test_incomplete_or_misindexed_rows_never_publish_a_bundle(source, corrupt):
    config, payload, _, _, _ = source
    corrupt(payload)
    with pytest.raises(RuntimeError):
        validation.ValidationSplitDownloader(config).run()
    assert not config.output_dir.exists()


def test_viewer_revision_drift_is_rejected_before_writing(source):
    config, _, _, response, _ = source
    response.headers["x-revision"] = "f" * 40
    with pytest.raises(RuntimeError, match="differs from pinned"):
        validation.ValidationSplitDownloader(config).run()
    assert not config.output_dir.exists()


def test_api_revision_drift_does_not_fetch_rows(source):
    config, _, calls, _, info = source
    info.sha = "f" * 40
    with pytest.raises(RuntimeError, match="unexpected pinned"):
        validation.ValidationSplitDownloader(config).run()
    assert not calls


def test_saved_split_corruption_fails_without_network_or_repair(source):
    config, _, calls, _, _ = source
    validation.ValidationSplitDownloader(config).run()
    path = config.output_dir / "heldout.jsonl"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises((ValueError, json.JSONDecodeError)):
        validation.ValidationSplitDownloader(config).run()
    assert len(calls) == 1


@pytest.mark.parametrize("kwargs", [{"offset": 99}, {"offset": True}, {"offset": 100.0},
                                   {"count": 0}, {"count": True}, {"test_count": 0},
                                   {"test_count": False}, {"count": 100, "test_count": 1}])
def test_invalid_or_overlapping_split_ranges_are_rejected(kwargs):
    with pytest.raises(ValueError):
        validation.ValidationDownloadConfig(**kwargs)


def test_original_dataset_must_match_known_pin_before_fetch(source):
    config, _, calls, _, _ = source
    config.existing_path.write_bytes(config.existing_path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="pinned source"):
        validation.ValidationSplitDownloader(config).run()
    assert not calls


def test_default_split_digest_guards_reproducibility():
    downloader = validation.ValidationSplitDownloader(validation.ValidationDownloadConfig())
    assert downloader._indices() == {"calibration": list(range(100,110)), "heldout": [110]}
    with pytest.raises(RuntimeError, match="SHA256 drift"):
        downloader._check_default_digest("calibration", "0" * 64)


def test_problem_overlap_is_disclosed_without_silently_changing_requested_indices(source):
    config, payload, _, _, _ = source
    payload["rows"][0]["row"]["problem"] = "original 4"
    payload["rows"][2]["row"]["problem"] = "original 4"
    result = validation.ValidationSplitDownloader(config).run()
    assert result["problem_text_overlap"] == {"calibration_with_existing_indices": [100],
                                             "heldout_with_existing_indices": [102],
                                             "heldout_with_calibration_indices": [102]}
    assert result["splits"]["heldout"]["source_row_indices"] == [102]
