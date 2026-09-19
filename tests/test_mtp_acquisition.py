"""Verify MTP acquisition guards using a tiny synthetic checkpoint, without GPU/network."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from inference_lab.models import mtp


@pytest.fixture
def mtp_source(monkeypatch):
    import huggingface_hub

    header = json.dumps({"fc.weight": {"dtype": "U32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    weight = len(header).to_bytes(8, "little") + header + bytes(4)
    payloads = {
        "config.json": b'{"model_type":"qwen3_5_mtp","quantization":{"bits":4,"group_size":64}}',
        "model.safetensors": weight,
        "model.safetensors.index.json": b'{"weight_map":{"fc.weight":"model.safetensors"}}',
        "README.md": b"Native MTP head\n",
        ".gitattributes": b"*.safetensors filter=lfs\n",
    }
    checksum = hashlib.sha256(weight).hexdigest()
    monkeypatch.setattr(mtp, "MTP_WEIGHT_BYTES", len(weight))
    monkeypatch.setattr(mtp, "MTP_WEIGHT_SHA256", checksum)
    siblings = [SimpleNamespace(rfilename=name, size=len(payload),
                               lfs=SimpleNamespace(sha256=checksum) if name == "model.safetensors" else None)
                for name, payload in payloads.items()]
    info = SimpleNamespace(sha=mtp.MTP_REVISION, siblings=siblings)
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: SimpleNamespace(model_info=lambda *args, **kwargs: info))
    calls = []

    def snapshot(repo_id, **kwargs):
        calls.append((repo_id, kwargs))
        for name, payload in payloads.items():
            (Path(kwargs["local_dir"]) / name).write_bytes(payload)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)
    return payloads, info, calls, checksum


def test_mtp_download_is_pinned_and_uses_only_head_files(mtp_source, tmp_path):
    _, _, calls, checksum = mtp_source
    manifest = mtp.MTPDownloader(mtp.MTPDownloadConfig(output_dir=tmp_path)).run()
    assert calls[0][0] == mtp.MTP_ID
    assert calls[0][1]["revision"] == mtp.MTP_REVISION
    assert set(calls[0][1]["allow_patterns"]) == {
        "config.json", "model.safetensors", "model.safetensors.index.json", "README.md", ".gitattributes"}
    assert manifest["tensor_dtypes"] == ["U32"]
    assert manifest["quantization"]["bits"] == 4
    assert manifest["tensor_count"] == 1
    saved_weight = next(item for item in manifest["files"] if item["path"] == "model.safetensors")
    assert saved_weight["sha256"] == saved_weight["source_lfs_sha256"] == checksum
    assert json.loads((tmp_path / "download-manifest.json").read_text()) == manifest


@pytest.mark.parametrize("directory", ["qwen3.5-9b-mlx-4bit", "qwen3.5-9b-dflash"])
def test_mtp_cannot_overwrite_other_models(directory):
    with pytest.raises(ValueError, match="must not overwrite"):
        mtp.MTPDownloader(mtp.MTPDownloadConfig(output_dir=mtp.PROJECT_ROOT / "models" / directory)).run()


def test_mtp_rejects_changed_source_before_download(mtp_source, tmp_path):
    _, info, calls, _ = mtp_source
    next(item for item in info.siblings if item.rfilename == "model.safetensors").lfs.sha256 = "0" * 64
    with pytest.raises(RuntimeError, match="source metadata"):
        mtp.MTPDownloader(mtp.MTPDownloadConfig(output_dir=tmp_path)).run()
    assert not calls
    assert not (tmp_path / "download-manifest.json").exists()


def test_mtp_rejects_corruption_without_completion_manifest(mtp_source, tmp_path):
    payloads, _, _, _ = mtp_source
    payloads["model.safetensors"] = payloads["model.safetensors"][:-1] + b"\x01"
    with pytest.raises(RuntimeError, match="Source SHA256 mismatch"):
        mtp.MTPDownloader(mtp.MTPDownloadConfig(output_dir=tmp_path)).run()
    assert not (tmp_path / "download-manifest.json").exists()
