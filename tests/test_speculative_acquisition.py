"""Exercise pinned speculative setup/download safeguards without network or GPU."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from inference_lab.environments import speculative
from inference_lab.models import draft


def test_existing_speculative_lock_is_never_recompiled(monkeypatch, tmp_path):
    monkeypatch.setattr(speculative, "PROJECT_ROOT", tmp_path)
    lock = tmp_path / "requirements-speculative.lock"
    original = f"dflash @ git+https://github.com/z-lab/dflash.git@{speculative.DFLASH_REVISION}\n"
    lock.write_text(original)
    setup = speculative.SpeculativeEnvironmentSetup(initialize_lock=True)
    monkeypatch.setattr(setup, "_run", lambda command: pytest.fail("Existing lock must not be recompiled"))
    setup._initialize_lock("uv")
    assert lock.read_text() == original


def test_missing_speculative_lock_requires_explicit_initialization(monkeypatch, tmp_path):
    monkeypatch.setattr(speculative, "PROJECT_ROOT", tmp_path)
    setup = speculative.SpeculativeEnvironmentSetup()
    monkeypatch.setattr(setup, "_run", lambda command: pytest.fail("Compilation must be explicit"))
    with pytest.raises(RuntimeError, match="--initialize-lock"):
        setup._initialize_lock("uv")


def test_draft_cannot_overwrite_target():
    target = draft.PROJECT_ROOT / "models/qwen3.5-9b-mlx-4bit"
    with pytest.raises(ValueError, match="must not overwrite"):
        draft.DraftDownloader(draft.DraftDownloadConfig(output_dir=target)).run()


def test_draft_manifest_records_actual_bf16_precision_and_exact_allowlist(monkeypatch, tmp_path):
    import huggingface_hub

    header = json.dumps({"weight": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]}}).encode()
    weight = len(header).to_bytes(8, "little") + header + b"\x00\x00"
    payloads = {"config.json": b'{"dtype":"bfloat16"}', "model.safetensors": weight,
                "README.md": b"DFlash draft\n", ".gitattributes": b"*.safetensors filter=lfs\n"}
    checksum = hashlib.sha256(weight).hexdigest()
    monkeypatch.setattr(draft, "DRAFT_WEIGHT_BYTES", len(weight))
    monkeypatch.setattr(draft, "DRAFT_WEIGHT_SHA256", checksum)
    siblings = [SimpleNamespace(rfilename=name, size=len(payload),
                               lfs=SimpleNamespace(sha256=hashlib.sha256(payload).hexdigest()))
                for name, payload in payloads.items()]
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: SimpleNamespace(
        model_info=lambda *args, **kwargs: SimpleNamespace(sha=draft.DRAFT_REVISION, siblings=siblings)))
    calls = []

    def snapshot(repo_id, **kwargs):
        calls.append((repo_id, kwargs))
        for name, payload in payloads.items():
            (Path(kwargs["local_dir"]) / name).write_bytes(payload)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)
    manifest = draft.DraftDownloader(draft.DraftDownloadConfig(output_dir=tmp_path)).run()
    assert calls[0][0] == draft.DRAFT_ID
    assert calls[0][1]["revision"] == draft.DRAFT_REVISION
    assert set(calls[0][1]["allow_patterns"]) == set(payloads)
    assert manifest["tensor_dtypes"] == ["BF16"]
    assert "4-bit" not in manifest["format"]
    assert next(item for item in manifest["files"] if item["path"] == "model.safetensors")["sha256"] == checksum
    assert json.loads((tmp_path / "download-manifest.json").read_text()) == manifest
