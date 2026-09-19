"""A GIF replay must be reconstructed from recorded events, never mean rates."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from inference_lab.visualization.replay_logs import BenchmarkReplayBuilder
from test_speculative_report import make_run, mutate_summary, write_rows


class Tokenizer:
    def decode(self, ids, **kwargs):
        return "".join(f"token{x} " for x in ids)


def record(directory, speculative):
    mutate_summary(directory, lambda s: s["config"].update(trace_generation=True))
    rows = [json.loads(line) for line in (directory / "samples.jsonl").read_text().splitlines()]
    for row in rows:
        p, d = row["prefill_seconds"], row["decode_seconds"]
        if not speculative:
            events = [dict(type="commit", t=p + d * i / 5, token_ids=[i], output_count=i + 1)
                      for i in range(6)]
        else:
            events = [dict(type="commit", t=p, token_ids=[0], output_count=1, round=0)]
            for round_, proposed, accepted, emitted, before, fraction in (
                (1, [1, 99], 1, [1, 2], 1, .5), (2, [3, 4], 2, [3, 4, 5], 3, 1.0),
            ):
                t = p + fraction * d
                events += [dict(type="draft", t=t - .02, token_ids=proposed, output_count=before, round=round_),
                           dict(type="commit", t=t, token_ids=emitted, output_count=before + len(emitted),
                                round=round_, proposed_token_ids=proposed, rejected_token_ids=proposed[accepted:],
                                draft_count=2, accepted_count=accepted,
                                verification_completed_t=t - .005, cache_commit_enqueued_t=t)]
            row.update(drafted_tokens=4, accepted_draft_tokens=3, speculative_rounds=2,
                       emitted_draft_tokens=3, emitted_target_tokens=3)
        row["generation_trace"] = dict(schema_version=1, token_ids=list(range(6)), events=events,
            prefill_seconds=p, decode_seconds=d, total_seconds=p + d, eos_token_ids=[5],
            instrumentation={"enabled": True})
    write_rows(directory, rows)


@pytest.fixture
def logs(tmp_path):
    baseline = make_run(tmp_path, "mtp-baseline-logs", family="mtp")
    mtp = make_run(tmp_path, "mtp-logs", family="mtp", speculative=True, factor=2)
    record(baseline, False)
    record(mtp, True)
    return baseline, mtp


def test_selected_trajectory_preserves_every_time_and_proposal(logs):
    baseline, mtp = logs
    original = [json.loads(line) for line in (mtp / "samples.jsonl").read_text().splitlines()][1]
    replay = BenchmarkReplayBuilder(baseline, mtp, 1, Tokenizer()).build()
    assert replay["metadata"]["dataset_index"] == 1
    assert replay["prompt_token_ids"] == [1001, 42]
    assert replay["parity"]["equal"]
    assert replay["eos_token_ids"] == [5]
    for source, enriched in zip(original["generation_trace"]["events"], replay["mtp"]["events"]):
        for key, value in source.items():
            assert enriched[key] == value
    assert replay["mtp"]["events"][1]["token_ids"] == [1, 99]
    assert "token99" in replay["mtp"]["events"][1]["text"]
    assert "token99" not in replay["mtp"]["output_text"]


def test_no_trace_is_not_synthesized_from_aggregate_times(logs):
    baseline, mtp = logs
    rows = [json.loads(line) for line in (baseline / "samples.jsonl").read_text().splitlines()]
    rows[0].pop("generation_trace")
    write_rows(baseline, rows)
    with pytest.raises(ValueError, match="generation_trace"):
        BenchmarkReplayBuilder(baseline, mtp, 0, Tokenizer()).build()


def series_manifest(logs, tmp_path):
    baseline, mtp = logs
    manifest = tmp_path / "manifest.json"
    evidence = {role: {name: {"path": str(directory / name),
                              "sha256": sha256((directory / name).read_bytes()).hexdigest()}
                       for name in BenchmarkReplayBuilder.RAW_FILES}
                for role, directory in (("baseline", baseline), ("mtp", mtp))}
    manifest.write_text(json.dumps({"status": "completed", "merged_runs": {"baseline": str(baseline), "mtp": str(mtp)},
                                    "merged_files": evidence}))
    return manifest


def test_series_uses_exact_saved_merged_paths_and_global_index(logs, tmp_path):
    manifest = series_manifest(logs, tmp_path)
    result = BenchmarkReplayBuilder.from_series(manifest, 1, Tokenizer()).build()
    assert result["metadata"]["dataset_index"] == 1
    with pytest.raises(ValueError, match="missing"):
        BenchmarkReplayBuilder.from_series(manifest, 99, Tokenizer()).build()


def test_replay_rejects_a_timing_edit_outside_measured_request(logs):
    baseline, mtp = logs
    rows = [json.loads(line) for line in (baseline / "samples.jsonl").read_text().splitlines()]
    rows[0]["generation_trace"]["events"][-1]["t"] = 999
    write_rows(baseline, rows)
    with pytest.raises(ValueError, match="chronological"):
        BenchmarkReplayBuilder(baseline, mtp, 0, Tokenizer()).build()


@pytest.mark.parametrize("name", BenchmarkReplayBuilder.RAW_FILES)
def test_series_rejects_changed_merged_file_bytes(logs, tmp_path, name):
    manifest = series_manifest(logs, tmp_path)
    path = logs[0] / name
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="hash/path mismatch"):
        BenchmarkReplayBuilder.from_series(manifest, 1, Tokenizer())


@pytest.mark.parametrize("status", ["running", "failed", "paused_battery"])
def test_series_requires_completed_state(logs, tmp_path, status):
    manifest = series_manifest(logs, tmp_path)
    data = json.loads(manifest.read_text())
    data["status"] = status
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="series must be completed"):
        BenchmarkReplayBuilder.from_series(manifest, 1, Tokenizer())


def test_series_requires_complete_hash_evidence(logs, tmp_path):
    manifest = series_manifest(logs, tmp_path)
    data = json.loads(manifest.read_text())
    del data["merged_files"]["mtp"]["samples.jsonl"]
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="incomplete merged-file evidence"):
        BenchmarkReplayBuilder.from_series(manifest, 1, Tokenizer())


def test_series_rechecks_sources_when_building_after_selection(logs, tmp_path):
    manifest = series_manifest(logs, tmp_path)
    builder = BenchmarkReplayBuilder.from_series(manifest, 1, Tokenizer())
    path = logs[1] / "samples.jsonl"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="hash/path mismatch"):
        builder.build()


def test_injected_tokenizer_is_explicitly_unverified(logs):
    trace = BenchmarkReplayBuilder(*logs, 0, Tokenizer()).build()
    assert trace["metadata"]["tokenizer"]["status"] == "injected_unverified"
    assert trace["metadata"]["tokenizer"]["selection"] == "caller_supplied_tokenizer"


def tokenizer_checkpoint(directory):
    directory.mkdir(parents=True)
    payloads = {"tokenizer.json": b'{"token":"fixture"}',
                "tokenizer_config.json": b'{"tokenizer_class":"Fixture"}',
                "config.json": b'{"model_type":"qwen3_5"}',
                "vocab.json": b'{"word":0}', "chat_template.jinja": b'{{messages}}'}
    records = []
    for name, payload in payloads.items():
        (directory / name).write_bytes(payload)
        records.append({"path": name, "bytes": len(payload), "sha256": sha256(payload).hexdigest()})
    # Listed but deliberately absent: offline rendering must not open any weight file.
    records.append({"path": "model.safetensors", "bytes": 6_000_000_000, "sha256": "f" * 64})
    manifest = directory / "download-manifest.json"
    manifest.write_text(json.dumps({"repo_id": "fixture/qwen", "resolved_revision": "a" * 40, "files": records}))
    return sha256(manifest.read_bytes()).hexdigest()


def point_to_checkpoint(logs, model, manifest_hash):
    for directory in logs:
        mutate_summary(directory, lambda summary: summary.update(model_manifest_sha256=manifest_hash))
        mutate_summary(directory, lambda summary: summary["config"].update(model_path=str(model)))


def fake_tokenizer_loader(monkeypatch):
    calls = []
    def load(path, **kwargs):
        calls.append((path, kwargs))
        return Tokenizer()
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=load)))
    return calls


def test_local_tokenizer_verifies_manifest_assets_and_records_provenance_without_weights(logs, tmp_path, monkeypatch):
    model = tmp_path / "tokenizer-model"
    manifest_hash = tokenizer_checkpoint(model)
    point_to_checkpoint(logs, model, manifest_hash)
    calls = fake_tokenizer_loader(monkeypatch)
    trace = BenchmarkReplayBuilder(*logs, 0).build()
    evidence = trace["metadata"]["tokenizer"]
    assert evidence["status"] == "verified"
    assert evidence["manifest"]["sha256"] == manifest_hash
    assert evidence["manifest"]["resolved_revision"] == "a" * 40
    assert set(evidence["assets"]) == {"tokenizer.json", "tokenizer_config.json", "config.json", "vocab.json", "chat_template.jinja"}
    assert not (model / "model.safetensors").exists()
    assert calls == [(str(model), {"local_files_only": True, "trust_remote_code": False})]


def test_same_size_tokenizer_drift_fails_before_loading(logs, tmp_path, monkeypatch):
    model = tmp_path / "tokenizer-model"
    manifest_hash = tokenizer_checkpoint(model)
    point_to_checkpoint(logs, model, manifest_hash)
    calls = fake_tokenizer_loader(monkeypatch)
    path = model / "vocab.json"
    path.write_bytes(path.read_bytes().replace(b"0", b"1"))
    with pytest.raises(ValueError, match="tokenizer asset hash/size mismatch"):
        BenchmarkReplayBuilder(*logs, 0).build()
    assert not calls


def test_changed_model_manifest_is_not_accepted_as_tokenizer_provenance(logs, tmp_path, monkeypatch):
    model = tmp_path / "tokenizer-model"
    manifest_hash = tokenizer_checkpoint(model)
    point_to_checkpoint(logs, model, manifest_hash)
    calls = fake_tokenizer_loader(monkeypatch)
    path = model / "download-manifest.json"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="manifest differs"):
        BenchmarkReplayBuilder(*logs, 0).build()
    assert not calls


def test_unpinned_override_asset_is_rejected(logs, tmp_path, monkeypatch):
    model = tmp_path / "tokenizer-model"
    manifest_hash = tokenizer_checkpoint(model)
    point_to_checkpoint(logs, model, manifest_hash)
    calls = fake_tokenizer_loader(monkeypatch)
    (model / "added_tokens.json").write_text('{"unexpected":9}')
    with pytest.raises(ValueError, match="Unpinned tokenizer assets"):
        BenchmarkReplayBuilder(*logs, 0).build()
    assert not calls


def test_missing_recorded_model_uses_only_hash_verified_fallback(logs, tmp_path, monkeypatch):
    from inference_lab.visualization import replay_logs
    fallback_root = tmp_path / "new-project"
    fallback = fallback_root / "models/qwen3.5-9b-mlx-4bit"
    manifest_hash = tokenizer_checkpoint(fallback)
    missing = tmp_path / "missing-old-project-model"
    point_to_checkpoint(logs, missing, manifest_hash)
    monkeypatch.setattr(replay_logs, "ROOT", fallback_root)
    calls = fake_tokenizer_loader(monkeypatch)
    trace = BenchmarkReplayBuilder(*logs, 0).build()
    assert trace["metadata"]["tokenizer"]["selection"] == "verified_project_fallback"
    assert trace["metadata"]["tokenizer"]["recorded_model_directory"] == str(missing)
    assert calls[0][0] == str(fallback)
    point_to_checkpoint(logs, missing, "e" * 64)
    with pytest.raises(ValueError, match="manifest differs"):
        BenchmarkReplayBuilder(*logs, 0).build()
    assert len(calls) == 1
