"""Offline replay must select its actual pair and preserve recorded evidence."""
from copy import deepcopy
import gzip
from hashlib import sha256
import json
from pathlib import Path

import pytest

from inference_lab.visualization.replay_sweep import SweepReplayBuilder, main
from test_replay_logs import tokenizer_checkpoint, fake_tokenizer_loader


def measured(speculative, seconds):
    ids = [0, 1, 2, 3]
    if speculative:
        events = [
            {"type": "commit", "t": 0.1, "token_ids": [0], "output_count": 1, "round": 0},
            {"type": "draft", "t": 0.4, "token_ids": [1, 99], "output_count": 1, "round": 1},
            {"type": "commit", "t": 1.0, "token_ids": [1, 2], "output_count": 3, "round": 1,
             "proposed_token_ids": [1, 99], "rejected_token_ids": [99], "draft_count": 2, "accepted_count": 1,
             "verification_completed_t": 0.9, "cache_commit_enqueued_t": 0.95},
            {"type": "draft", "t": 1.3, "token_ids": [99], "output_count": 3, "round": 2},
            {"type": "commit", "t": 1.8, "token_ids": [3], "output_count": 4, "round": 2,
             "proposed_token_ids": [99], "rejected_token_ids": [99], "draft_count": 1, "accepted_count": 0,
             "verification_completed_t": 1.7, "cache_commit_enqueued_t": 1.75}]
    else:
        events = [{"type": "commit", "t": t, "token_ids": [i], "output_count": i + 1}
                  for i, t in enumerate((0.1, 0.8, 2.0, 4.0))]
    row = {"generated_token_ids": ids, "generated_tokens": 4, "decode_tokens": 3,
           "prompt_tokens": 2, "prefill_seconds": 0.2, "decode_seconds": seconds}
    if speculative:
        row.update(drafted_tokens=3, accepted_draft_tokens=1, speculative_rounds=2)
    row["generation_trace"] = {"schema_version": 1, "token_ids": ids.copy(), "events": events,
        "prefill_seconds": 0.2, "decode_seconds": seconds, "total_seconds": 0.2 + seconds,
        "eos_token_ids": [], "instrumentation": {"enabled": True}}
    return row


@pytest.fixture
def sweep(tmp_path, monkeypatch):
    model = tmp_path / "model"
    manifest_hash = tokenizer_checkpoint(model)
    calls = fake_tokenizer_loader(monkeypatch)
    prompts = [{"index": 42, "prompt_tokens": [1001, 42], "problem": "Example math problem"}]
    prompt_path = tmp_path / "prompts.json"
    prompt_path.write_text(json.dumps(prompts))
    vocab_config = {"kind": "mtp_draft_only_vocabulary", "shortlist_size": 3, "target_vocab_size": 248320,
                    "target_manifest_sha256": manifest_hash,
                    "tokenizer_sha256": sha256((model / "tokenizer.json").read_bytes()).hexdigest()}
    vocab_path = tmp_path / "vocab.json"
    vocab_path.write_text(json.dumps(vocab_config))
    backend = {"framework": "mlx-vlm-mtp", "model_path": str(model), "model_type": "qwen3_5",
               "weight_bits": 4, "weight_group_size": 64, "kv_bits": None, "versions": {"mlx": "fixture"},
               "sampling": "greedy", "ignore_eos": True, "batch_size": 1, "fresh_cache_per_request": True,
               "trace_generation": False}
    def run(mode, pair, seconds):
        return {"mode": mode, "pair_id": pair, "prompt_index": 0, "repeat": 0, "role": "paired",
                "block_size": 1 if mode == "ar" else 3, "paired_block_size": 3,
                "sleep": {"sleep_detected": False}, "measurement": measured(mode != "ar", seconds)}
    rows = [run("ar", "stock", 9), run("mtp-stock", "stock", 8),
            run("ar", "selected", 4), run("mtp", "selected", 2)]
    data = {"schema_version": 1, "status": "completed", "protocol": {"trace": True,
        "expected_runs": 4, "tokens": 4, "block_sizes": [3], "prompts": prompts,
        "prompts_source": str(prompt_path), "prompts_sha256": sha256(prompt_path.read_bytes()).hexdigest()},
        "backend": backend, "baseline": {**backend, "framework": "mlx-vlm"}, "runs": rows,
        "draft_vocabulary": {"vocabulary_path": str(vocab_path), "vocabulary_sha256": sha256(vocab_path.read_bytes()).hexdigest(),
                             "shortlist_size": 3, "target_vocab_size": 248320}}
    raw = tmp_path / "raw.json"
    raw.write_text(json.dumps(data))
    return raw, data, calls


def rewrite(sweep):
    path, data, _ = sweep
    path.write_text(json.dumps(data))


def test_build_uses_exact_neighbor_not_stock_control_and_preserves_events(sweep):
    path, data, calls = sweep
    trace = SweepReplayBuilder(path).build()
    assert trace["baseline"]["decode_seconds"] == 4
    assert trace["mtp"]["decode_seconds"] == 2
    assert trace["parity"]["equal"] is True
    assert trace["metadata"]["dataset_index"] == 42
    assert trace["metadata"]["provenance"]["pair_id"] == "selected"
    assert trace["metadata"]["provenance"]["baseline_run_index"] == 2
    assert trace["metadata"]["provenance"]["mtp_run_index"] == 3
    assert "draft shortlist 3/248,320" in trace["metadata"]["method_label"]
    assert trace["metadata"]["tokenizer"]["status"] == "verified"
    for role, index in (("baseline", 2), ("mtp", 3)):
        for original, enriched in zip(data["runs"][index]["measurement"]["generation_trace"]["events"], trace[role]["events"]):
            assert all(enriched[key] == value for key, value in original.items())
    assert "token99" in trace["mtp"]["events"][1]["text"]
    assert "token99" not in trace["mtp"]["output_text"]
    assert len(calls) == 1 and calls[0][1] == {"local_files_only": True, "trust_remote_code": False}
    assert not (Path(data["backend"]["model_path"]) / "model.safetensors").exists()


def test_gzip_source_has_compressed_and_document_hashes(sweep):
    raw, data, _ = sweep
    compressed = raw.with_suffix(".json.gz")
    compressed.write_bytes(gzip.compress(raw.read_bytes()))
    trace = SweepReplayBuilder(compressed).build()
    evidence = trace["metadata"]["provenance"]["raw"]
    assert evidence["sha256"] == sha256(compressed.read_bytes()).hexdigest()
    assert evidence["decompressed_json_sha256"] == sha256(raw.read_bytes()).hexdigest()


@pytest.mark.parametrize("mutate,match", [
    (lambda d: d.update(status="running"), "completed"),
    (lambda d: d["protocol"].update(trace=False), "no recorded trace"),
    (lambda d: d["protocol"].update(expected_runs=5), "run count"),
    (lambda d: d["runs"][3].update(mode="mtp-stock"), "mode=mtp"),
    (lambda d: d["runs"][2].update(pair_id="wrong"), "exactly one AR"),
    (lambda d: d["runs"][2].update(repeat=1), "metadata differs"),
    (lambda d: d["runs"][2]["sleep"].update(sleep_detected=True), "awake-host"),
    (lambda d: d["runs"][3]["measurement"].pop("generation_trace"), "generation_trace"),
    (lambda d: d["runs"][3]["measurement"]["generation_trace"]["events"][-1].update(t=100), "chronological"),
    (lambda d: d["runs"][3]["measurement"].update(prompt_tokens=3), "prompt/output"),
    (lambda d: d["baseline"].update(weight_bits=8), "metadata differs"),
    (lambda d: d["baseline"].update(sampling="random"), "sampling/cache"),
])
def test_invalid_pair_or_recording_rejected(sweep, mutate, match):
    mutate(sweep[1]); rewrite(sweep)
    with pytest.raises(ValueError, match=match):
        SweepReplayBuilder(sweep[0]).build()


def test_rejects_nonadjacent_same_pair_and_duplicate_treatment(sweep):
    path, data, _ = sweep
    data["runs"][1], data["runs"][2] = data["runs"][2], data["runs"][1]
    rewrite(sweep)
    with pytest.raises(ValueError, match="not neighboring"):
        SweepReplayBuilder(path).build()
    data["runs"].append(deepcopy(data["runs"][-1]))
    data["protocol"]["expected_runs"] += 1
    rewrite(sweep)
    with pytest.raises(ValueError, match="exactly one selected"):
        SweepReplayBuilder(path).build()


def test_token_parity_recomputed_from_full_ids_not_a_saved_flag(sweep):
    path, data, _ = sweep
    row = data["runs"][3]["measurement"]
    row["generated_token_ids"][-1] = 6
    row["generation_trace"]["token_ids"][-1] = 6
    row["generation_trace"]["events"][-1]["token_ids"][-1] = 6
    data["runs"][3]["matches_baseline"] = True
    rewrite(sweep)
    with pytest.raises(ValueError, match="output IDs differ"):
        SweepReplayBuilder(path).build()


@pytest.mark.parametrize("source", ["prompts", "vocabulary", "tokenizer", "manifest"])
def test_pinned_asset_changes_rejected(sweep, source):
    path, data, calls = sweep
    selected = {"prompts": data["protocol"]["prompts_source"], "vocabulary": data["draft_vocabulary"]["vocabulary_path"],
                "tokenizer": str(Path(data["backend"]["model_path"]) / "tokenizer.json"),
                "manifest": str(Path(data["backend"]["model_path"]) / "download-manifest.json")}[source]
    target = Path(selected)
    target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(ValueError, match="hash|manifest differs"):
        SweepReplayBuilder(path).build()
    assert calls == []


def test_raw_change_after_selection_and_embedded_prompt_tamper_rejected(sweep):
    path, data, _ = sweep
    builder = SweepReplayBuilder(path)
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="raw file changed"):
        builder.build()
    data["protocol"]["prompts"][0]["problem"] = "Changed"
    rewrite(sweep)
    with pytest.raises(ValueError, match="Embedded prompts differ"):
        SweepReplayBuilder(path).build()


def test_plain_mtp_requires_recorded_manifest_then_loads_pinned_assets(sweep):
    path, data, _ = sweep
    config = json.loads(Path(data["draft_vocabulary"]["vocabulary_path"]).read_text())
    del data["draft_vocabulary"]
    rewrite(sweep)
    with pytest.raises(ValueError, match="no recorded target manifest"):
        SweepReplayBuilder(path).build()
    data["target_manifest_sha256"] = config["target_manifest_sha256"]
    rewrite(sweep)
    trace = SweepReplayBuilder(path).build()
    assert trace["metadata"]["method_label"] == "Native MTP · K=3"
    assert trace["metadata"]["draft_vocabulary"] is None


def test_cli_only_writes_trace_and_protects_raw_source(sweep, tmp_path):
    path, _, _ = sweep
    output = tmp_path / "demo/trace.json"
    assert main(["--input", str(path), "--output", str(output)]) == 0
    assert output.is_file()
    assert list(output.parent.iterdir()) == [output]
    with pytest.raises(SystemExit):
        main(["--input", str(path), "--output", str(path)])
