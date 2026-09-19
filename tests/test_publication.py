"""CPU-only checks of public arithmetic, trace evidence and privacy boundaries."""
import hashlib
import json
from pathlib import Path

import pytest

from inference_lab.benchmarking.publication import PublicationExporter, main, object_digest
from inference_lab.benchmarking.speculative_report import SpeculativeReport
from inference_lab.visualization.recording import recording_policy
from test_speculative_report import make_run, mutate_summary, write_rows


def read_rows(directory):
    return [json.loads(line) for line in (directory / "samples.jsonl").read_text().splitlines()]


@pytest.fixture
def exporter(tmp_path):
    baseline = make_run(tmp_path / "runs", "mtp-baseline", family="mtp")
    speculative = make_run(tmp_path / "runs", "mtp-k3", speculative=True, factor=2, family="mtp")
    return PublicationExporter(baseline, speculative, tmp_path / "public" / "paired")


def instrument(directory, speculative=False):
    mutate_summary(directory, lambda s: s["config"].update(trace_generation=True))
    rows = read_rows(directory)
    for row in rows:
        prefill, decode = row["prefill_seconds"], row["decode_seconds"]
        events = [{"type": "commit", "t": prefill, "token_ids": [0], "output_count": 1}]
        if speculative:
            row.update(drafted_tokens=4, accepted_draft_tokens=2, speculative_rounds=3,
                       emitted_draft_tokens=2, emitted_target_tokens=4)
            for rnd, (proposals, committed, accepted, prefix) in enumerate(
                    [([1, 99], [1, 2], 1, 1), ([3], [3, 4], 1, 3), ([99], [5], 0, 5)], 1):
                t = prefill + rnd * decode / 3
                events.extend([
                    {"type": "draft", "t": t - decode / 6, "token_ids": proposals, "output_count": prefix, "round": rnd},
                    {"type": "commit", "t": t, "token_ids": committed, "output_count": prefix + len(committed),
                     "accepted_count": accepted, "draft_count": len(proposals), "round": rnd,
                     "proposed_token_ids": proposals, "rejected_token_ids": proposals[accepted:],
                     "verification_completed_t": t, "cache_commit_enqueued_t": t}])
        else:
            events.extend({"type": "commit", "t": prefill + i * decode / 5,
                           "token_ids": [i], "output_count": i + 1} for i in range(1, 6))
        row["generation_trace"] = {"schema_version": 1, "token_ids": list(range(6)), "events": events,
                                   "eos_token_ids": [3], "prefill_seconds": prefill,
                                   "decode_seconds": decode, "total_seconds": prefill + decode,
                                   "instrumentation": recording_policy(speculative)}
    write_rows(directory, rows)


def test_recomputed_arithmetic_pooled_and_paired_are_distinct(exporter):
    rows = read_rows(exporter.speculative)
    rows[1]["decode_seconds"] *= 2  # Per-prompt speedups become 2 and 1, not uniform.
    write_rows(exporter.speculative, rows)
    report = exporter.build()
    assert report["status"] == "completed"
    old, new = [report["runs"][r] for r in ("baseline", "speculative")]
    assert old["statistics"]["decode"]["mean_tokens_per_second"] == 15
    assert new["statistics"]["decode"]["mean_tokens_per_second"] == 20
    assert old["statistics"]["decode"]["sample_variance_tokens_per_second"] == 50
    assert report["speedup"]["decode"]["ratio_of_mean_rates"] == pytest.approx(4 / 3)
    assert report["speedup"]["decode"]["ratio_of_aggregate_rates"] == 1.5
    assert report["paired_speedup"]["statistics"]["decode"]["mean"] == 1.5
    assert report["paired_speedup"]["statistics"]["decode"]["sample_variance"] == .5
    assert old["samples"][0]["end_to_end_output_tokens_per_second"] == pytest.approx(6 / (.04 + .5))
    assert old["statistics"]["end_to_end"]["aggregate_tokens_per_second"] == pytest.approx(12 / .81)
    assert report["acceptance"]["weighted_draft_acceptance_rate"] == pytest.approx(8 / 12)
    assert "ddof=1" in exporter.output.with_suffix(".md").read_text()


def test_privacy_source_hashes_and_immutability(exporter):
    sentinel = "PRIVATE_HOST_SENTINEL"
    def contaminate(s):
        s["environment"].update(pid=123, hostname=sentinel, process_inventory=[sentinel],
                                 packages={**s["environment"]["packages"], sentinel: sentinel})
        s["backend"].update(unreviewed_secret=sentinel,
                            upstream_runtime_sources={"mlx.module": {"path": "/Users/private_name/checkout/.venv/module.py", "sha256": "d" * 64}})
        s["config"]["model_path"] = "/Users/private_name/checkout/models/model"
        s["config"]["dataset_path"] = "/home/private_name/checkout/data/data.jsonl"
    for directory in (exporter.baseline, exporter.speculative):
        mutate_summary(directory, contaminate)
        rows = read_rows(directory)
        for row in rows:
            row["output_text"] = sentinel
            row["processes"] = [sentinel]
        write_rows(directory, rows)
    before = {p: p.read_bytes() for d in (exporter.baseline, exporter.speculative) for p in d.iterdir()}
    report = exporter.build()
    public = exporter.output.with_suffix(".json").read_text()
    assert report["status"] == "completed"
    for forbidden in (sentinel, "private_name", "/Users/", "/home/", '"pid"', '"generated_token_ids"', '"output_text"'):
        assert forbidden not in public
    for path, raw in before.items():
        assert path.read_bytes() == raw
    for role, directory in (("baseline", exporter.baseline), ("speculative", exporter.speculative)):
        for name, evidence in report["runs"][role]["sources"].items():
            assert not Path(evidence["path"]).is_absolute()
            assert evidence["sha256"] == hashlib.sha256(before[directory / name]).hexdigest()


def test_instrumented_trace_eos_uses_commit_time_and_omits_raw_events(exporter):
    instrument(exporter.baseline)
    instrument(exporter.speculative, True)
    report = exporter.build()
    assert report["status"] == "completed", report["errors"]
    assert report["methodology"]["instrumented"] is True
    row = report["runs"]["speculative"]["samples"][0]
    assert row["eos"]["first_eos_index"] == 3
    assert row["eos"]["tokens_through_first_eos"] == 4
    assert row["eos"]["time_to_first_eos_seconds"] == pytest.approx(row["prefill_seconds"] + row["decode_seconds"] * 2 / 3)
    assert row["eos"]["commit_token_start_index"] == 3
    assert row["eos"]["commit_token_end_index"] == 4
    assert row["generation_trace_sha256"] == object_digest(read_rows(exporter.speculative)[0]["generation_trace"])
    assert row["instrumentation"]["observer_overhead_included"] is True
    assert report["runs"]["speculative"]["statistics"]["eos"]["completed_within_budget"] == 2
    public = exporter.output.with_suffix(".json").read_text()
    assert '"events"' not in public and '"generated_token_ids"' not in public
    assert "host reads" in report["methodology"]["timing_overhead"]


@pytest.mark.parametrize("fault", ["partial", "different_trace_config", "missing_trace", "bad_trace", "different_policy"])
def test_fails_closed_for_incomplete_or_uncomparable_evidence(exporter, fault):
    if fault in ("missing_trace", "bad_trace", "different_policy"):
        instrument(exporter.baseline)
        instrument(exporter.speculative, True)
        rows = read_rows(exporter.speculative)
        if fault == "missing_trace":
            del rows[0]["generation_trace"]
        elif fault == "bad_trace":
            rows[0]["generation_trace"]["events"][-1]["output_count"] = 1000
        else:
            rows[0]["generation_trace"]["instrumentation"]["version"] = 2
        write_rows(exporter.speculative, rows)
    elif fault == "partial":
        mutate_summary(exporter.speculative, lambda s: s.update(status="running"))
    else:
        mutate_summary(exporter.speculative, lambda s: s["config"].update(trace_generation=True))
    report = exporter.build()
    assert report["status"] == "failed" and report["errors"]
    assert "runs" not in report
    with pytest.raises(SystemExit) as error:
        main(["--baseline", str(exporter.baseline), "--speculative", str(exporter.speculative), "--output", str(exporter.output)])
    assert error.value.code == 1


def test_parity_mismatch_publishes_qualified_valid_measurements(exporter):
    rows = read_rows(exporter.speculative)
    rows[1]["generated_token_ids"][2] = 100
    write_rows(exporter.speculative, rows)
    report = exporter.build()
    assert report["status"] == "completed" and report["parity_status"] == "failed"
    assert report["unsuitable_for_lossless_claim"] is True
    assert report["parity"]["per_prompt"][1]["first_mismatch_token_position"] == 2
    assert "Output IDs differ" in exporter.output.with_suffix(".md").read_text()
    main(["--baseline", str(exporter.baseline), "--speculative", str(exporter.speculative), "--output", str(exporter.output)])


def test_verified_config_eos_and_unknown_is_not_zero(exporter, tmp_path):
    unknown = exporter.analyze()["runs"]["baseline"]["statistics"]["eos"]
    assert unknown["completed_within_budget"] is None
    model = tmp_path / "models" / "fixture"
    model.mkdir(parents=True)
    raw = b'{"text_config":{"eos_token_id":[3,99]}}'
    (model / "config.json").write_bytes(raw)
    for directory in (exporter.baseline, exporter.speculative):
        def update(s):
            s["config"]["model_path"] = str(model)
            s["model_manifest"] = {"files": [{"path": "config.json", "sha256": hashlib.sha256(raw).hexdigest()}]}
        mutate_summary(directory, update)
    report = exporter.analyze()
    assert report["status"] == "completed"
    assert report["runs"]["baseline"]["samples"][0]["eos"]["first_eos_index"] == 3
    assert report["runs"]["baseline"]["samples"][0]["eos"]["time_to_first_eos_seconds"] is None
    (model / "config.json").write_text('{"eos_token_id":0}')
    report = exporter.analyze()
    assert report["status"] == "failed" and "hash mismatch" in report["errors"][0]


def test_detects_mutation_between_validation_and_export(exporter, monkeypatch):
    analyze = SpeculativeReport.analyze
    def change_after_validation(self):
        result = analyze(self)
        with (exporter.baseline / "samples.jsonl").open("a") as stream:
            stream.write("\n")
        return result
    monkeypatch.setattr(SpeculativeReport, "analyze", change_after_validation)
    report = exporter.analyze()
    assert report["status"] == "failed"
    assert "after validation" in report["errors"][0]


def test_single_sample_and_output_suffix_and_raw_protection(tmp_path):
    baseline = make_run(tmp_path / "runs", "mtp-baseline", count=1, family="mtp")
    speculative = make_run(tmp_path / "runs", "mtp-k3", True, count=1, family="mtp")
    exporter = PublicationExporter(baseline, speculative, tmp_path / "public" / "paired.json")
    report = exporter.build()
    assert report["paired_speedup"]["statistics"]["decode"]["sample_variance"] is None
    assert (tmp_path / "public" / "paired.md").is_file()
    with pytest.raises(ValueError, match="outside original"):
        PublicationExporter(baseline, speculative, baseline / "summary")


def test_chunk_provenance_preserves_order_and_verifies_hashes(exporter, tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"private_hostname":"do-not-publish"}')
    for role, directory in (("baseline", exporter.baseline), ("mtp", exporter.speculative)):
        block = tmp_path / "blocks" / role
        block.mkdir(parents=True)
        files = {}
        for name in ("summary.json", "prompts.json", "samples.jsonl"):
            path = block / name
            path.write_bytes((directory / name).read_bytes())
            files[name] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        series = {"label": "abba", "suite_manifest": str(manifest), "process_count": 1,
                  "warmups_per_process": 1, "total_warmup_requests": 1, "model_load_count": 1,
                  "execution_order": ["A0", "B0"], "source_block_runs": [
                      {"id": "A0" if role == "baseline" else "B0", "role": role, "chunk_index": 0,
                       "start_index": 0, "count": 2, "attempt": 1, "run_directory": str(block),
                       "command": "SECRET_MACHINE_COMMAND", "files": files}]}
        mutate_summary(directory, lambda s: s.update(series=series))
    report = exporter.build()
    series = report["runs"]["baseline"]["series"]
    assert series["execution_order"] == ["A0", "B0"] and series["model_load_count"] == 1
    assert series["source_block_runs"][0]["files"]["samples.jsonl"]["sha256"]
    public = exporter.output.with_suffix(".json").read_text()
    assert "SECRET_MACHINE_COMMAND" not in public and "do-not-publish" not in public
    (tmp_path / "blocks" / "baseline" / "samples.jsonl").write_text("changed")
    assert exporter.analyze()["status"] == "failed"


def add_metadata_series(exporter, tmp_path):
    """Mimic the real merger: common software only, rich metadata in 1-row chunks."""
    for role, directory in (("baseline", exporter.baseline), ("mtp", exporter.speculative)):
        summary = json.loads((directory / "summary.json").read_text())
        prompts = json.loads((directory / "prompts.json").read_text())
        rows = read_rows(directory)
        blocks = []
        for i in range(2):
            block = tmp_path / "source-blocks" / role / str(i)
            block.mkdir(parents=True)
            chunk = json.loads(json.dumps(summary))
            chunk["config"].update(count=1, start_index=i)
            chunk.update(completed_samples=1, started_at=f"20260919T120{i}00.000000Z", elapsed_seconds=12.5 + i)
            chunk["environment"].update(os="macOS-15.6.1", machine="arm64", processor="Apple M2 Pro", memory_bytes="17179869184",
                hostname="PRIVATE_MACHINE", pid=12345,
                power_source={"raw": "Now drawing from 'AC Power'\n -InternalBattery-0 (id=SECRET_ID)\t87%; charging; present: true"},
                thermal_state={"name": "nominal", "value": 0, "source": "NSProcessInfo.thermalState"})
            chunk["resources_after"] = {"processes": ["PRIVATE_MACHINE"], "power_source": {
                "raw": "Now drawing from 'Battery Power'\n -InternalBattery-0 (id=SECRET_ID)\t86%; discharging; present: true"},
                "thermal_state": {"name": "fair", "value": 1, "source": "NSProcessInfo.thermalState"}}
            chunk["model_manifest"] = {"repo_id": "example/target", "resolved_revision": "a" * 40,
                "requested_revision": "main", "files": [], "local_dir": "/Users/private_user/models/target"}
            (block / "summary.json").write_text(json.dumps(chunk))
            (block / "prompts.json").write_text(json.dumps(prompts[i:i + 1]))
            write_rows(block, rows[i:i + 1])
            files = {name: {"path": str(block / name), "sha256": hashlib.sha256((block / name).read_bytes()).hexdigest()}
                     for name in ("summary.json", "prompts.json", "samples.jsonl")}
            blocks.append({"id": f"{i:03d}-{role}", "chunk_index": i, "role": role,
                           "start_index": i, "count": 1, "attempt": 1, "run_directory": str(block), "files": files})
        summary["series"] = {"label": "metadata", "process_count": 2,
            "source_block_runs": blocks, "execution_order": ["000-baseline", "000-mtp", "001-mtp", "001-baseline"]}
        summary.pop("model_manifest", None)
        (directory / "summary.json").write_text(json.dumps(summary))


def test_recovers_common_hardware_and_model_with_per_chunk_conditions(exporter, tmp_path):
    add_metadata_series(exporter, tmp_path)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*.json")}
    report = exporter.build()
    assert report["status"] == "completed", report["errors"]
    for run in report["runs"].values():
        assert run["runtime"]["hardware"]["processor"] == "Apple M2 Pro"
        assert run["model"]["repo_id"] == "example/target"
        assert run["model"]["resolved_revision"] == "a" * 40
        assert run["runtime"]["conditions_before"] is None
        provenance = run["series"]["metadata_provenance"]
        assert provenance["verified_chunk_count"] == 2
        assert provenance["hardware_available"] and provenance["target_manifest_available"]
        assert len(provenance["source_summary_files"]) == 2
        for i, block in enumerate(run["series"]["source_block_runs"]):
            assert block["started_at"] == f"20260919T120{i}00.000000Z"
            assert block["elapsed_seconds"] == 12.5 + i
            assert block["conditions_before"]["power_source"]["kind"] == "AC Power"
            assert block["conditions_after"]["power_source"]["kind"] == "Battery Power"
            assert block["conditions_after"]["power_source"]["battery_percent"] == 86
            assert block["conditions_after"]["thermal_state"]["name"] == "fair"
    public = exporter.output.with_suffix(".json").read_text()
    for private in ("PRIVATE_MACHINE", "SECRET_ID", "private_user", '"pid"', '"processes"', "/Users/"):
        assert private not in public
    for path, raw in before.items():
        assert path.read_bytes() == raw


@pytest.mark.parametrize("fault", ["hardware", "target_revision", "manifest_hash"])
def test_rejects_inconsistent_verified_chunk_metadata(exporter, tmp_path, fault):
    add_metadata_series(exporter, tmp_path)
    summary = json.loads((exporter.baseline / "summary.json").read_text())
    block = summary["series"]["source_block_runs"][1]
    source = block["files"]["summary.json"]
    chunk = json.loads(Path(source["path"]).read_text())
    if fault == "hardware":
        chunk["environment"]["processor"] = "Other processor"
    elif fault == "target_revision":
        chunk["model_manifest"]["resolved_revision"] = "b" * 40
    else:
        chunk["model_manifest_sha256"] = "c" * 64
    raw = json.dumps(chunk).encode()
    Path(source["path"]).write_bytes(raw)
    source["sha256"] = hashlib.sha256(raw).hexdigest()  # Hash is valid; semantics are inconsistent.
    (exporter.baseline / "summary.json").write_text(json.dumps(summary))
    report = exporter.build()
    assert report["status"] == "failed" and "runs" not in report
    assert "chunk" in report["errors"][0]


def test_rejects_different_hardware_between_consistent_role_groups(exporter, tmp_path):
    add_metadata_series(exporter, tmp_path)
    path = exporter.speculative / "summary.json"
    summary = json.loads(path.read_text())
    for block in summary["series"]["source_block_runs"]:
        source = block["files"]["summary.json"]
        chunk = json.loads(Path(source["path"]).read_text())
        chunk["environment"]["processor"] = "Another machine"
        raw = json.dumps(chunk).encode()
        Path(source["path"]).write_bytes(raw)
        source["sha256"] = hashlib.sha256(raw).hexdigest()
    path.write_text(json.dumps(summary))
    report = exporter.analyze()
    assert report["status"] == "failed" and "paired source chunk groups" in report["errors"][0]


def rewrite_chunk_file(summary, block_index, name, content):
    source = summary["series"]["source_block_runs"][block_index]["files"][name]
    raw = content.encode()
    Path(source["path"]).write_bytes(raw)
    source["sha256"] = hashlib.sha256(raw).hexdigest()


def test_rejects_altered_merged_timing_even_with_consistent_trace(exporter, tmp_path):
    instrument(exporter.baseline)
    instrument(exporter.speculative, True)
    add_metadata_series(exporter, tmp_path)
    rows = read_rows(exporter.speculative)
    for row in rows:
        for key in ("prefill_seconds", "decode_seconds"):
            row[key] /= 2
        trace = row["generation_trace"]
        for key in ("prefill_seconds", "decode_seconds", "total_seconds"):
            trace[key] /= 2
        for event in trace["events"]:
            for key in ("t", "verification_completed_t", "cache_commit_enqueued_t"):
                if key in event:
                    event[key] /= 2
    write_rows(exporter.speculative, rows)
    report = exporter.build()
    assert report["status"] == "failed" and "runs" not in report
    assert "full sample differs" in report["errors"][0]


def test_rejects_merged_prompt_metadata_change_not_only_token_ids(exporter, tmp_path):
    add_metadata_series(exporter, tmp_path)
    path = exporter.speculative / "prompts.json"
    prompts = json.loads(path.read_text())
    prompts[0]["problem"] = "Changed metadata with unchanged prompt token IDs"
    path.write_text(json.dumps(prompts))
    report = exporter.analyze()
    assert report["status"] == "failed" and "Merged prompt" in report["errors"][0]


@pytest.mark.parametrize("fault", ["missing_chunk", "duplicate_chunk", "wrong_role", "wrong_backend",
                                  "wrong_range", "missing_raw_file", "extra_order_id", "duplicate_order_id",
                                  "wrong_abba_order", "wrong_protocol", "raw_invalid_object"])
def test_rejects_inconsistent_chunk_lineage(exporter, tmp_path, fault):
    add_metadata_series(exporter, tmp_path)
    path = exporter.baseline / "summary.json"
    summary = json.loads(path.read_text())
    series = summary["series"]
    if fault == "missing_chunk":
        series["source_block_runs"].pop()
    elif fault == "duplicate_chunk":
        series["source_block_runs"].append(series["source_block_runs"][0])
    elif fault == "wrong_role":
        series["source_block_runs"][0]["role"] = "mtp"
    elif fault == "wrong_range":
        series["source_block_runs"][0]["count"] = 2
    elif fault == "missing_raw_file":
        del series["source_block_runs"][0]["files"]["samples.jsonl"]
    elif fault == "extra_order_id":
        series["execution_order"].append("unrelated-extra-run")
    elif fault == "duplicate_order_id":
        series["execution_order"].append(series["execution_order"][0])
    elif fault == "wrong_abba_order":
        series["execution_order"][0:2] = reversed(series["execution_order"][0:2])
    elif fault == "raw_invalid_object":
        rewrite_chunk_file(summary, 0, "samples.jsonl", "42\n")
    else:
        source = series["source_block_runs"][0]["files"]["summary.json"]
        chunk = json.loads(Path(source["path"]).read_text())
        chunk["config"]["backend" if fault == "wrong_backend" else "max_new_tokens"] = "mlx-mtp" if fault == "wrong_backend" else 7
        rewrite_chunk_file(summary, 0, "summary.json", json.dumps(chunk))
    path.write_text(json.dumps(summary))
    report = exporter.analyze()
    assert report["status"] == "failed" and report["errors"] and "runs" not in report


def test_rejects_extra_or_overlapping_source_chunk_coverage(exporter, tmp_path):
    add_metadata_series(exporter, tmp_path)
    path = exporter.baseline / "summary.json"
    summary = json.loads(path.read_text())
    block = summary["series"]["source_block_runs"][1]
    original = json.loads(json.dumps(summary))
    for target_index, error in ((0, "Duplicate source chunk prompt coverage"), (2, "Extra source chunk prompt")):
        summary = json.loads(json.dumps(original))
        block = summary["series"]["source_block_runs"][1]
        block["start_index"] = target_index
        chunk = json.loads(Path(block["files"]["summary.json"]["path"]).read_text())
        chunk["config"]["start_index"] = target_index
        rewrite_chunk_file(summary, 1, "summary.json", json.dumps(chunk))
        for name in ("prompts.json", "samples.jsonl"):
            source = block["files"][name]
            values = json.loads(Path(source["path"]).read_text()) if name == "prompts.json" else [json.loads(Path(source["path"]).read_text())]
            values[0]["index"] = target_index
            rewrite_chunk_file(summary, 1, name, json.dumps(values) if name == "prompts.json" else json.dumps(values[0]) + "\n")
        path.write_text(json.dumps(summary))
        report = exporter.analyze()
        assert report["status"] == "failed" and error in report["errors"][0]


def test_genuine_merged_output_mismatch_is_still_usable_and_disclosed(exporter, tmp_path):
    add_metadata_series(exporter, tmp_path)
    rows = read_rows(exporter.speculative)
    rows[0]["generated_token_ids"][2] = 999
    write_rows(exporter.speculative, rows)
    path = exporter.speculative / "summary.json"
    summary = json.loads(path.read_text())
    rewrite_chunk_file(summary, 0, "samples.jsonl", json.dumps(rows[0]) + "\n")
    path.write_text(json.dumps(summary))
    report = exporter.build()
    assert report["status"] == "completed" and report["parity_status"] == "failed"
    assert report["unsuitable_for_lossless_claim"] is True
    assert report["parity"]["per_prompt"][0]["first_mismatch_token_position"] == 2
