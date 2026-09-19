"""No GPU or network: exercise suite ordering, source integrity and resumable evidence."""
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path

import pytest

from inference_lab.benchmarking.metrics import aggregate
from inference_lab.benchmarking.mtp_series import MTPSeriesConfig, MTPSeriesRunner
from inference_lab.benchmarking.speculative_report import SpeculativeReport


PACKAGES = {"mlx": "0.32.2", "mlx-metal": "0.32.2", "mlx-lm": "0.31.3", "mlx-vlm": "0.7.1"}
FINGERPRINT = {"project_files": {"runner.py": "a" * 64},
               "runtime": {"python": "3.13.5", "packages": PACKAGES},
               "inputs": {"dataset_sha256": "b" * 64, "model_manifest_sha256": "c" * 64,
                          "draft_manifest_sha256": "d" * 64}}


def raw_trace(row, speculative):
    ids = row["generated_token_ids"]
    first = {"type": "commit", "t": 0.5, "round": 0, "token_ids": ids[:1], "output_count": 1}
    events = [first]
    if speculative:
        proposed = ids[1:]
        events += [{"type": "draft", "t": 1.1, "round": 1, "token_ids": proposed, "output_count": 1},
                   {"type": "commit", "t": 2.0, "round": 1, "token_ids": proposed,
                    "output_count": len(ids), "accepted_count": len(proposed), "draft_count": len(proposed),
                    "proposed_token_ids": proposed, "rejected_token_ids": [],
                    "verification_completed_t": 1.8, "cache_commit_enqueued_t": 1.9}]
    else:
        events += [{"type": "commit", "t": 1.0 + i * 0.1, "round": 0,
                    "token_ids": [token], "output_count": i + 1} for i, token in enumerate(ids[1:], 1)]
    return {"schema_version": 1, "token_ids": ids, "events": events,
            "prefill_seconds": 1.0, "decode_seconds": 2.0, "total_seconds": 3.0,
            "eos_token_ids": [999], "instrumentation": {"enabled": True, "observer_overhead_included": True}}


class FakeSeries(MTPSeriesRunner):
    """Substitute only external process/power boundaries, retain real validation/merge."""
    def __init__(self, config, output, resume=False):
        super().__init__(config, output, resume)
        self.calls = []
        self.fingerprint = deepcopy(FINGERPRINT)
        self.fail_id = None
        self.wrong_index = False
        self.mismatch = False
        self.corrupt_trace = False
        self.power = {"percent": 57, "discharging": False, "stop": False}
        self.post_power = None
        self.power_calls = self.clock_calls = 0
        self.clock_offset = 0
        self.simulated_sleep = 0.0

    def _fingerprint(self):
        return deepcopy(self.fingerprint)

    def _battery(self):
        self.power_calls += 1
        values = self.post_power if self.post_power is not None and self.power_calls % 2 == 0 else self.power
        source = "AC Power" if values.get("on_ac", not values["discharging"]) else "Battery Power"
        state = "discharging" if values["discharging"] else "charging"
        raw = f"Now drawing from '{source}'\n -InternalBattery-0 {values['percent']}%; {state}; 1:00 remaining"
        return {"checked_at": "2026-09-19T00:00:00Z", **self._parse_power(raw)}

    def _clock_snapshot(self):
        self.clock_calls += 1
        if self.clock_calls % 2 == 0:
            self.clock_offset += int(self.simulated_sleep * 1_000_000_000)
        awake = 1_000_000_000_000 + self.clock_calls * 2_000_000_000
        return {"schema_version": 1, "absolute_ticks": awake, "continuous_ticks": awake + self.clock_offset,
                "timebase_numer": 1, "timebase_denom": 1, "sampling_span_ticks": 0}

    def _execute(self, block, attempt):
        self.calls.append(block["id"])
        directory = self.output / "fixture-workers" / f"{block['id']}-{attempt['attempt']}"
        directory.mkdir(parents=True)
        attempt["run_directory"] = str(directory)
        Path(attempt["log"]).write_text("fixture log\n")
        self._save()
        config = asdict(self._worker_config(block))
        prompts, rows = [], []
        for index in range(block["start_index"], block["start_index"] + block["count"]):
            index = 0 if self.wrong_index else index
            prompt = {"index": index, "prompt_tokens": [index + 100, 42], "original_prompt_tokens": 2}
            prompts.append(prompt)
            row = {"index": index, "prompt_tokens": 2, "generated_tokens": config["max_new_tokens"],
                   "decode_tokens": config["max_new_tokens"] - 1, "prefill_seconds": 1.0,
                   "decode_seconds": 2.0, "generated_token_ids": list(range(config["max_new_tokens"])),
                   "prompt_token_sha256": SpeculativeReport._hash(prompt["prompt_tokens"]),
                   "timing_method": "fixture synchronized", "custom_preserved_field": {"index": index}}
            speculative = block["role"] == "mtp"
            if speculative and self.mismatch:
                row["generated_token_ids"][-1] += 1000
            if speculative:
                row.update(drafted_tokens=config["max_new_tokens"] - 1,
                           accepted_draft_tokens=config["max_new_tokens"] - 1, speculative_rounds=1,
                           emitted_draft_tokens=config["max_new_tokens"] - 1, emitted_target_tokens=1)
            if config["trace_generation"]:
                row["generation_trace"] = raw_trace(row, speculative)
                if self.corrupt_trace:
                    row["generation_trace"]["events"][-1]["output_count"] = 999
            rows.append(row)
        summary = {"status": "completed", "completed_samples": block["count"], "config": config,
                   "started_at": f"2026-09-19T00:{len(self.calls):02d}:00Z", "elapsed_seconds": 10.0,
                   "environment": {"python": "3.13.5", "packages": PACKAGES, "caffeinate_assertions": "di"},
                   "backend": {"sampling": "greedy", "wired_memory": True, "versions": PACKAGES,
                               "draft_manifest_sha256": FINGERPRINT["inputs"]["draft_manifest_sha256"]},
                   "dataset_sha256": FINGERPRINT["inputs"]["dataset_sha256"],
                   "model_manifest_sha256": FINGERPRINT["inputs"]["model_manifest_sha256"],
                   "prompt_tokens_sha256": SpeculativeReport._hash([p["prompt_tokens"] for p in prompts]),
                   "metrics": aggregate(rows)}
        if self.fail_id == block["id"]:
            rows = rows[:1]
            summary.update(status="failed", completed_samples=len(rows), metrics=aggregate(rows))
        (directory / "summary.json").write_text(json.dumps(summary))
        (directory / "prompts.json").write_text(json.dumps(prompts))
        (directory / "samples.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        return 1 if self.fail_id == block["id"] else 0


@pytest.fixture
def setup(tmp_path):
    config = MTPSeriesConfig(count=6, chunk_size=2, max_new_tokens=4, label="fixture")
    return config, tmp_path / "series"


def test_abba_order_and_global_slice_commands(setup):
    config, output = setup
    runner = FakeSeries(config, output)
    assert [b["id"] for b in runner.plan()] == ["000-baseline", "000-mtp", "001-mtp", "001-baseline", "002-baseline", "002-mtp"]
    assert [b["start_index"] for b in runner.plan()] == [0, 0, 2, 2, 4, 4]
    command = runner.command(runner.plan()[2])
    assert command[command.index("--start-index") + 1] == "2"
    assert command[command.index("--warmup") + 1] == "1"
    assert "--trace-generation" in command
    assert MTPSeriesConfig().count == 100 and MTPSeriesConfig().max_new_tokens == 2048


def test_complete_series_merges_exact_coverage_and_preserves_trajectories(setup):
    config, output = setup
    runner = FakeSeries(config, output)
    manifest = runner.run()
    assert manifest["status"] == "completed" and manifest["parity_status"] == "passed"
    for role, directory in manifest["merged_runs"].items():
        directory = Path(directory)
        summary = json.loads((directory / "summary.json").read_text())
        rows = [json.loads(line) for line in (directory / "samples.jsonl").read_text().splitlines()]
        expected = [row for block in sorted([b for b in manifest["blocks"] if b["role"] == role], key=lambda b:b["start_index"])
                    for row in map(json.loads, (Path(block["attempts"][-1]["run_directory"]) / "samples.jsonl").read_text().splitlines())]
        assert rows == expected
        assert [row["index"] for row in rows] == list(range(6))
        assert all(row["generation_trace"]["events"] for row in rows)
        assert summary["metrics"] == aggregate(rows)
        assert summary["config"]["start_index"] == 0 and summary["config"]["count"] == 6
        assert summary["series"]["process_count"] == summary["series"]["model_load_count"] == 3
        assert summary["series"]["total_warmup_requests"] == 3
    before = {p: p.read_bytes() for p in output.glob("fixture-workers/**/*") if p.is_file()}
    resumed = FakeSeries(config, output, resume=True)
    assert resumed.run()["status"] == "completed"
    assert resumed.calls == []
    assert all(path.read_bytes() == content for path, content in before.items())


def test_failed_partial_attempt_is_preserved_and_only_incomplete_blocks_rerun(setup):
    config, output = setup
    first = FakeSeries(config, output)
    first.fail_id = "000-mtp"
    with pytest.raises(RuntimeError, match="Child failed"):
        first.run()
    failed = json.loads((output / "manifest.json").read_text())
    assert failed["status"] == "failed" and "merged_runs" not in failed
    path = Path(failed["blocks"][1]["attempts"][0]["run_directory"]) / "samples.jsonl"
    partial = path.read_bytes()
    resumed = FakeSeries(config, output, resume=True)
    result = resumed.run()
    assert "000-baseline" not in resumed.calls and resumed.calls[0] == "000-mtp"
    assert len(result["blocks"][1]["attempts"]) == 2
    assert path.read_bytes() == partial
    assert result["status"] == "completed"


@pytest.mark.parametrize("file", ["summary.json", "prompts.json", "samples.jsonl"])
def test_resume_rejects_changed_completed_raw_hash_before_work(setup, file):
    config, output = setup
    result = FakeSeries(config, output).run()
    path = Path(result["blocks"][0]["attempts"][0]["run_directory"]) / file
    path.write_bytes(path.read_bytes() + b"\n")
    resumed = FakeSeries(config, output, resume=True)
    with pytest.raises(ValueError, match="hash mismatch"):
        resumed.run()
    assert resumed.calls == []
    assert json.loads((output / "manifest.json").read_text())["status"] == "failed"


def test_resume_rejects_source_or_protocol_drift(setup):
    config, output = setup
    FakeSeries(config, output).run()
    changed = FakeSeries(config, output, resume=True)
    changed.fingerprint["project_files"]["runner.py"] = "f" * 64
    with pytest.raises(ValueError, match="source hashes/runtime/inputs"):
        changed.run()
    assert changed.calls == []
    protocol = FakeSeries(replace(config, max_new_tokens=8), output, resume=True)
    with pytest.raises(ValueError, match="suite protocol"):
        protocol.run()
    assert protocol.calls == []


def test_wrong_global_coverage_fails_closed(setup):
    config, output = setup
    runner = FakeSeries(config, output)
    runner.wrong_index = True
    with pytest.raises(ValueError, match="Duplicate prompt indices|global indices"):
        runner.run()
    assert "merged_runs" not in json.loads((output / "manifest.json").read_text())


def test_low_battery_stops_before_launch_and_resumes(setup):
    config, output = setup
    runner = FakeSeries(config, output)
    runner.power = {"percent": 19, "discharging": True, "stop": True}
    assert runner.run()["status"] == "paused_battery"
    assert not runner.calls
    resumed = FakeSeries(config, output, resume=True)
    assert resumed.run()["status"] == "completed"


def test_missing_hash_evidence_cannot_silently_recover_completed_block(setup):
    config, output = setup
    result = FakeSeries(config, output).run()
    del result["blocks"][0]["attempts"][0]["files"]
    (output / "manifest.json").write_text(json.dumps(result))
    resumed = FakeSeries(config, output, resume=True)
    with pytest.raises(ValueError, match="hash evidence"):
        resumed.run()
    assert resumed.calls == []


def test_incomplete_series_cannot_be_merged(setup):
    config, output = setup
    runner = FakeSeries(config, output)
    runner.manifest = {"blocks": [{"status": "pending"}]}
    with pytest.raises(ValueError, match="partial series"):
        runner._merge()


def test_output_mismatch_is_recorded_without_aborting_valid_series(setup):
    config, output = setup
    runner = FakeSeries(config, output)
    runner.mismatch = True
    result = runner.run()
    assert result["status"] == "completed" and result["parity_status"] == "failed"
    assert len(runner.calls) == 6
    report = json.loads(Path(result["comparison_path"]).read_text())
    assert report["comparisons"][0]["parity"]["matched_prompts"] == 0
    assert report["comparisons"][0]["parity"]["total_prompts"] == 6


def test_trace_corruption_fails_before_any_merge(setup):
    config, output = setup
    runner = FakeSeries(config, output)
    runner.corrupt_trace = True
    with pytest.raises(ValueError, match="output_count"):
        runner.run()
    assert len(runner.calls) == 1
    assert "merged_runs" not in json.loads((output / "manifest.json").read_text())


def test_recovers_complete_child_after_parent_died_before_checkpoint(setup):
    config, output = setup
    result = FakeSeries(config, output).run()
    block = result["blocks"][0]
    block["status"] = block["attempts"][0]["status"] = "running"
    block["attempts"][0].pop("files")
    result["status"] = "running"
    (output / "manifest.json").write_text(json.dumps(result))
    resumed = FakeSeries(config, output, resume=True)
    saved = resumed.run()
    assert resumed.calls == []
    assert saved["blocks"][0]["attempts"][0]["recovered"] is True


def test_recovers_partial_interrupted_child_by_new_whole_attempt(setup):
    config, output = setup
    failed = FakeSeries(config, output)
    failed.fail_id = "000-mtp"
    with pytest.raises(RuntimeError):
        failed.run()
    state = json.loads((output / "manifest.json").read_text())
    state["blocks"][1]["status"] = state["blocks"][1]["attempts"][0]["status"] = "running"
    (output / "manifest.json").write_text(json.dumps(state))
    resumed = FakeSeries(config, output, resume=True)
    result = resumed.run()
    assert resumed.calls[0] == "000-mtp" and "000-baseline" not in resumed.calls
    assert result["blocks"][1]["attempts"][0]["status"] == "interrupted"
    assert result["blocks"][1]["attempts"][1]["status"] == "completed"


@pytest.mark.parametrize("percent,state,stop", [(19, "discharging", True), (20, "discharging", False), (19, "charging", False)])
def test_battery_threshold_uses_real_pmset_state(setup, monkeypatch, percent, state, stop):
    from inference_lab.benchmarking import mtp_series
    config, output = setup
    monkeypatch.setattr(mtp_series.subprocess, "check_output", lambda *a, **k: f"Now drawing from Battery Power\n -InternalBattery-0 {percent}%; {state}; 1:00 remaining")
    result = MTPSeriesRunner(config, output)._battery()
    assert result["percent"] == percent and result["stop"] is stop


def test_sleep_invalidates_completed_raw_attempt_and_stops_without_auto_retry(setup):
    config, output = setup
    runner = FakeSeries(config, output)
    runner.simulated_sleep = 3.5
    result = runner.run()
    assert result["status"] == "paused_sleep"
    assert runner.calls == ["000-baseline"]
    block = result["blocks"][0]
    assert block["status"] == "invalid" and len(block["attempts"]) == 1
    attempt = block["attempts"][0]
    observation = attempt["host_observation"]
    assert observation["sleep_assessment"]["sleep_seconds"] == 3.5
    assert observation["sleep_assessment"]["sleep_detected"] is True
    assert observation["valid"] is False and attempt["status"] == "invalid"
    path = Path(attempt["run_directory"]) / "samples.jsonl"
    preserved = path.read_bytes()
    assert json.loads((path.parent / "summary.json").read_text())["status"] == "completed"
    assert "merged_runs" not in result
    with pytest.raises(ValueError, match="partial series"):
        runner._merge()
    resumed = FakeSeries(config, output, resume=True)
    completed = resumed.run()
    assert resumed.calls[0] == "000-baseline"
    assert completed["blocks"][0]["attempts"][0]["status"] == "invalid"
    assert completed["blocks"][0]["attempts"][1]["status"] == "completed"
    assert path.read_bytes() == preserved


def test_requires_ac_before_worker_by_default_but_explicit_battery_is_supported(setup):
    config, output = setup
    assert config.require_ac is True
    runner = FakeSeries(config, output)
    runner.power = {"percent": 75, "discharging": True, "on_ac": False}
    result = runner.run()
    assert result["status"] == "paused_power" and not runner.calls
    assert not result["blocks"][0]["attempts"]
    allowed = FakeSeries(replace(config, require_ac=False), output.parent / "battery-allowed")
    allowed.power = runner.power
    completed = allowed.run()
    assert completed["status"] == "completed"
    assert completed["host_guard_policy"]["require_ac"] is False


def test_loss_of_ac_after_block_invalidates_raw_and_resume_waits_for_ac(setup):
    config, output = setup
    runner = FakeSeries(config, output)
    runner.post_power = {"percent": 57, "discharging": True, "on_ac": False}
    result = runner.run()
    assert result["status"] == "paused_power" and len(runner.calls) == 1
    attempt = result["blocks"][0]["attempts"][0]
    assert attempt["status"] == "invalid" and attempt["host_observation"]["valid"] is False
    assert not attempt["host_observation"]["sleep_assessment"]["sleep_detected"]
    resumed = FakeSeries(config, output, resume=True)
    resumed.power = runner.post_power
    assert resumed.run()["status"] == "paused_power" and resumed.calls == []
    assert len(resumed.manifest["blocks"][0]["attempts"]) == 1
    stable = FakeSeries(config, output, resume=True)
    assert stable.run()["status"] == "completed"
    assert stable.calls[0] == "000-baseline"


def test_ac_connected_but_discharging_is_not_accepted(setup):
    config, output = setup
    runner = FakeSeries(config, output)
    runner.power = {"percent": 57, "discharging": True, "on_ac": True}
    assert runner.run()["status"] == "paused_power"
    assert not runner.calls


def test_completed_block_without_clock_evidence_cannot_resume(setup):
    config, output = setup
    result = FakeSeries(config, output).run()
    del result["blocks"][0]["attempts"][0]["host_observation"]
    (output / "manifest.json").write_text(json.dumps(result))
    resumed = FakeSeries(config, output, resume=True)
    with pytest.raises(ValueError, match="Missing complete host clock/power evidence"):
        resumed.run()
    assert not resumed.calls


def test_legacy_completed_series_cannot_enter_guarded_protocol(setup):
    config, output = setup
    result = FakeSeries(config, output).run()
    result["schema_version"] = 1
    result.pop("host_guard_policy")
    (output / "manifest.json").write_text(json.dumps(result))
    resumed = FakeSeries(config, output, resume=True)
    with pytest.raises(ValueError, match="current sleep/power guard protocol"):
        resumed.run()
    assert not resumed.calls


def test_saved_assessment_is_recomputed_on_resume(setup):
    config, output = setup
    result = FakeSeries(config, output).run()
    result["blocks"][0]["attempts"][0]["host_observation"]["sleep_assessment"]["sleep_seconds"] = 123
    (output / "manifest.json").write_text(json.dumps(result))
    resumed = FakeSeries(config, output, resume=True)
    with pytest.raises(ValueError, match="sleep assessment"):
        resumed.run()
    assert not resumed.calls


def test_merged_sources_preserve_every_clock_and_power_endpoint(setup):
    config, output = setup
    result = FakeSeries(config, output).run()
    for role, directory in result["merged_runs"].items():
        summary = json.loads((Path(directory) / "summary.json").read_text())
        series = summary["series"]
        assert series["host_guard_policy"] == result["host_guard_policy"]
        assert series["host_guard_policy"]["power_observation_scope"] == "before/after child process only; intervening power changes are not observed"
        for source in series["source_block_runs"]:
            block = next(block for block in result["blocks"] if block["id"] == source["id"])
            assert source["host_observation"] == block["attempts"][-1]["host_observation"]
            assert source["host_observation"]["valid"] is True


def test_missing_after_power_stops_with_unusable_evidence(setup):
    class MissingAfterPower(FakeSeries):
        def _battery(self):
            if self.power_calls == 1:
                raise OSError("pmset unavailable after worker")
            return super()._battery()
    config, output = setup
    runner = MissingAfterPower(config, output)
    result = runner.run()
    assert result["status"] == "paused_host_evidence"
    assert len(runner.calls) == 1
    assert result["blocks"][0]["attempts"][0]["status"] == "invalid"
    assert result["blocks"][0]["attempts"][0]["host_observation"]["collection_errors"]


def test_missing_clock_endpoint_prevents_recovery_of_complete_child(setup):
    config, output = setup
    result = FakeSeries(config, output).run()
    block = result["blocks"][0]
    block["status"] = block["attempts"][0]["status"] = "running"
    del block["attempts"][0]["host_observation"]["clock_after"]
    result.pop("merged_runs")
    result.pop("merged_files")
    (output / "manifest.json").write_text(json.dumps(result))
    resumed = FakeSeries(config, output, resume=True)
    repaired = resumed.run()
    assert resumed.calls == ["000-baseline"]
    assert repaired["blocks"][0]["attempts"][0]["status"] == "interrupted"
    assert repaired["blocks"][0]["attempts"][1]["status"] == "completed"
