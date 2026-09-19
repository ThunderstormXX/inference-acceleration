"""Generation logging must preserve GPU scheduling and retain auditable events."""
from copy import deepcopy

import pytest

from inference_lab.visualization.recording import GenerationRecorder, validate_generation_trace
from inference_lab.visualization.trace import MTPTraceObserver
import test_mlx_backend as mlx_fakes
from test_trace import fake_observer_round, assert_restored


def measured_baseline(enabled=True):
    fixture = mlx_fakes.MLXMeasurementTests()
    fixture.setUp()
    backend = fixture.backend
    backend.trace_generation = enabled
    backend._tokenizer.eos_token_id = 9
    backend._model_config = {"text_config": {"eos_token_id": 10}}
    result = backend.measure([1, 2, 3, 4, 5], max_new_tokens=4)
    return backend, result


def test_baseline_logging_preserves_gpu_call_sequence_and_exact_outputs():
    plain, old = measured_baseline(False)
    traced, row = measured_baseline(True)
    assert "generation_trace" not in old
    assert old["generated_token_ids"] == row["generated_token_ids"] == [6, 7, 8, 9]
    assert old["output_text"] == row["output_text"] == "6,7,8,9"
    assert plain._mx.events == traced._mx.events
    validate_generation_trace(row)
    lane = row["generation_trace"]
    assert lane["eos_token_ids"] == [9, 10]
    assert lane["prefill_seconds"] == row["prefill_seconds"]
    assert lane["decode_seconds"] == row["decode_seconds"]
    assert [event["output_count"] for event in lane["events"]] == [1, 2, 3, 4]
    assert all(event["type"] == "commit" for event in lane["events"])
    assert "token_texts" not in lane and "decoded_prefixes" not in lane
    assert lane["instrumentation"]["observer_overhead_included"] is True


def test_recording_resets_between_requests_and_one_token_has_no_decode():
    backend, first = measured_baseline(True)
    row = backend.measure([7], max_new_tokens=1)
    validate_generation_trace(row)
    assert row["generated_token_ids"] == [8]
    assert row["generation_trace"]["events"][0]["output_count"] == 1
    assert row["decode_seconds"] == 0
    assert first["generation_trace"]["token_ids"] == [6, 7, 8, 9]


@pytest.mark.parametrize("corrupt", [
    lambda row: row["generation_trace"].update(token_ids=[999]),
    lambda row: row["generation_trace"]["events"][1].update(token_ids=[999]),
    lambda row: row["generation_trace"]["events"][1].update(output_count=99),
    lambda row: row["generation_trace"]["events"][1].update(t=-1),
    lambda row: row["generation_trace"]["events"][1].update(t=float("nan")),
    lambda row: row["generation_trace"]["events"][1].update(t=float("inf")),
    lambda row: row["generation_trace"]["events"][1].update(t=row["generation_trace"]["total_seconds"]+1),
    lambda row: row["generation_trace"].update(prefill_seconds=99),
    lambda row: row["generation_trace"].update(eos_token_ids=[True]),
    lambda row: row["generation_trace"].update(instrumentation={}),
])
def test_validation_rejects_inconsistent_raw_evidence(corrupt):
    _, row = measured_baseline(True)
    corrupt(row)
    with pytest.raises(ValueError):
        validate_generation_trace(row)


def observed_mtp(*, fail=None):
    backend, module, _, _, calls, originals = fake_observer_round(1, [11, 13], fail=fail)
    backend._model_config = {}
    backend._tokenizer = backend.tokenizer
    ticks = iter(range(100, 1000))
    recorder = GenerationRecorder(backend, clock=lambda: float(next(ticks)), speculative=True)
    observer = MTPTraceObserver(backend, recorder, module=module, synchronize_first=False,
                                suppress_detokenization=False, record_first=False)
    return backend, module, calls, originals, recorder, observer


def test_mtp_benchmark_observer_leaves_first_token_sync_and_text_decode_to_backend():
    backend, module, calls, originals, recorder, observer = observed_mtp()
    with observer.observe():
        assert backend.tokenizer.decode([10]) == "original decode"
        recorder.start_at(99)
        stream = backend._run_rounds()
        first, _ = next(stream)
        assert recorder.events == []
        # Simulate the backend's existing first-token phase boundary.
        recorder.commit([first], round=0, accepted_count=0, draft_count=0)
        prefill = recorder.events[0]["t"] + 0.5
        assert list(stream) == [(11, None), (13, None)]
    row = {"generated_token_ids": [10, 11, 13], "generated_tokens": 3,
           "prefill_seconds": prefill, "decode_seconds": recorder.now() - prefill,
           "drafted_tokens": 2, "accepted_draft_tokens": 1, "speculative_rounds": 1}
    row["generation_trace"] = recorder.finish_measurement(row)
    validate_generation_trace(row)
    assert "sync" not in calls
    assert not any(isinstance(call, tuple) and call[0] == "eval" for call in calls)
    assert recorder.events[1]["token_ids"] == [11, 12]
    assert recorder.events[2]["rejected_token_ids"] == [12]
    assert_restored(backend, module, originals)
    broken = deepcopy(row)
    broken["accepted_draft_tokens"] = 2
    with pytest.raises(ValueError, match="counter differs"):
        validate_generation_trace(broken)


@pytest.mark.parametrize("failure", ["verify", "commit"])
def test_benchmark_observer_restores_hooks_on_failed_round(failure):
    backend, module, calls, originals, recorder, observer = observed_mtp(fail=failure)
    with pytest.raises(RuntimeError, match=failure):
        with observer.observe():
            recorder.start_at(99)
            stream = backend._run_rounds()
            token, _ = next(stream)
            recorder.commit([token])
            list(stream)
    assert_restored(backend, module, originals)
    assert "sync" not in calls


@pytest.mark.parametrize("failure", [None, "bad-second-trace", "missing-second-trace"])
def test_runner_validates_and_durably_preserves_trace_rows(monkeypatch, tmp_path, failure):
    import json
    import test_benchmark_protocol as protocol
    from inference_lab.core.config import BenchmarkConfig
    from inference_lab.benchmarking import runner as runner_module

    dataset = tmp_path / "rows.jsonl"
    dataset.write_text("".join(json.dumps({"problem": f"problem {i}", "response": "teacher"}) + "\n" for i in range(2)))
    config = BenchmarkConfig("mlx", dataset_path=str(dataset), count=2, max_new_tokens=4,
                             warmup=0, trace_generation=True)
    backend = protocol.FakeBackend()
    backend.trace_generation = False
    original_measure = backend.measure

    def measure(tokens, max_new_tokens):
        row = original_measure(tokens, max_new_tokens)
        assert backend.trace_generation is True
        prefill = row["prefill_seconds"]
        lane = {"schema_version": 1, "token_ids": list(row["generated_token_ids"]),
                "prefill_seconds": prefill, "decode_seconds": row["decode_seconds"],
                "total_seconds": prefill + row["decode_seconds"], "eos_token_ids": [],
                "instrumentation": {"enabled": True},
                "events": [{"type": "commit", "t": prefill / 2 if i == 0 else prefill + i * 0.1,
                            "token_ids": [3], "output_count": i + 1} for i in range(max_new_tokens)]}
        row["generation_trace"] = lane
        if len(backend.calls) == 2:
            if failure == "bad-second-trace":
                lane["events"][-1]["token_ids"] = [999]
            elif failure == "missing-second-trace":
                del row["generation_trace"]
        return row

    backend.measure = measure
    protocol.prepare_runner(monkeypatch, tmp_path, backend)
    synced = []
    monkeypatch.setattr(runner_module.os, "fsync", lambda descriptor: synced.append(descriptor))
    if failure:
        with pytest.raises(RuntimeError, match="Benchmark failed"):
            runner_module.BenchmarkRunner(config).run()
    else:
        runner_module.BenchmarkRunner(config).run()
    summary, directory = protocol.read_summary(tmp_path)
    rows = [json.loads(line) for line in (directory / "samples.jsonl").read_text().splitlines()]
    assert summary["status"] == ("failed" if failure else "completed")
    assert len(rows) == summary["completed_samples"] == len(synced) == (1 if failure else 2)
    for row in rows:
        validate_generation_trace(row)
