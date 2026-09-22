import json

import pytest

from inference_lab.profiling import ProfileRecorder, tensor_metadata


class FakeClock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value

    def advance(self, milliseconds):
        self.value += milliseconds / 1000


class LazyTensor:
    shape = (2, 3)
    dtype = "float16"
    nbytes = 12

    def __init__(self, cost_ms=0, parents=()):
        self.cost_ms = cost_ms
        self.parents = parents
        self.ready = False

    def evaluate(self, clock):
        if not self.ready:
            for parent in self.parents:
                parent.evaluate(clock)
            clock.advance(self.cost_ms)
            self.ready = True


def runtime(sync_ms=0):
    clock = FakeClock()
    evaluated = []

    def evaluate(*values):
        evaluated.append(values)
        for value in values:
            value.evaluate(clock)

    recorder = ProfileRecorder(synchronize=lambda: clock.advance(sync_ms),
                               evaluate=evaluate, clock=clock)
    return recorder, clock, evaluated


def test_lazy_work_is_materialized_before_call_timing_ends():
    recorder, clock, evaluated = runtime(sync_ms=1)

    def operation():
        clock.advance(2)
        return {"logits": LazyTensor(7)}

    result = recorder.measured_call("forward", operation)
    event = recorder.spans[0]
    assert result["logits"].ready
    assert event.inclusive_ms == pytest.approx(11)
    assert event.materialize_ms == pytest.approx(7)
    assert event.entry_sync_ms == pytest.approx(1)
    assert event.exit_sync_ms == pytest.approx(1)
    assert len(evaluated) == 1


def test_explicit_input_materialization_keeps_upstream_work_out_of_callee():
    recorder, clock, _ = runtime()
    inputs = LazyTensor(13)
    recorder.materialize("inputs", inputs)
    recorder.measured_call("target", lambda: LazyTensor(5, (inputs,)))
    assert recorder.spans[0].inclusive_ms == pytest.approx(13)
    assert recorder.spans[1].inclusive_ms == pytest.approx(5)
    assert recorder.to_dict()["root_total_ms"] == pytest.approx(18)


def test_nested_spans_have_additive_exclusive_not_inclusive_durations():
    recorder, clock, _ = runtime(sync_ms=1)
    with recorder.span("round", {"round": 7}):
        clock.advance(3)
        recorder.measured_call("draft", lambda: LazyTensor(5))
        with recorder.span("verify"):
            clock.advance(11)
    report = recorder.to_dict()
    root, draft, verify = report["spans"]
    assert root["inclusive_ms"] == pytest.approx(25)
    assert root["exclusive_ms"] == pytest.approx(5)
    assert draft["inclusive_ms"] == pytest.approx(7)
    assert verify["inclusive_ms"] == pytest.approx(13)
    assert draft["parent_id"] == verify["parent_id"] == root["id"]
    assert report["root_total_ms"] == pytest.approx(report["exclusive_total_ms"])
    assert report["entry_sync_total_ms"] + report["exit_sync_total_ms"] == pytest.approx(6)
    assert draft["start_ms"] >= root["start_ms"]
    assert verify["end_ms"] <= root["end_ms"]
    json.dumps(report)


def test_side_effect_state_is_evaluated_and_duplicate_tensors_are_deduplicated():
    recorder, _, evaluated = runtime()
    output, state = LazyTensor(3), LazyTensor(17)
    result = recorder.measured_call("forward", lambda: (output, output),
                                    materialize=lambda: {"cache": [state, output]})
    assert result == (output, output)
    assert recorder.spans[0].inclusive_ms == pytest.approx(20)
    assert evaluated == [(output, state)]


def test_body_exception_finishes_all_spans_and_preserves_original_error():
    clock = FakeClock()
    calls = 0

    def synchronize():
        nonlocal calls
        calls += 1
        clock.advance(1)
        if calls == 3:
            raise RuntimeError("cleanup failed")

    recorder = ProfileRecorder(synchronize=synchronize, evaluate=lambda *x: None, clock=clock)
    with pytest.raises(ValueError, match="model failed"):
        with recorder.span("round"):
            with recorder.span("forward"):
                raise ValueError("model failed")
    report = recorder.to_dict()
    assert all(item["status"] == "error" for item in report["spans"])
    assert report["spans"][1]["cleanup_error"]["message"] == "cleanup failed"
    assert report["spans"][1]["error"]["type"] == "ValueError"
    assert report["root_total_ms"] == pytest.approx(report["exclusive_total_ms"])


def test_cleanup_failure_without_body_error_is_propagated_and_recorded():
    recorder, _, _ = runtime()
    calls = 0

    def synchronize():
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("device failed")

    recorder._synchronize = synchronize
    with pytest.raises(RuntimeError, match="device failed"):
        recorder.measured_call("forward", lambda: None)
    assert recorder.to_dict()["spans"][0]["status"] == "error"


def test_entry_failure_closes_span_without_retrying_failed_sync():
    calls = 0

    def synchronize():
        nonlocal calls
        calls += 1
        raise RuntimeError("entry failed")

    recorder = ProfileRecorder(synchronize=synchronize, evaluate=lambda *x: None)
    with pytest.raises(RuntimeError, match="entry failed"):
        with recorder.span("round"):
            pytest.fail("body must not execute")
    assert calls == 1
    assert recorder.to_dict()["spans"][0]["error"]["message"] == "entry failed"


def test_plain_span_does_not_claim_to_evaluate_unreturned_lazy_work():
    recorder, _, evaluated = runtime()
    with recorder.span("graph_only"):
        output = LazyTensor(100)
    assert not output.ready
    assert not evaluated
    assert recorder.spans[0].inclusive_ms == 0


def test_tensor_metadata_only_reads_descriptors_and_handles_cycles():
    tensor = LazyTensor(100)
    nested = {"tensor": tensor, "items": (tensor, 42)}
    nested["cycle"] = nested
    description = tensor_metadata(nested)
    assert description["tensor"] == {"shape": [2, 3], "dtype": "float16", "nbytes": 12}
    assert description["items"][1] == {"type": "int"}
    assert description["cycle"]["recursive_reference"]
    assert not tensor.ready
    json.dumps(description)


def test_tensor_metadata_can_derive_bytes_from_itemsize():
    class Array:
        shape = (4, 8)
        dtype = "float32"
        itemsize = 4

    assert tensor_metadata(Array())["nbytes"] == 128


def test_materialization_failure_is_timed_and_recorded():
    clock = FakeClock()

    def evaluate(*values):
        clock.advance(9)
        raise RuntimeError("evaluation failed")

    recorder = ProfileRecorder(synchronize=lambda: None, evaluate=evaluate, clock=clock)
    with pytest.raises(RuntimeError, match="evaluation failed"):
        recorder.measured_call("target", LazyTensor)
    assert recorder.spans[0].materialize_ms == pytest.approx(9)
    assert recorder.spans[0].inclusive_ms == pytest.approx(9)
    assert recorder.to_dict()["spans"][0]["status"] == "error"


def test_profile_limit_and_active_export_are_explicit():
    recorder = ProfileRecorder(synchronize=lambda: None, evaluate=lambda *x: None, max_spans=1)
    with recorder.span("round"):
        with pytest.raises(RuntimeError, match="active"):
            recorder.to_dict()
        with pytest.raises(RuntimeError, match="limit"):
            with recorder.span("extra"):
                pass
    assert len(recorder.to_dict()["spans"]) == 1
    with pytest.raises(ValueError):
        ProfileRecorder(synchronize=lambda: None, evaluate=lambda *x: None, max_spans=0)


def test_public_evaluate_attributes_declared_state_to_current_span():
    recorder, _, evaluated = runtime()
    state = LazyTensor(8)
    with recorder.span("cache_update"):
        recorder.evaluate(state, {"duplicate": state})
    assert evaluated == [(state,)]
    assert recorder.spans[0].materialize_ms == pytest.approx(8)
    with pytest.raises(RuntimeError, match="active span"):
        recorder.evaluate(state)
