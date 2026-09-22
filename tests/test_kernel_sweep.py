"""CPU-only checks for kernel experiment timing, pairing and scope cleanup."""
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from inference_lab.optimizations import kernel_sweep
from inference_lab.optimizations import affine, greedy_head, zmlx


def measurement(seconds=1, ids=None):
    ids = [1] * 16 if ids is None else ids
    return {"generated_token_ids": ids, "generated_tokens": len(ids), "decode_tokens": len(ids) - 1,
            "prompt_tokens": 2, "prefill_seconds": 0.25, "decode_seconds": seconds,
            "output_text": "example", "peak_memory_gb": 5}


def run_row(variant, seconds, *, phase="measurement", prompt=0, repeat=0, parity=True):
    m = measurement(seconds)
    return {"variant": variant, "measurement": m, "phase": phase,
            "prompt_index": prompt, "repeat": repeat, "matches_baseline": parity,
            "tok_s": m["decode_tokens"] / seconds, "sleep": {"sleep_detected": False}}


def test_summary_excludes_references_and_brackets_and_matches_prompt_repeat():
    rows = [run_row("ar", 99, phase="reference", repeat=-1),
            run_row("ar", 2), run_row("ar-head", 1),
            run_row("ar", 3, prompt=1), run_row("ar-head", 4, prompt=1),
            run_row("ar", 10, repeat=1), run_row("ar-head", 2, repeat=1),
            run_row("ar", 99, phase="bracket", repeat=2)]
    summary = kernel_sweep.summary({"runs": rows, "errors": []})
    result = summary["by_variant"]["ar-head"]
    assert result["tok_s"]["n"] == 3
    assert result["pooled_tok_s"] == pytest.approx(45 / 7)
    assert result["speedup_vs_matched_ar"]["mean"] == pytest.approx((2 + 0.75 + 5) / 3)
    assert summary["by_variant"]["ar"]["tok_s"]["n"] == 3


def test_summary_keeps_mismatching_reference_visible_and_handles_missing_pair():
    report = {"runs": [run_row("ar", 1, phase="reference", parity=False), run_row("ar-head", 1)], "errors": []}
    summary = kernel_sweep.summary(report)
    assert summary["all_token_ids_match"] is False
    assert summary["by_variant"]["ar-head"]["speedup_vs_matched_ar"] is None


def bare_runner():
    runner = kernel_sweep.KernelSweep.__new__(kernel_sweep.KernelSweep)
    runner.args = SimpleNamespace(tokens=16)
    runner.clock = SimpleNamespace(snapshot=lambda: {"schema_version": 1, "absolute_ticks": 10, "continuous_ticks": 10, "timebase_numer": 1, "timebase_denom": 1, "sampling_span_ticks": 0})
    runner.prompts = [{"prompt_tokens": [2, 3]}]
    runner.baselines = {0: [1] * 16}
    runner.report = {"runs": [], "errors": []}
    runner.failed = set()
    runner.save = lambda: None
    return runner


def test_run_one_records_first_mismatch(monkeypatch):
    runner = bare_runner()
    ids = [1] * 16
    ids[7] = 2
    monkeypatch.setattr(runner, "measure", lambda *args: (measurement(ids=ids), {}))
    runner.run_one("ar-head", 0, 0, "measurement")
    row = runner.report["runs"][0]
    assert row["matches_baseline"] is False
    assert row["first_mismatch"] == 7


def test_variant_failure_preserves_error_and_disables_variant(monkeypatch):
    runner = bare_runner()
    def fail(*args):
        raise RuntimeError("bad kernel")
    monkeypatch.setattr(runner, "measure", fail)
    runner.run_one("ar-head", 0, 0, "measurement")
    assert runner.failed == {"ar-head"}
    assert runner.report["runs"] == []
    assert "bad kernel" in runner.report["errors"][0]["traceback"]
    with pytest.raises(RuntimeError, match="bad kernel"):
        runner.run_one("ar", 0, 0, "reference")


def test_affine_and_argmax_contexts_exit_in_reverse_order_on_error(monkeypatch):
    events = []
    class FakeContext:
        def __init__(self, label):
            self.label = label
        def __enter__(self):
            events.append("enter-" + self.label)
            return self
        def __exit__(self, *exc):
            events.append("exit-" + self.label)
        def metadata(self):
            return {"label": self.label}
    class Backend:
        _model = object()
        _draft = object()
        def measure(self, *args):
            events.append("measure")
            raise RuntimeError("generation failed")
    runner = bare_runner()
    runner.backend = Backend()
    monkeypatch.setattr(affine, "ScopedAffineOptimization", lambda *args, **kwargs: FakeContext("affine"))
    monkeypatch.setattr(greedy_head, "ScopedGreedyHead", lambda *args: FakeContext("argmax"))
    with pytest.raises(RuntimeError, match="generation failed"):
        runner.measure("dflash-all-argmax", [1], 16)
    assert events == ["enter-affine", "enter-argmax", "measure", "exit-argmax", "exit-affine"]


def test_ar_zmlx_dispatches_base_ar_measure_with_scoped_patch(monkeypatch):
    runner = bare_runner()
    runner.backend = SimpleNamespace(_model=object(), measure=lambda *args: pytest.fail("Wrong AR dispatch"))
    events = []
    @contextmanager
    def scoped(model, patterns):
        assert model is runner.backend._model and patterns == ["deltanet"]
        events.append("enter")
        try:
            yield {"patched_count": 24}
        finally:
            events.append("exit")
    def ar_measure(backend, prompt, tokens):
        assert backend is runner.backend
        assert events == ["enter"]
        return measurement()
    monkeypatch.setattr(zmlx, "scoped_zmlx", scoped)
    monkeypatch.setattr(kernel_sweep.MLXBackend, "measure", ar_measure)
    result, metadata = runner.measure("zmlx-deltanet", [1], 16)
    assert result["generated_tokens"] == 16
    assert metadata["zmlx"]["patched_count"] == 24
    assert events == ["enter", "exit"]


@pytest.mark.parametrize("bad", [
    {"generated_token_ids": [1] * 15, "generated_tokens": 15, "decode_tokens": 14},
    {"generated_token_ids": [True] * 16},
    {"decode_tokens": 16},
    {"decode_seconds": float("nan")},
])
def test_invalid_measurement_cannot_be_reported_as_speed(monkeypatch, bad):
    runner = bare_runner()
    m = measurement()
    m.update(bad)
    monkeypatch.setattr(runner, "measure", lambda *args: (m, {}))
    runner.run_one("ar-head", 0, 0, "measurement")
    assert runner.report["runs"] == []
    assert runner.failed == {"ar-head"}
    assert runner.report["errors"]
