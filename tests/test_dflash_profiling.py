"""CPU-only contracts for scoped DFlash stream and module instrumentation."""

from contextlib import contextmanager, nullcontext
import inspect
from types import SimpleNamespace

import pytest

from inference_lab.profiling.dflash import DFlashProfiler, instrument_stream, plain_target


mx = SimpleNamespace(stream=lambda selected: nullcontext())
generation_stream = object()


def observer_hook(events):
    events.append("original observer")


def specimen(events, max_tokens=2, *, fail=False):
    n = 0
    _capture = object()
    try:
        yield "prefill"
        while n < max_tokens:
            events.append(("round", n))
            with mx.stream(generation_stream):
                events.append("draft")
            if _capture is not None:
                observer_hook(events)
                if fail:
                    raise RuntimeError("verification failed")
            with mx.stream(generation_stream):
                events.append("verify compute")
            d_list = [n]
            events.append("commit")
            n += 1
            yield d_list
            trim = 1
            events.append(("rollback", trim))
        yield "terminal"
    finally:
        events.append("cleanup")


class PhaseProbe:
    def __init__(self):
        self.events = []
        self.active = []

    @contextmanager
    def stage(self, name, state):
        self.events.append(("enter", name, state["n"]))
        self.active.append(name)
        try:
            yield
        finally:
            self.events.append(("exit", name))
            assert self.active.pop() == name

    def drain(self, name, state):
        assert self.active[-1] == name
        self.events.append(("drain", name))


def test_ast_preserves_outputs_original_event_order_and_yield_before_rollback():
    baseline_events, transformed_events = [], []
    expected = list(specimen(baseline_events))
    probe = PhaseProbe()
    transformed = instrument_stream(specimen, probe)
    stream = transformed(transformed_events)
    assert next(stream) == "prefill"
    assert not probe.events
    assert next(stream) == [0]
    assert not probe.active  # Consumer latency must not enter any timed phase.
    assert ("rollback", 1) not in transformed_events
    assert next(stream) == [1]
    assert transformed_events.count(("rollback", 1)) == 1
    assert not probe.active
    assert list(stream) == ["terminal"]
    assert expected == ["prefill", [0], [1], "terminal"]
    assert transformed_events == baseline_events
    enters = [row[1] for row in probe.events if row[0] == "enter"]
    assert enters == ["draft", "verify", "acceptance_commit", "rollback"] * 2
    assert len([row for row in probe.events if row[0] == "drain"]) == 8


def test_ast_exception_and_generator_close_preserve_cleanup():
    events, probe = [], PhaseProbe()
    stream = instrument_stream(specimen, probe)(events, fail=True)
    assert next(stream) == "prefill"
    with pytest.raises(RuntimeError, match="verification failed"):
        next(stream)
    assert not probe.active
    assert events[-1] == "cleanup"
    assert ("drain", "verify") not in probe.events

    events = []
    stream = instrument_stream(specimen, probe)(events)
    next(stream)
    next(stream)
    stream.close()
    assert not probe.active
    assert events[-1] == "cleanup"
    assert ("rollback", 1) not in events


def test_ast_copies_current_observer_globals_without_mutating_original(monkeypatch):
    original = observer_hook

    def installed_observer(events):
        events.append("installed observer")

    monkeypatch.setitem(specimen.__globals__, "observer_hook", installed_observer)
    transformed = instrument_stream(specimen, PhaseProbe())
    monkeypatch.setitem(specimen.__globals__, "observer_hook", original)
    events = []
    list(transformed(events, max_tokens=1))
    assert "installed observer" in events
    assert "original observer" not in events
    original_events = []
    list(specimen(original_events, max_tokens=1))
    assert "original observer" in original_events
    assert "__dflash_profiler" not in specimen.__globals__


@pytest.mark.parametrize("before,after", [
    ("while n < max_tokens:", "while n <= max_tokens:"),
    ("with mx.stream(generation_stream):", "with nullcontext():"),
    ("d_list = [n]", "renamed_decision = [n]"),
    ("trim = 1", "renamed_trim = 1"),
    ("yield d_list\n            trim = 1", "yield d_list\n            events.append('between')\n            trim = 1"),
    ("d_list = [n]", "d_list = [n]\n            d_list = [n]"),
])
def test_ast_rejects_changed_or_ambiguous_boundaries(monkeypatch, before, after):
    source = inspect.getsource(specimen)
    assert before in source
    monkeypatch.setattr("inference_lab.profiling.dflash.inspect.getsource",
                        lambda function: source.replace(before, after))
    with pytest.raises(ValueError):
        instrument_stream(specimen, PhaseProbe())


class FakeTensor:
    shape = (1, 3, 8)
    dtype = "float16"
    nbytes = 48
    size = 24


class FakeLeaf:
    def __call__(self, value, **kwargs):
        return value

    def named_modules(self):
        return [("", self)]


class FakeLayer:
    def __call__(self, value, **kwargs):
        return value

    def named_modules(self):
        return [("", self)]


class FakeModel:
    def __init__(self, head, layers):
        self.lm_head = head
        self.layers = layers

    def __call__(self, value, **kwargs):
        return value

    def named_modules(self):
        # Upstream feature wrappers can hide decoder layers here.
        return [("", self), ("lm_head", self.lm_head)]


class FakeDraft(FakeModel):
    def bind(self, model):
        self.bound_model = model


def fake_profiler(detail="operators"):
    events = []
    stream = object()

    @contextmanager
    def stream_context(selected):
        assert selected is stream
        events.append("stream enter")
        try:
            yield
        finally:
            events.append("stream exit")

    def synchronize(selected):
        assert selected is stream
        events.append("sync")

    mx = SimpleNamespace(
        stream=stream_context, synchronize=synchronize,
        eval=lambda *values: events.append(("eval", values)),
        fast=SimpleNamespace(scaled_dot_product_attention=lambda value: value),
    )
    head, target_layer, draft_layer = FakeLeaf(), FakeLayer(), FakeLayer()
    target = FakeModel(head, [SimpleNamespace(_layer=target_layer)])
    draft = FakeDraft(head, [draft_layer])
    upstream = SimpleNamespace(
        generation_stream=stream, _stream_generate=specimen,
        _get_layers=lambda model: model.layers,
        _gd_mod=SimpleNamespace(gated_delta_update=lambda value: value),
    )

    @contextmanager
    def observe(observer):
        events.append(("observe enter", observer))
        try:
            yield
        finally:
            events.append(("observe exit", observer))

    backend = SimpleNamespace(_mx=mx, _upstream=upstream, _model=target,
                              _draft=draft, _observe=observe)
    profiler = DFlashProfiler(backend, detail=detail, detail_rounds=(1,))
    return profiler, backend, events, target_layer, draft_layer


def phase_state():
    return {"n": 1, "bs": 3, "prompt": FakeTensor(), "hidden": FakeTensor()}


def test_registration_unwraps_hidden_layers_and_assigns_shared_head_by_phase():
    profiler, backend, _, target_layer, draft_layer = fake_profiler()
    profiler._register_modules()
    assert profiler._module_paths[id(target_layer)]["target"][0] == "target.layers.0"
    assert profiler._module_paths[id(draft_layer)]["draft"][0] == "draft.layers.0"
    aliases = profiler._module_paths[id(backend._model.lm_head)]
    assert set(aliases) == {"target", "draft"}
    tensor = FakeTensor()
    for phase in ("draft", "verify"):
        with profiler.stage(phase, phase_state()):
            assert profiler._module_call(FakeLeaf.__call__, backend._model.lm_head,
                                         (tensor,), {}) is tensor
    paths = [event.name for event in profiler.recorder.spans if event.metadata.get("kind") == "module"]
    assert paths == ["draft.lm_head", "target.lm_head"]
    assert profiler.phase is None and profiler.event is None


def test_selected_rounds_and_layer_only_mode_limit_detail():
    profiler, backend, _, target_layer, _ = fake_profiler(detail="layers")
    profiler._register_modules()
    tensor = FakeTensor()
    with profiler.stage("draft", phase_state()):
        profiler._module_call(FakeLeaf.__call__, backend._draft.lm_head, (tensor,), {})
        profiler._operation_call("attention", lambda value: value, (tensor,), {})
    with profiler.stage("draft", phase_state()):
        profiler._module_call(FakeLeaf.__call__, backend._draft.lm_head, (tensor,), {})
    modules = [event for event in profiler.recorder.spans if event.metadata.get("kind") == "module"]
    assert len(modules) == 1 and modules[0].metadata["round"] == 1
    assert not [event for event in profiler.recorder.spans if event.metadata.get("kind") == "operation"]


def test_scoped_install_restores_observer_function_operators_and_calls_on_failure():
    profiler, backend, _, _, _ = fake_profiler()
    original_observe = backend._observe
    original_stream = backend._upstream._stream_generate
    original_attention = backend._mx.fast.scaled_dot_product_attention
    original_gdn = backend._upstream._gd_mod.gated_delta_update
    original_leaf_call = FakeLeaf.__call__
    with pytest.raises(RuntimeError, match="request failed"):
        with profiler.installed():
            assert FakeLeaf.__call__ is not original_leaf_call
            with backend._observe("request"):
                assert backend._upstream._stream_generate is not original_stream
                raise RuntimeError("request failed")
    assert backend._observe is original_observe
    assert backend._upstream._stream_generate is original_stream
    assert backend._mx.fast.scaled_dot_product_attention is original_attention
    assert backend._upstream._gd_mod.gated_delta_update is original_gdn
    assert FakeLeaf.__call__ is original_leaf_call


def test_plain_target_restores_feature_wrappers_and_hidden_state_on_error():
    profiler, backend, _, layer, _ = fake_profiler()
    original_layers = list(backend._model.layers)
    hidden = [FakeTensor()]
    backend._model._hidden_states = hidden
    with pytest.raises(RuntimeError):
        with plain_target(backend):
            assert backend._model.layers == [layer]
            assert not hasattr(backend._model, "_hidden_states")
            raise RuntimeError("baseline failed")
    assert backend._model.layers == original_layers
    assert backend._model._hidden_states is hidden


def test_invalid_detail_rejected_without_patching():
    _, backend, _, _, _ = fake_profiler()
    with pytest.raises(ValueError, match="detail"):
        DFlashProfiler(backend, detail="unknown")


def test_inherited_module_call_is_wrapped_once_without_subclass_attribute_leak():
    profiler, backend, _, _, _ = fake_profiler()
    class LocalDraft(FakeModel):
        bind = FakeDraft.bind

    backend._draft = LocalDraft(backend._model.lm_head, [FakeLayer()])
    originally_owned = "__call__" in LocalDraft.__dict__
    original_call = LocalDraft.__call__
    tensor = FakeTensor()
    try:
        with profiler.installed():
            with profiler.stage("draft", phase_state()):
                assert backend._draft(tensor) is tensor
        events = [event for event in profiler.recorder.spans
                  if event.metadata.get("kind") == "module" and event.name == "draft."]
        assert len(events) == 1
        assert LocalDraft.__call__ is original_call
        assert ("__call__" in LocalDraft.__dict__) == originally_owned
    finally:
        # Preserve this test process if the restoration assertion detects a leak.
        if not originally_owned and "__call__" in LocalDraft.__dict__:
            del LocalDraft.__call__


def test_ast_rejects_verify_stream_moved_before_capture_boundary(monkeypatch):
    source = inspect.getsource(specimen)
    capture = '''            if _capture is not None:
                observer_hook(events)
                if fail:
                    raise RuntimeError("verification failed")
'''
    verify = '''            with mx.stream(generation_stream):
                events.append("verify compute")
'''
    assert capture + verify in source
    monkeypatch.setattr("inference_lab.profiling.dflash.inspect.getsource",
                        lambda function: source.replace(capture + verify, verify + capture))
    with pytest.raises(ValueError, match="ordering"):
        instrument_stream(specimen, PhaseProbe())


def test_operator_timing_nests_under_module_and_unwinds_after_failure():
    profiler, backend, _, _, _ = fake_profiler()
    profiler._register_modules()
    tensor = FakeTensor()

    def module_operation(module, value):
        return profiler._operation_call("gated_delta_update", lambda arg: arg, (value,), {})

    with profiler.stage("draft", phase_state()):
        result = profiler._module_call(module_operation, backend._draft.lm_head, (tensor,), {})
    assert result is tensor
    events = profiler.recorder.spans
    module = next(event for event in events if event.metadata.get("kind") == "module")
    operation = next(event for event in events if event.metadata.get("kind") == "operation")
    assert operation.parent_id == module.id
    assert operation.name == "draft.lm_head.gdn_0.gated_delta_update"
    assert operation.metadata["inputs"][0]["shape"] == [1, 3, 8]
    assert not profiler._module_stack

    def fail(module, value):
        raise RuntimeError("module failed")

    with pytest.raises(RuntimeError, match="module failed"):
        with profiler.stage("verify", phase_state()):
            profiler._module_call(fail, backend._model.lm_head, (tensor,), {})
    assert not profiler._module_stack
    assert profiler.phase is None and profiler.event is None
    assert profiler.recorder.to_dict()["spans"][-1]["status"] == "error"


def test_runner_summary_uses_decode_tokens_and_phase_roots_only():
    from inference_lab.profiling.runner import compact_summary

    measurement = {
        "generated_tokens": 128, "decode_tokens": 127, "decode_seconds": 4.0,
        "speculative_rounds": 47, "draft_acceptance_rate": 0.85,
    }
    row = {
        "mode": "phases", "block_size": 3, "prompt_index": 0, "repeat": 0,
        "matches_baseline": True, "measurement": measurement,
        "profile": {"unattributed_decode_ms": 12, "spans": [
            {"name": "draft", "inclusive_ms": 10,
             "metadata": {"kind": "phase", "first_round": True}},
            {"name": "draft", "inclusive_ms": 20,
             "metadata": {"kind": "phase", "first_round": False}},
            {"name": "draft", "inclusive_ms": 999,
             "metadata": {"kind": "module", "first_round": False}},
        ]},
    }
    summary = compact_summary({"runs": [row]})
    run = summary["runs"][0]
    assert run["tok_s"] == 127 / 4
    assert run["ms_per_token"] == 4000 / 127
    assert run["useful_tokens_per_round"] == 127 / 47
    phases = summary["phase_by_block"]["3"]
    assert phases["all"]["draft"]["mean"] == 15
    assert phases["all"]["draft"]["n"] == 2
    assert phases["first"]["draft"]["mean"] == 10
    assert phases["steady"]["draft"]["mean"] == 20
    assert phases["all"]["verify"] is None
    assert phases["unattributed_ms_per_run"]["mean"] == 12
