"""CPU-only contracts for fused head boundaries and the pinned DFlash rewrite."""
import ast
from contextlib import contextmanager
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from inference_lab.optimizations.greedy_head import ScopedGreedyHead, transform_greedy_stream


class Tensor:
    def __init__(self, label, shape=(1, 3, 4096)):
        self.label, self.shape = label, shape
        self.ndim = len(shape)

    def __getitem__(self, key):
        if key is None:
            return Tensor(self.label, (1, 1))
        if isinstance(key, tuple) and key[-1] == 0:
            return Tensor(self.label, (self.shape[0],))
        return Tensor(self.label, (1, 1, self.shape[-1]))


class Head:
    def __init__(self, events):
        self.events = events

    def __call__(self, hidden):
        self.events.append("full head")
        return Tensor("full logits", (*hidden.shape[:2], 248320))


class Target:
    def __init__(self, events):
        self.events = events
        self.language_model = SimpleNamespace(args=SimpleNamespace(tie_word_embeddings=False),
                                              model=self.hidden, lm_head=Head(events))

    def hidden(self, inputs, cache):
        self.events.append("target hidden")
        return Tensor("target hidden")

    def __call__(self, inputs, cache):
        return self.language_model.lm_head(self.hidden(inputs, cache))


class Draft:
    def __init__(self, events, head):
        self.events, self.lm_head = events, head
        self.config = SimpleNamespace(output_multiplier=1.0, final_logit_softcapping=None)

    def hidden_states(self, block, hidden, cache, logits_start=0):
        self.events.append("draft hidden")
        return Tensor("draft hidden", (1, 2, 4096))

    def compute_logits(self, hidden):
        self.events.append("draft full logits")
        return self.lm_head(hidden)

    def __call__(self, block, hidden, cache, logits_start=0):
        return self.compute_logits(self.hidden_states(block, hidden, cache, logits_start))


class DFlash2DraftModel(Draft):
    pass


def observer_hook(events):
    events.append("original observer")


mx = SimpleNamespace(argmax=lambda logits, axis=-1: Tensor("fallback", logits.shape[:2]))


def _stream_generate(model, draft, tokenizer, prompt, block_size=3, max_tokens=2,
                     temperature=0.0, top_p=1.0, top_k=0, prefill_step_size=512):
    observer_hook(model.events)
    yield "stock prefill"
    n = 1
    hidden = prompt
    target_cache, draft_cache = [], []
    while n < max_tokens:
        block = prompt
        draft_logits = draft(block, hidden, draft_cache, logits_start=1)
        if temperature > 0:
            draft_tokens = "sampling is unreachable"
        else:
            draft_tokens = mx.argmax(draft_logits, axis=-1)
        verify_input = prompt
        logits = model(verify_input, target_cache)
        if temperature > 0:
            target_probs = "sampling is unreachable"
        else:
            target_tokens = mx.argmax(logits, axis=-1)
        yield (draft_tokens, target_tokens)
        n += 1


class Backend:
    def __init__(self, *, dflash=False, verifier_fallback=False):
        self.events = []
        self._model = Target(self.events)
        self._mx = mx
        self._draft = Draft(self.events, self._model.language_model.lm_head) if dflash else None
        self._upstream = SimpleNamespace(_stream_generate=_stream_generate,
            DFlash2DraftModel=DFlash2DraftModel) if dflash else None
        self.verifier = SimpleNamespace(optimized_affine_argmax=(
            lambda head, hidden: None if verifier_fallback else Tensor("fused", hidden.shape[:2])))

    def _next_token(self, tokens, cache):
        self.events.append("stock next")
        return Tensor("stock", (1,))

    def _quantize_cache(self, cache):
        self.events.append("quantize cache")

    def _measure(self, count=3):
        cache, token = [], Tensor("input", (1,))
        return [self._next_token(token, cache).label for _ in range(count)]

    @contextmanager
    def _observe(self, observer):
        namespace = _stream_generate.__globals__
        previous = namespace["observer_hook"]
        namespace["observer_hook"] = lambda events: events.append("installed observer")
        try:
            yield
        finally:
            namespace["observer_hook"] = previous


def test_ar_keeps_each_request_prefill_stock_and_restores_lookup():
    backend = Backend()
    original_next = backend._next_token.__func__
    original_measure = backend._measure.__func__
    with ScopedGreedyHead(backend, backend.verifier) as optimization:
        assert backend._measure() == ["stock", "fused", "fused"]
        assert backend._measure(count=2) == ["stock", "fused"]
    assert backend._next_token.__func__ is original_next
    assert backend._measure.__func__ is original_measure
    assert "_next_token" not in vars(backend)
    assert "_measure" not in vars(backend)
    assert optimization.metadata()["stock_ar_prefill_calls"] == 2
    assert optimization.metadata()["optimized_argmax_calls"] == {"ar:T1": 3}
    assert backend.events.count("quantize cache") == 3
    assert "full head" not in backend.events


def test_ar_kernel_fallback_uses_full_head_and_counts():
    backend = Backend(verifier_fallback=True)
    with ScopedGreedyHead(backend, backend.verifier) as optimization:
        assert backend._measure(count=2) == ["stock", "fallback"]
    assert backend.events.count("full head") == 1
    assert optimization.metadata()["fallback_argmax_calls"] == {"ar:T1": 1}
    optimization.reset_counters()
    assert optimization.metadata()["fallback_argmax_calls"] == {}
    assert optimization.metadata()["stock_ar_prefill_calls"] == 0


def test_dflash_copies_globals_after_observer_install_and_keeps_prefill():
    backend = Backend(dflash=True)
    original_stream = backend._upstream._stream_generate
    original_observe = backend._observe.__func__
    with ScopedGreedyHead(backend, backend.verifier) as optimization:
        assert backend._upstream._stream_generate is original_stream
        with backend._observe(object()):
            assert backend._upstream._stream_generate is not original_stream
            result = list(backend._upstream._stream_generate(
                backend._model, backend._draft, None, Tensor("prompt")))
        assert backend._upstream._stream_generate is original_stream
    assert backend._observe.__func__ is original_observe
    assert result[0] == "stock prefill"
    assert [value.label for value in result[1]] == ["fused", "fused"]
    assert backend.events == ["installed observer", "draft hidden", "target hidden"]
    metadata = optimization.metadata()
    assert metadata["optimized_argmax_calls"] == {"draft:T2": 1, "target_verify:T3": 1}
    assert len(metadata["dflash_stream_source_sha256"]) == 64
    assert len(metadata["dflash_transformed_source_sha256"]) == 64


def test_dflash_kernel_fallback_keeps_original_full_head():
    backend = Backend(dflash=True, verifier_fallback=True)
    with ScopedGreedyHead(backend, backend.verifier) as optimization:
        with backend._observe(object()):
            result = list(backend._upstream._stream_generate(
                backend._model, backend._draft, None, Tensor("prompt")))
    assert [value.label for value in result[1]] == ["fallback", "fallback"]
    assert backend.events.count("full head") == 2
    assert backend.events.count("draft full logits") == 1
    assert optimization.metadata()["fallback_argmax_calls"] == {"draft:T2": 1, "target_verify:T3": 1}


def test_sampling_is_rejected_before_prefill_and_scopes_restore():
    backend = Backend(dflash=True)
    original_stream = backend._upstream._stream_generate
    with pytest.raises(ValueError, match="temperature=0"):
        with ScopedGreedyHead(backend, backend.verifier):
            with backend._observe(object()):
                list(backend._upstream._stream_generate(
                    backend._model, backend._draft, None, Tensor("prompt"), temperature=0.5))
    assert backend.events == []
    assert backend._upstream._stream_generate is original_stream
    assert "_observe" not in vars(backend)
    assert "_next_token" not in vars(backend)
    assert "_measure" not in vars(backend)


@pytest.mark.parametrize("change, message", [
    (lambda backend: setattr(backend._draft.config, "output_multiplier", 2.0), "output_multiplier"),
    (lambda backend: setattr(backend._draft.config, "final_logit_softcapping", 1.0), "softcapping"),
    (lambda backend: setattr(backend._draft.config, "final_logit_softcapping", float("nan")), "softcapping"),
    (lambda backend: setattr(backend, "_draft", DFlash2DraftModel(backend.events, backend._model.language_model.lm_head)), "DFlash2"),
    (lambda backend: setattr(backend._model.language_model.args, "tie_word_embeddings", True), "Tied"),
])
def test_unsupported_model_options_rejected_before_patching(change, message):
    backend = Backend(dflash=True)
    change(backend)
    with pytest.raises(ValueError, match=message):
        ScopedGreedyHead(backend, backend.verifier)
    assert "_next_token" not in vars(backend)


def test_same_context_cannot_be_reentered_and_outer_scope_survives():
    backend = Backend()
    optimization = ScopedGreedyHead(backend, backend.verifier)
    with optimization:
        with pytest.raises(RuntimeError, match="entered twice"):
            with optimization:
                pass
        assert backend._measure(count=2) == ["stock", "fused"]
    assert "_next_token" not in vars(backend)


@pytest.mark.parametrize("edit", [
    lambda source: source.replace("draft_logits = draft(", "draft_logits = changed_draft("),
    lambda source: source.replace("logits = model(verify_input, target_cache)", "logits = model(verify_input, cache=target_cache)"),
    lambda source: source.replace("yield (draft_tokens, target_tokens)", "target_tokens = None\n        yield (draft_tokens, target_tokens)"),
    lambda source: source.replace("while n < max_tokens:", "while n <= max_tokens:"),
])
def test_ast_fail_closed_on_changed_pinned_source(monkeypatch, edit):
    backend = Backend(dflash=True)
    optimization = ScopedGreedyHead(backend, backend.verifier)
    source = edit(inspect.getsource(_stream_generate))
    monkeypatch.setattr(inspect, "getsource", lambda function: source)
    with pytest.raises(ValueError):
        transform_greedy_stream(_stream_generate, optimization)


def test_actual_installed_dflash_stream_parses_without_importing_gpu(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    path = root / ".venv-speculative/lib/python3.13/site-packages/dflash/model_mlx.py"
    if not path.exists():
        pytest.skip("Local pinned DFlash source is not installed")
    source = path.read_text()
    tree = ast.parse(source)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_stream_generate")
    actual = ast.get_source_segment(source, function)
    monkeypatch.setattr(inspect, "getsource", lambda function: actual)
    backend = Backend(dflash=True)
    optimization = ScopedGreedyHead(backend, backend.verifier)
    transformed = transform_greedy_stream(_stream_generate, optimization)
    assert callable(transformed)
    assert optimization.stream_source_sha256
