"""CPU-only checks for MTP feature alignment, phase accounting and cleanup."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from inference_lab.backends.apple.speculative.mtp_backend import MTPBackend, _acceptance_metrics


class Array:
    def __init__(self, values, *, batch=False):
        self.values = list(values)
        self.batch = batch
        self.dtype = "uint32"

    def __getitem__(self, key):
        if key is None:
            return Array(self.values, batch=True)
        return Array(self.values[key], batch=self.batch)

    def item(self):
        assert len(self.values) == 1
        return self.values[0]

    @property
    def shape(self):
        return (1, len(self.values)) if self.batch else (len(self.values),)


class Logits:
    def __init__(self, token):
        self.token = token

    def __getitem__(self, key):
        return self


class Device:
    uint32 = "uint32"

    def __init__(self, events):
        self.events = events
        self.wired_limit = 100

    def array(self, values, dtype=None):
        return Array(values)

    def stream(self, stream):
        return nullcontext()

    def concatenate(self, chunks, axis):
        assert axis == 1
        return Array([item for chunk in chunks for item in chunk.values], batch=True)

    def argmax(self, logits, axis):
        return Array([logits.token])

    def eval(self, *arrays):
        self.events.append(("eval", arrays))

    def synchronize(self):
        self.events.append("sync")

    def clear_cache(self):
        pass

    def reset_peak_memory(self):
        pass

    def get_peak_memory(self):
        return 456_000_000

    def device_info(self):
        return {"max_recommended_working_set_size": 200}

    def set_wired_limit(self, limit):
        previous, self.wired_limit = self.wired_limit, limit
        self.events.append(("wired", limit))
        return previous


class Draft:
    def __init__(self, events):
        self.events = events
        self.accept_lens = []
        self.draft_lens = []
        self.seed = None

    def reset(self, model):
        self.events.append("draft reset")
        self.accept_lens = []
        self.draft_lens = []
        self.seed = None

    def draft_eval_state(self):
        return [self.seed]


class Target:
    def __init__(self, events):
        self.events = events
        self._position_ids = "previous request"
        self._rope_deltas = "previous request"

    def __call__(self, inputs, *, cache, **kwargs):
        assert self._position_ids is None
        assert self._rope_deltas is None
        assert kwargs["return_hidden"] is True
        assert kwargs["return_shared_kv"] is True
        self.events.append(("target", inputs.values, kwargs.get("skip_logits", False)))
        cache[0].state = ["prefill cache"]
        return SimpleNamespace(
            logits=Logits(inputs.values[-1] + 1),
            hidden_states=[Array(inputs.values, batch=True)], shared_kv_states={},
        )


def fake_backend(accepted_rounds, *, block_size=3, fail=False):
    events = []
    backend = MTPBackend("/tmp/target", "/tmp/draft", prefill_step_size=2, block_size=block_size)
    backend._mx = Device(events)
    backend._model = Target(events)
    backend._draft = Draft(events)
    backend._generation_stream = object()
    backend._tokenizer = SimpleNamespace(decode=lambda ids, **kw: ",".join(map(str, ids)))
    caches = []

    def make_cache(model):
        cache = [SimpleNamespace(state=[])]
        caches.append(cache)
        return cache

    backend._make_prompt_cache = make_cache

    def run_rounds(model, draft, cache, full_input_ids, first, logprobs, output, *, max_tokens, **kwargs):
        events.append(("full prompt", full_input_ids.values, output.hidden_states[-1].values))
        assert kwargs["draft_kind"] == "mtp"
        assert kwargs["sampler_is_greedy"] is True
        try:
            yield first.item(), None
            events.append("draft prefill")
            draft.reset(model)
            generated = 1
            for accepted in accepted_rounds:
                if generated >= max_tokens:
                    break
                drafted = min(block_size, max_tokens - generated + 1) - 1
                draft.accept_lens.append(accepted)
                draft.draft_lens.append(drafted)
                emitted = min(accepted + 1, max_tokens - generated)
                cache[0].state = ["lazy committed recurrent state"]
                draft.seed = "lazy next draft seed"
                events.append("commit")
                for _ in range(emitted):
                    yield first.item() + generated, None
                    generated += 1
                    if fail:
                        raise RuntimeError("verify failed")
        finally:
            events.append("rounds close")

    backend._run_rounds = run_rounds
    return backend, events, caches


def test_full_prompt_hidden_capture_and_phase_boundary(monkeypatch):
    backend, events, _ = fake_backend([0, 2, 1])
    ticks = iter([1.0, 3.0, 4.0, 9.0])
    monkeypatch.setattr("inference_lab.backends.apple.speculative.mtp_backend.perf_counter", lambda: next(ticks))
    result = backend.measure([1, 2, 3, 4, 5], max_new_tokens=7)
    assert [(event[1], event[2]) for event in events if isinstance(event, tuple) and event[0] == "target"] == [
        ([1, 2], True), ([3, 4], True), ([5], False),
    ]
    assert ("full prompt", [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) in events
    boundary = events.index("sync", events.index(("full prompt", [1, 2, 3, 4, 5], [1, 2, 3, 4, 5])))
    assert boundary < events.index("draft prefill")
    assert result["generated_token_ids"] == list(range(6, 13))
    assert result["prefill_seconds"] == 2.0
    assert result["decode_seconds"] == 5.0
    assert result["decode_tokens"] == 6
    assert result["accepted_draft_tokens"] == 3
    assert result["drafted_tokens"] == 6
    assert result["emitted_draft_tokens"] == 3
    assert result["emitted_target_tokens"] == 4
    assert backend._mx.wired_limit == 100
    assert backend._draft.accept_lens == []
    last_eval = [event for event in events if isinstance(event, tuple) and event[0] == "eval"][-1]
    assert "lazy committed recurrent state" in str(last_eval)
    assert "lazy next draft seed" in str(last_eval)


def test_capped_final_all_acceptance_excludes_unemitted_bonus():
    backend, _, _ = fake_backend([2])
    result = backend.measure([1], max_new_tokens=3)
    assert result["accepted_draft_tokens"] == result["emitted_draft_tokens"] == 2
    assert result["drafted_tokens"] == 2
    assert result["emitted_target_tokens"] == 1
    assert result["speculative_round_details"][0]["emitted_target_tokens"] == 0


def test_failed_verification_closes_stream_clears_draft_and_restores_limit():
    backend, events, _ = fake_backend([0], fail=True)
    with pytest.raises(RuntimeError, match="verify failed"):
        backend.measure([1, 2], max_new_tokens=4)
    assert "rounds close" in events
    assert backend._draft.seed is None
    assert backend._draft.accept_lens == []
    assert backend._mx.wired_limit == 100


def test_independent_requests_use_fresh_caches_and_acceptance_histories():
    backend, _, caches = fake_backend([2])
    first = backend.measure([1, 2], max_new_tokens=3)
    second = backend.measure([1, 2], max_new_tokens=3)
    assert len(caches) == 2
    assert caches[0] is not caches[1]
    assert first["accepted_draft_tokens"] == second["accepted_draft_tokens"] == 2


def test_single_output_never_prefills_drafter():
    backend, events, _ = fake_backend([])
    result = backend.measure([1], max_new_tokens=1)
    assert "draft prefill" not in events
    assert result["decode_tokens"] == result["speculative_rounds"] == 0
    assert result["decode_seconds"] == 0.0
    assert result["draft_acceptance_rate"] is None


def test_incomplete_generation_raises():
    backend, _, _ = fake_backend([0])
    with pytest.raises(RuntimeError, match="fixed output budget"):
        backend.measure([1], max_new_tokens=4)


@pytest.mark.parametrize("accepted,drafted,output_tokens", [
    ([1], [], 2), ([3], [2], 4), ([True], [1], 2), ([0], [0], 2),
    ([0], [1], 4), ([0, 0], [1, 1], 2),
])
def test_invalid_acceptance_histories_are_rejected(accepted, drafted, output_tokens):
    with pytest.raises(RuntimeError):
        _acceptance_metrics(accepted, drafted, output_tokens)


@pytest.mark.parametrize("kwargs", [{"block_size": 1}, {"block_size": 6}, {"block_size": True}, {"wired_memory": False}])
def test_unsupported_mtp_settings_rejected(kwargs):
    with pytest.raises(ValueError):
        MTPBackend("/tmp/target", "/tmp/draft", **kwargs)
