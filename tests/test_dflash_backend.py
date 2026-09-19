"""CPU-only tests for official DFlash phase boundaries and acceptance accounting."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from inference_lab.backends.apple.speculative.dflash_backend import (
    DFlashBackend, _RoundObserver, _temporary_attribute,
)


class FakeDevice:
    uint32 = "uint32"

    def __init__(self, events):
        self.events = events

    def array(self, values, dtype=None):
        return values

    def eval(self, *args):
        self.events.append(("eval", args))

    def synchronize(self):
        self.events.append("sync")

    def clear_cache(self):
        pass

    def reset_peak_memory(self):
        pass

    def get_peak_memory(self):
        return 123_000_000


def fake_backend(accepted_rounds, *, block_size=5, fail_after_yield=False):
    """Mirror upstream's yield-before-rollback and final empty response."""
    events = []
    backend = DFlashBackend("/tmp/target", "/tmp/draft", block_size=block_size)
    backend._mx = FakeDevice(events)
    backend._model = SimpleNamespace(_hidden_states=["retained hidden"])
    backend._draft = object()
    backend._timing_tokenizer = object()
    backend._tokenizer = SimpleNamespace(decode=lambda ids, **kw: ",".join(map(str, ids)))

    class Capture:
        def rollback(self, cache, accepted, trim):
            events.append(("rollback", accepted, trim))
            cache[0].state = ["lazy final rollback"]

        def close(self):
            events.append("capture close")

    @contextmanager
    def wired_limit(*args):
        events.append("wire")
        try:
            yield
        finally:
            events.append("unwire")

    upstream = SimpleNamespace(
        _GDNStateCapture=Capture,
        make_prompt_cache=lambda model: [SimpleNamespace(state=["cache"])],
        can_trim_prompt_cache=lambda cache: False,
        generation_stream=object(), wired_limit=wired_limit,
    )

    def stream(model, draft, tokenizer, prompt, *, max_tokens, **kwargs):
        target_cache = upstream.make_prompt_cache(model)
        upstream.make_prompt_cache(draft)
        events.append("prefill")
        yield SimpleNamespace(tokens=[10], accepted=None, finish_reason=None)
        events.append("decode")
        capture = upstream._GDNStateCapture()
        n = 1
        try:
            for accepted in accepted_rounds:
                if n >= max_tokens:
                    break
                bs = min(block_size, max_tokens - n + 1)
                emitted = min(accepted + 1, max_tokens - n)
                output = list(range(10 + n, 10 + n + emitted))
                n += emitted
                yield SimpleNamespace(tokens=output, accepted=emitted, finish_reason=None)
                if fail_after_yield:
                    raise RuntimeError("verify failed")
                trim = bs - accepted - 1
                if trim:
                    capture.rollback(target_cache, accepted, trim)
            yield SimpleNamespace(tokens=[], accepted=None, finish_reason="length")
        finally:
            capture.close()

    upstream._stream_generate = stream
    backend._upstream = upstream
    return backend, events, Capture, upstream.make_prompt_cache


def test_zero_partial_and_all_acceptance_with_final_capped_full_acceptance(monkeypatch):
    # N=9: first + rejection(1) + partial(3) + all-accepted capped(4).
    backend, events, original_capture, original_cache = fake_backend([0, 2, 4])
    ticks = iter([1.0, 3.0, 4.0, 9.0])
    monkeypatch.setattr("inference_lab.backends.apple.speculative.dflash_backend.perf_counter", lambda: next(ticks))
    result = backend.measure([1, 2, 3], max_new_tokens=9)
    assert result["generated_token_ids"] == list(range(10, 19))
    assert result["decode_tokens"] == 8
    assert result["prefill_seconds"] == 2.0
    assert result["decode_seconds"] == 5.0
    assert result["drafted_tokens"] == 12
    assert result["accepted_draft_tokens"] == 6
    assert result["draft_acceptance_rate"] == 0.5
    assert result["emitted_draft_tokens"] == 6
    assert result["emitted_target_tokens"] == 3
    assert result["speculative_rounds"] == 3
    assert result["speculative_round_details"][-1]["accepted_draft_tokens"] == 4
    # First-token sync precedes all draft/verify work.
    first_sync = events.index("sync", events.index("prefill"))
    assert first_sync < events.index("decode")
    assert events.index("capture close") < len(events) - 1
    assert events[-1] == "unwire"
    assert backend._upstream._GDNStateCapture is original_capture
    assert backend._upstream.make_prompt_cache is original_cache
    assert backend._model._hidden_states == [None]


def test_capped_rejection_uses_rollback_to_disambiguate_accepted_count():
    backend, events, _, _ = fake_backend([2])
    result = backend.measure([1], max_new_tokens=4)
    # Three tokens emitted could be 3 accepted drafts, or 2 plus target bonus.
    assert result["drafted_tokens"] == 3
    assert result["accepted_draft_tokens"] == 2
    assert result["emitted_target_tokens"] == 2  # first target + bonus
    rollback = events.index(("rollback", 2, 1))
    last_eval = max(i for i, event in enumerate(events) if isinstance(event, tuple) and event[0] == "eval")
    assert rollback < last_eval
    assert "lazy final rollback" in str(events[last_eval])


def test_exception_closes_capture_restores_patches_and_wired_limit():
    backend, events, original_capture, original_cache = fake_backend([0], fail_after_yield=True)
    with pytest.raises(RuntimeError, match="verify failed"):
        backend.measure([1], max_new_tokens=4)
    assert "capture close" in events
    assert events[-1] == "unwire"
    assert backend._upstream._GDNStateCapture is original_capture
    assert backend._upstream.make_prompt_cache is original_cache
    assert backend._model._hidden_states == [None]


def test_one_token_has_no_speculative_decode_or_acceptance():
    backend, events, _, _ = fake_backend([])
    result = backend.measure([1], max_new_tokens=1)
    assert result["decode_tokens"] == 0
    assert result["decode_seconds"] == 0.0
    assert result["drafted_tokens"] == result["accepted_draft_tokens"] == 0
    assert result["draft_acceptance_rate"] is None


def test_incomplete_upstream_output_is_not_reported_as_complete():
    backend, _, _, _ = fake_backend([0])
    with pytest.raises(RuntimeError, match="fixed output budget"):
        backend.measure([1], max_new_tokens=8)


def test_observer_rejects_inconsistent_rollback_contract():
    observer = _RoundObserver(5, 20)
    observer.yielded(1, 2)
    with pytest.raises(RuntimeError, match="rollback accounting"):
        observer.rollback(1, 2)


def test_local_loader_patch_restores_even_on_failure():
    original = object()
    upstream = SimpleNamespace(snapshot_download=original)
    with pytest.raises(RuntimeError):
        with _temporary_attribute(upstream, "snapshot_download", lambda **kw: "/tmp/local"):
            assert upstream.snapshot_download() == "/tmp/local"
            raise RuntimeError("loader failed")
    assert upstream.snapshot_download is original


@pytest.mark.parametrize("kwargs", [
    {"block_size": 1}, {"block_size": 6}, {"block_size": True},
    {"draft_bits": 3}, {"draft_bits": True}, {"kv_bits": 4}, {"wired_memory": False},
])
def test_unsupported_experiment_settings_rejected(kwargs):
    with pytest.raises(ValueError):
        DFlashBackend("/tmp/target", "/tmp/draft", **kwargs)
