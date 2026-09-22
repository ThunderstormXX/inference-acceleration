"""CPU tests for live retrieval's cache transaction and committed-only history."""
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from inference_lab.backends.apple.speculative import ngram_backend as module
from inference_lab.backends.apple.speculative.ngram_backend import (
    NgramBackend, _NgramRoundTrace, greedy_verified_prefix, verify_proposal,
)
from inference_lab.optimizations.ngram import LongestSuffixProposer
from inference_lab.visualization.recording import GenerationRecorder, validate_generation_trace


class FakeMX:
    uint32 = np.uint32
    array = staticmethod(np.array)
    argmax = staticmethod(np.argmax)
    eval = staticmethod(lambda *args: None)
    synchronize = staticmethod(lambda *args: None)
    clear_cache = staticmethod(lambda: None)
    reset_peak_memory = staticmethod(lambda: None)
    get_peak_memory = staticmethod(lambda: 5_000_000_000)
    stream = staticmethod(lambda stream: nullcontext())


@pytest.mark.parametrize("targets,accepted,emitted", [
    ([2, 3, 4], 2, [2, 3, 4]),
    ([2, 8, 9], 1, [2, 8]),
    ([8, 9, 10], 0, [8]),
])
def test_commit_receives_draft_count_not_already_incremented_count(targets, accepted, emitted):
    commits, aborts = [], []
    result = SimpleNamespace(target_tokens=np.array([targets]),
        commit=lambda *args: commits.append(args), abort=lambda: aborts.append(True))
    model, cache = object(), object()
    def verify(actual_model, inputs, actual_cache, sampler, **kwargs):
        assert actual_model is model and actual_cache is cache
        assert inputs.tolist() == [[1, 2, 3]]
        assert kwargs == {"sample_target_tokens": True}
        return result
    assert verify_proposal(model, cache, 1, [2, 3], mx=FakeMX, verify_target=verify) == (accepted, emitted)
    assert commits == [(model, cache, accepted, 3)]
    assert aborts == []


@pytest.mark.parametrize("kind", ["missing_tokens", "invalid_count", "commit_failure"])
def test_unfinished_returned_transaction_aborts(kind):
    aborts = []
    def commit(*args):
        if kind == "commit_failure":
            raise RuntimeError("commit failed")
    target_tokens = None if kind == "missing_tokens" else np.array([[2]] if kind == "invalid_count" else [[2, 3]])
    result = SimpleNamespace(target_tokens=target_tokens, commit=commit, abort=lambda: aborts.append(True))
    with pytest.raises((ValueError, RuntimeError)):
        verify_proposal(object(), [], 1, [2], mx=FakeMX, verify_target=lambda *args, **kwargs: result)
    assert aborts == [True]


def test_forward_failure_is_left_to_upstream_transaction_owner():
    def fail(*args, **kwargs):
        raise RuntimeError("upstream already aborted")
    with pytest.raises(RuntimeError, match="upstream already aborted"):
        verify_proposal(object(), [], 1, [2], mx=FakeMX, verify_target=fail)


@pytest.mark.parametrize("proposal,targets", [([], [1]), ([1], [1]), ([True], [1, 2]), ([1], [True, 2])])
def test_invalid_prediction_shapes_and_types_rejected(proposal, targets):
    with pytest.raises(ValueError):
        greedy_verified_prefix(proposal, targets)


@pytest.mark.parametrize("traced", [False, True])
def test_rejected_speculative_inputs_never_leak_into_next_round_or_retrieval(monkeypatch, traced):
    clock_calls = []
    def clock():
        value = 100.0 + len(clock_calls) * 0.125
        clock_calls.append(value)
        return value
    monkeypatch.setattr(module, "perf_counter", clock)
    class Cache:
        def __init__(self):
            self.tokens = []
        @property
        def state(self):
            return np.array(self.tokens)
    cache = Cache()
    def oracle(ids):
        return (sum(ids) + len(ids)) % 7
    class Model:
        def __call__(self, inputs, cache):
            cache[0].tokens.extend(int(t) for t in inputs.reshape(-1))
    model = Model()
    histories = []
    class ForcedWrongProposer:
        def __init__(self, **kwargs):
            self.history = []
        @property
        def committed_tokens(self):
            return len(self.history)
        def commit(self, tokens):
            self.history.extend(tokens)
        def propose(self, width):
            histories.append(list(self.history))
            return SimpleNamespace(token_ids=(9,) * width, matched_suffix_length=2,
                                   source_start=0, source_end=width)
    monkeypatch.setattr(module, "LongestSuffixProposer", ForcedWrongProposer)
    backend = NgramBackend("unused", width=2)
    backend.trace_generation = traced
    backend._model, backend._mx, backend._generation_stream = model, FakeMX, None
    backend._make_prompt_cache = lambda model: [cache]
    backend._tokenizer = SimpleNamespace(decode=lambda ids, **kwargs: str(ids))
    def next_token(inputs, caches):
        model(inputs, caches)
        return np.array([oracle(caches[0].tokens)])
    backend._next_token = next_token
    committed = []
    def verify(model, inputs, caches, sampler, **kwargs):
        start = len(cache.tokens)
        targets = []
        for token in inputs.reshape(-1):
            cache.tokens.append(int(token))
            targets.append(oracle(cache.tokens))
        def commit(model, caches, accepted, block_size):
            committed.append((accepted, block_size))
            del cache.tokens[start + accepted + 1:]
        def abort():
            del cache.tokens[start:]
        return SimpleNamespace(target_tokens=np.array([targets]), commit=commit, abort=abort)
    backend._verify_target = verify
    prompt = [1, 2, 1, 2]
    expected = []
    for i in range(16):
        expected.append(oracle(prompt + expected))
    result = backend._measure(prompt, 16)
    assert result["generated_token_ids"] == expected
    assert cache.tokens == prompt + expected[:-1]
    assert all(a == 0 for a, block in committed)
    assert result["accepted_draft_tokens"] == 0
    assert result["speculative_rounds"] == 14
    assert result["fallback_rounds"] == 1
    for index, history in enumerate(histories, 1):
        assert history == prompt + expected[:index]
        assert 9 not in history
    assert sum(len(r["emitted_token_ids"]) for r in result["round_details"]) == 15
    if traced:
        validate_generation_trace(result)
        lane = result["generation_trace"]
        drafts = [event for event in lane["events"] if event["type"] == "draft"]
        assert len(drafts) == 14
        assert lane["events"][-1]["type"] == "commit"
        assert lane["events"][-1]["round"] == 0
        assert lane["events"][-1]["ngram_round"] == 15
        # Prefill/decode use one contiguous request clock, without an omitted
        # interval between reading prefill_seconds and starting decode.
        assert lane["total_seconds"] == clock_calls[-1] - clock_calls[0]
        backend._make_prompt_cache = lambda model: [Cache()]
        one = backend._measure(prompt, 1)
        validate_generation_trace(one)
        assert one["decode_seconds"] == 0
        assert len(one["generation_trace"]["events"]) == 1
    else:
        assert "generation_trace" not in result


def test_real_proposer_does_not_observe_or_commit_rejected_suffix():
    proposer = LongestSuffixProposer(min_match=2, max_match=4)
    proposer.commit([1, 2, 3, 4, 1, 2])
    assert proposer.propose(2).token_ids == (3, 4)
    accepted, emitted = greedy_verified_prefix([3, 4], [9, 8, 7])
    assert accepted == 0
    proposer.commit(emitted)
    assert proposer.committed_tokens == 7
    assert proposer.propose(2).token_ids == ()


@pytest.mark.parametrize("targets,accepted,emitted", [
    ([2, 3, 4], 2, [2, 3, 4]),
    ([2, 8, 9], 1, [2, 8]),
    ([8, 9, 10], 0, [8]),
])
def test_traced_verification_events_validate_for_all_acceptance_outcomes(targets, accepted, emitted):
    ticks = iter(range(101, 1000))
    backend = SimpleNamespace(tokenizer=SimpleNamespace(eos_token_id=None), _model_config={})
    recorder = GenerationRecorder(backend, clock=lambda: float(next(ticks)), speculative=True)
    recorder.start_at(100)
    recorder.commit([1], round=0)
    prefill = recorder.now()
    observer = _NgramRoundTrace(recorder, [2, 3], 1)
    transaction_events = []
    result = SimpleNamespace(target_tokens=np.array([targets]),
                             commit=lambda *args: transaction_events.append("commit"),
                             abort=lambda: transaction_events.append("abort"))
    actual_accepted, actual_emitted = verify_proposal(
        object(), [], 1, [2, 3], mx=FakeMX,
        verify_target=lambda *args, **kwargs: result, observer=observer)
    assert (actual_accepted, actual_emitted) == (accepted, emitted)
    observer.finish(actual_emitted, actual_accepted)
    recorder.commit([77], round=0, ngram_round=2, accepted_count=0, draft_count=0)
    ids = [1, *emitted, 77]
    measured = {"generated_token_ids": ids, "generated_tokens": len(ids),
                "prefill_seconds": prefill, "decode_seconds": recorder.now() - prefill,
                "drafted_tokens": 2, "accepted_draft_tokens": accepted, "speculative_rounds": 1}
    measured["generation_trace"] = recorder.finish_measurement(measured)
    validate_generation_trace(measured)
    draft, commit = recorder.events[1:3]
    assert draft["token_ids"] == commit["proposed_token_ids"] == [2, 3]
    assert commit["rejected_token_ids"] == [2, 3][accepted:]
    assert commit["target_token_ids"] == targets
    assert draft["t"] < commit["verification_completed_t"] < commit["cache_commit_enqueued_t"] <= commit["t"]
    assert transaction_events == ["commit"]
