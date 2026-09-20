"""CPU checks for confidence gating, native drafter rollback and trace truth."""
import ast
from contextlib import nullcontext
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from inference_lab.backends.apple.speculative.confidence_backend import ConfidenceMTPBackend
from inference_lab.visualization.recording import validate_generation_trace


class Device:
    uint32 = np.uint32
    int32 = np.int32
    float32 = np.float32
    array = staticmethod(np.array)
    concatenate = staticmethod(np.concatenate)
    take_along_axis = staticmethod(np.take_along_axis)
    argmax = staticmethod(np.argmax)

    def __init__(self):
        self.wired_limit = 100

    def logsumexp(self, values, axis):
        maximum = np.max(values, axis=axis, keepdims=True)
        return np.squeeze(maximum, axis=axis) + np.log(np.sum(np.exp(values - maximum), axis=axis))

    def stream(self, value):
        return nullcontext()

    def eval(self, *values):
        pass

    async_eval = eval
    synchronize = eval
    clear_cache = eval
    reset_peak_memory = eval

    def get_peak_memory(self):
        return 1_000_000

    def device_info(self):
        return {"max_recommended_working_set_size": 200}

    def set_wired_limit(self, limit):
        old, self.wired_limit = self.wired_limit, limit
        return old


class Cache:
    def __init__(self):
        self.tokens = []
        self.trims = []

    @property
    def state(self):
        return [np.array(self.tokens)]

    @property
    def offset(self):
        return len(self.tokens)

    def trim(self, count):
        self.trims.append(count)
        self.tokens = self.tokens[:-count] if count else self.tokens

    def empty(self):
        return not self.tokens


class Target:
    vocab = 64

    def __call__(self, inputs, *, cache, **kwargs):
        cache[0].tokens.extend(inputs.reshape(-1).tolist())
        logits = np.zeros((*inputs.shape, self.vocab), dtype=np.float32)
        np.put_along_axis(logits, (inputs + 1)[..., None], 10, axis=-1)
        return SimpleNamespace(logits=logits, hidden_states=[inputs[..., None]], shared_kv_states={})

    def speculative_logits_from_hidden(self, hidden):
        chosen = hidden[..., 0].astype(np.int32)
        probability = hidden[..., 1]
        logits = np.zeros((*chosen.shape, self.vocab), dtype=np.float32)
        values = np.log(probability * (self.vocab - 1) / (1 - probability))
        np.put_along_axis(logits, chosen[..., None], values[..., None], axis=-1)
        return logits


class Draft:
    def __init__(self, probabilities=None, overrides=None):
        self.probabilities = probabilities or {}
        self.overrides = overrides or {}
        self.fail_greedy = False
        self.cache_history = []
        self.config = SimpleNamespace(block_size=3)
        self.reset(None)

    def reset(self, model):
        self._cache = [Cache()]
        self.cache_history.append(self._cache[0])
        self._seed_hidden = self._seed_token = None
        self._round_appended = self._next_position = self._draft_round = 0
        self.accept_lens, self.draft_lens = [], []
        self._input_embed = object()
        self._lm_head_fn = lambda hidden: None
        self._greedy_argmax_fn = self.choose

    def choose(self, hidden):
        if self.fail_greedy:
            raise RuntimeError("draft argmax failed")
        return hidden[..., 0].astype(np.uint32)

    def draft_eval_state(self):
        return [self._seed_hidden, self._seed_token, self._cache[0].state]

    def _forward_tokens(self, tokens, hidden, token_dtype):
        values = [int(value) for value in tokens.reshape(-1).tolist()]
        self._cache[0].tokens.extend(values)
        self._next_position += len(values)
        return np.array([[[self.overrides.get(value, value + 1), self.probabilities.get(value, .9)]
                          for value in values]])

    _forward_token = _forward_tokens


@pytest.fixture(autouse=True)
def native_drafter_methods(monkeypatch):
    # Execute only the installed pure-Python orchestration functions with NumPy
    # arrays. This tests the real trim/seed code without importing MLX/Metal.
    try:
        package = distribution("mlx-vlm")
    except PackageNotFoundError:
        pytest.skip("Native MTP orchestration source is not installed; no MLX import is needed")
    path = package.locate_file("mlx_vlm/speculative/drafters/qwen3_5_mtp/qwen3_5_mtp.py")
    tree = ast.parse(Path(path).read_text())
    source_class = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    methods = {"_greedy_token", "_set_seed_from_hidden", "prefill_from_target_hidden",
               "accept_verified_tokens", "draft_block", "set_shared_kv"}
    nodes = [node for node in source_class.body if isinstance(node, ast.FunctionDef) and node.name in methods]
    unit = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes], type_ignores=[])
    scope = {"mx": Device()}
    exec(compile(ast.fix_missing_locations(unit), str(path), "exec"), scope)
    assert len(nodes) == len(methods)
    for name in methods:
        monkeypatch.setattr(Draft, name, scope[name], raising=False)


def backend(policy="collect", threshold=.5, *, probabilities=None, overrides=None, fail_walk=False):
    instance = ConfidenceMTPBackend("/tmp/target", "/tmp/draft", policy=policy, threshold=threshold, prefill_step_size=2)
    instance._mx = Device()
    instance._model = Target()
    instance._draft = Draft(probabilities, overrides)
    instance._generation_stream = object()
    instance._tokenizer = SimpleNamespace(eos_token_id=63, decode=lambda ids, **kwargs: str(ids))
    instance.cache_history, instance.transactions = [], []

    def make_cache(model):
        entries = [Cache()]
        instance.cache_history.append(entries[0])
        return entries

    instance._make_prompt_cache = make_cache

    class RNG:
        def __init__(self, *args, **kwargs):
            pass

        def draft_call(self, fn, *args, **kwargs):
            return fn(*args, **kwargs)

        draft_tokens = draft_call

    def verify(model, inputs, caches, sampler, **kwargs):
        assert kwargs["sample_target_tokens"] is True
        before = caches[0].tokens[:]
        appended = [int(value) for value in inputs.reshape(-1)]
        caches[0].tokens.extend(appended)
        transaction = SimpleNamespace(hidden=inputs[..., None], target_tokens=inputs + 1,
                                      shared_kv_states={}, aborted=False, committed=False, inputs=appended)
        instance.transactions.append(transaction)

        def commit(model, caches, accepted, block_size):
            assert block_size == len(appended)
            caches[0].tokens = before + appended[:accepted + 1]
            transaction.committed = True
            # Before capped final rounds, drafter state must represent the
            # target's shifted committed context plus the next bonus.
            expected = caches[0].tokens[1:] + [int(transaction.target_tokens[0, accepted])]
            if instance._draft._cache[0].tokens[-1] == expected[-1]:
                assert instance._draft._cache[0].tokens == expected

        def abort():
            if not transaction.committed:
                caches[0].tokens = before
                transaction.aborted = True

        transaction.commit, transaction.abort = commit, abort
        return transaction

    def walk(model, result, draft_tokens, sampler, budget, **kwargs):
        if fail_walk:
            raise RuntimeError("target walk failed")
        ids = [int(value) for value in draft_tokens.reshape(-1)]
        target = [int(value) for value in result.target_tokens.reshape(-1)]
        accepted = 0
        for proposed, actual in zip(ids, target):
            if proposed != actual:
                break
            accepted += 1
        return accepted, (ids[:accepted] + target[accepted:accepted + 1])[:budget]

    def record(draft, accepted, drafted):
        draft.accept_lens.append(accepted)
        draft.draft_lens.append(drafted)

    instance._mtp_module = SimpleNamespace(
        _buffer_mtp_target_cache=lambda *a: None, _SpeculativeSamplerRNG=RNG,
        _mtp_draft_hidden=lambda model, hidden: hidden,
        _mtp_cache_offset_max=lambda caches: caches[0].offset,
        _mtp_draft_position=lambda value: value - 1,
        _mtp_verify_target=verify, _mtp_acceptance_walk=walk,
        _record_speculative_round=record,
        _slice_shared_kv_after_reject=lambda shared, rejected: shared,
    )
    instance._run_rounds = instance._confidence_rounds
    return instance


@pytest.mark.parametrize("policy,threshold", [("collect", 0), ("collect", 1), ("gate", 0), ("gate", 1), ("no_draft", .5)])
@pytest.mark.parametrize("budget", [1, 2, 3, 7])
def test_greedy_output_trace_and_budget_are_exact(policy, threshold, budget):
    instance = backend(policy, threshold)
    result = instance.measure([1, 2], max_new_tokens=budget)
    assert result["generated_token_ids"] == list(range(3, 3 + budget))
    validate_generation_trace(result)
    assert result["decode_tokens"] == budget - 1
    assert instance._mx.wired_limit == 100
    assert instance._confidence_recorder is None
    assert "_greedy_token" not in vars(instance._draft)
    assert instance._draft._seed_hidden is None
    rounds = result["confidence_rounds"]
    if policy == "no_draft":
        assert result["computed_draft_tokens"] == result["drafted_tokens"] == 0
        assert result["fallback_rounds"] == budget - 1
        assert all(row["p1"] is row["p2"] is row["accepted_count"] is None for row in rounds)
    elif policy == "gate" and threshold == 1:
        assert result["drafted_tokens"] == result["accepted_draft_tokens"] == 0
        assert result["computed_draft_tokens"] == result["gated_out_draft_tokens"]
        assert all(row["accepted_count"] is None and row["decision"] == "fallback" for row in rounds)
        assert all(event["type"] == "commit" for event in result["generation_trace"]["events"])
    else:
        assert result["fallback_rounds"] == 0
        assert all(row["accepted_count"] == len(row["proposal_token_ids"]) for row in rounds)
    for row in rounds:
        assert row["confidence_product"] is None or 0 <= row["confidence_product"] <= 1
        for value in row["timing"].values():
            assert np.isfinite(value) and value >= 0


@pytest.mark.parametrize("overrides,accepted", [({3: 10}, 0), ({4: 10}, 1), ({}, 2)])
def test_observed_acceptance_comes_from_target_prefix(overrides, accepted):
    instance = backend(overrides=overrides)
    result = instance.measure([1, 2], max_new_tokens=8)
    assert result["generated_token_ids"] == list(range(3, 11))
    assert result["confidence_rounds"][0]["accepted_count"] == accepted
    validate_generation_trace(result)


def test_gate_transitions_rollback_partial_draft_then_resume_verification():
    instance = backend("gate", .5, probabilities={3: .2})
    result = instance.measure([1, 2], max_new_tokens=8)
    rounds = result["confidence_rounds"]
    assert rounds[0]["decision"] == "fallback"
    assert rounds[0]["p1"] == pytest.approx(.2, abs=1e-6)
    assert rounds[0]["p2"] == pytest.approx(.9, abs=1e-6)
    assert rounds[0]["confidence_product"] == pytest.approx(.18, abs=1e-6)
    assert rounds[0]["accepted_count"] is None
    assert rounds[1]["decision"] == "verify"
    assert any(cache.trims == [1] for cache in instance._draft.cache_history)
    assert len(instance.transactions[0].inputs) == 1
    assert len(instance.transactions[1].inputs) == 3
    assert result["generated_token_ids"] == list(range(3, 11))
    validate_generation_trace(result)


def test_failed_target_verification_aborts_transaction_and_clears_request_state():
    instance = backend(fail_walk=True)
    with pytest.raises(RuntimeError, match="target walk failed"):
        instance.measure([1, 2], max_new_tokens=7)
    assert instance.transactions[0].aborted
    assert instance.cache_history[0].tokens == [1, 2]
    assert instance._mx.wired_limit == 100
    assert instance._confidence_recorder is None
    assert instance._draft._seed_hidden is None
    assert "_greedy_token" not in vars(instance._draft)


def test_proposal_hook_restored_if_original_argmax_raises():
    instance = backend()
    original = instance._draft._greedy_token
    instance._draft.fail_greedy = True
    with pytest.raises(RuntimeError, match="draft argmax failed"):
        with instance._proposal_hidden_capture():
            instance._draft._greedy_token(np.array([[[4, .9]]]))
    assert instance._draft._greedy_token == original
    assert "_greedy_token" not in vars(instance._draft)


def test_new_request_has_fresh_cache_rounds_and_first_seed_confidence():
    instance = backend(probabilities={3: .3})
    first = instance.measure([1, 2], max_new_tokens=4)
    second = instance.measure([1, 2], max_new_tokens=4)
    assert first["confidence_rounds"] is not second["confidence_rounds"]
    assert instance.cache_history[0] is not instance.cache_history[1]
    assert first["confidence_rounds"][0]["p1"] == pytest.approx(.3, abs=1e-6)
    assert first["generated_token_ids"] == second["generated_token_ids"]


def test_confidence_scores_use_original_chosen_id_and_stable_normalization():
    instance = backend()
    instance._model.speculative_logits_from_hidden = lambda hidden: np.array([[[10000., 10001., 9999.]]])
    score = instance._proposal_logprobs([np.zeros((1, 1, 2))], np.array([[0]]))[0]
    expected = -np.log(np.exp(0.) + np.exp(1.) + np.exp(-1.))
    assert score == pytest.approx(expected, abs=.001)
    assert score < -1  # Probability of chosen index 0, not the maximal index 1.


@pytest.mark.parametrize("options", [
    {"threshold": True}, {"threshold": float("nan")}, {"threshold": -.1}, {"threshold": 1.1},
    {"policy": "sample"}, {"block_size": 2}, {"block_size": True},
])
def test_invalid_policies_fail_closed(options):
    with pytest.raises(ValueError):
        ConfidenceMTPBackend("/tmp/target", "/tmp/draft", **options)
