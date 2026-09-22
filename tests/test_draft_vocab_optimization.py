"""CPU-only calibration isolation, packed-row readout and scope contracts."""
from copy import deepcopy
from hashlib import sha256
import json
from types import SimpleNamespace

import numpy as np
import pytest

from inference_lab.optimizations.draft_vocab import (
    ScopedDraftVocabulary, build_draft_vocabulary, load_draft_vocabulary,
)


def bundle():
    return {"chains": [
        {"source_row_index": row, "source_method": "confidence_collect",
         "prompt_token_ids": [6], "generated_token_ids": [1, 3] * 1024}
        for row in range(100, 110)] + [
        {"source_row_index": 110, "source_method": "ar_before",
         "prompt_token_ids": [5], "generated_token_ids": [4] * 2048}],
        "provenance": [{"path": "artifacts/calibration2048/samples.jsonl", "target_manifest_sha256": "abc"}]}


def build(value=None):
    return build_draft_vocabulary(value or bundle(), {"added_tokens": [
        {"id": 7, "special": True}, {"id": 5, "special": False}]}, {"text_config": {"vocab_size": 8}}, "abc")


def test_vocabulary_uses_calibration_outputs_and_declared_specials_only():
    config = build()
    assert config["token_ids"] == [1, 3, 7]
    assert config["calibration"]["distinct_output_tokens"] == 2
    assert config["calibration"]["total_generated_tokens"] == 20480
    changed = bundle()
    # Neither prompt contents nor heldout output labels enter the shortlist.
    for row in changed["chains"]:
        row["prompt_token_ids"] = [0, 2, 4, 5, 6]
    changed["chains"][-1]["generated_token_ids"] = [0, 2, 4, 5, 6] * 500
    assert build(changed)["token_ids"] == config["token_ids"]


@pytest.mark.parametrize("mutate", [
    lambda value: value["chains"].pop(0),
    lambda value: value["chains"].append(deepcopy(value["chains"][0])),
    lambda value: value["chains"][0]["generated_token_ids"].pop(),
    lambda value: value["chains"][0]["generated_token_ids"].__setitem__(0, True),
    lambda value: value["chains"][0]["generated_token_ids"].__setitem__(0, 8),
    lambda value: value["provenance"][0].__setitem__("target_manifest_sha256", "different"),
])
def test_calibration_rejects_partial_duplicate_invalid_or_different_target(mutate):
    value = bundle()
    mutate(value)
    with pytest.raises(ValueError):
        build(value)


class FakeQuantizedLinear(dict):
    def __init__(self):
        super().__init__()
        self.weight = np.arange(8, dtype=np.float32)[:, None]
        self.scales = np.arange(8, dtype=np.float32)[:, None] + 20
        self.biases = np.arange(8, dtype=np.float32)[:, None] + 40
        self.mode, self.bits, self.group_size = "affine", 4, 64

    def __call__(self, hidden):
        raise AssertionError("Full shared head must not be used for a shortlist draft")


class Qwen3_5MTPDraftModel:
    def __init__(self):
        self._greedy_argmax_fn = object()
    def reset(self, target):
        # Real reset rebinds callbacks but leaves _greedy_token as a method.
        self._greedy_argmax_fn = object()
    def _greedy_token(self, hidden):
        return np.array([[4]], dtype=np.int32)
    def draft_block(self, hidden):
        return self._greedy_token(hidden)


class FakeMX:
    int32 = np.int32
    array = staticmethod(np.array)
    take = staticmethod(np.take)
    argmax = staticmethod(np.argmax)
    def __init__(self):
        self.calls = []
    def quantized_matmul(self, hidden, weight, **kwargs):
        self.calls.append((weight, kwargs))
        return hidden @ weight.T
    def eval(self, *args):
        pass
    def synchronize(self):
        pass


def prepared(tmp_path, *, trace=False):
    model_path = tmp_path / "model"
    model_path.mkdir()
    manifest, tokenizer = b"target manifest", b"tokenizer"
    (model_path / "download-manifest.json").write_bytes(manifest)
    (model_path / "tokenizer.json").write_bytes(tokenizer)
    config = build()
    config["target_manifest_sha256"] = sha256(manifest).hexdigest()
    config["tokenizer_sha256"] = sha256(tokenizer).hexdigest()
    path = tmp_path / "vocab.json"
    path.write_text(json.dumps(config))
    head = FakeQuantizedLinear()
    backend = SimpleNamespace(_draft=Qwen3_5MTPDraftModel(), _model=SimpleNamespace(lm_head=head),
                              _mx=FakeMX(), model_path=str(model_path), trace_generation=trace,
                              _run_rounds=lambda *args: iter([]))
    return backend, path


def test_draft_only_readout_gathers_original_rows_maps_global_ids_and_restores(tmp_path):
    backend, path = prepared(tmp_path)
    head = backend._model.lm_head
    original_weight = head.weight
    original_method = backend._draft._greedy_token.__func__
    optimization = ScopedDraftVocabulary(backend, path, quantized_linear_class=FakeQuantizedLinear)
    np.testing.assert_array_equal(optimization.weight, original_weight[[1, 3, 7]])
    np.testing.assert_array_equal(optimization.scales, head.scales[[1, 3, 7]])
    np.testing.assert_array_equal(optimization.biases, head.biases[[1, 3, 7]])
    with optimization:
        assert backend._draft._greedy_token(np.ones((1, 1, 1))).tolist() == [[7]]
        backend._draft.reset(backend._model)
        assert backend._draft._greedy_token(-np.ones((1, 1, 1))).tolist() == [[1]]
        assert backend._model.lm_head is head
        assert head.weight is original_weight
    assert backend._draft._greedy_token.__func__ is original_method
    assert "_greedy_token" not in vars(backend._draft)
    assert optimization.metadata()["draft_readout_calls"] == 2
    for _, kwargs in backend._mx.calls:
        assert kwargs["bits"] == 4 and kwargs["group_size"] == 64 and kwargs["transpose"] is True


def test_suspended_stock_run_retains_arrays_and_restores_shortlist_on_error(tmp_path):
    backend, path = prepared(tmp_path)
    optimization = ScopedDraftVocabulary(backend, path, quantized_linear_class=FakeQuantizedLinear)
    retained = optimization.weight
    with optimization:
        with pytest.raises(RuntimeError, match="stock failure"):
            with optimization.suspended():
                assert backend._draft._greedy_token(None).tolist() == [[4]]
                assert optimization.weight is retained
                raise RuntimeError("stock failure")
        assert backend._draft._greedy_token(np.ones((1, 1, 1))).tolist() == [[7]]
    with pytest.raises(RuntimeError, match="active"):
        with optimization.suspended():
            pass


def test_standard_trace_observer_records_global_shortlist_ids(tmp_path):
    from inference_lab.visualization.trace import MTPTraceObserver
    backend, path = prepared(tmp_path, trace=True)
    class Verify:
        def commit(self, *args):
            pass
    module = SimpleNamespace(_mtp_verify_target=lambda *args: None, _mtp_acceptance_walk=lambda *args: None,
                             _MTPVerifyResult=Verify)
    recorder = SimpleNamespace(events=[], token_ids=[], now=lambda: 0.01)
    observer = MTPTraceObserver(backend, recorder, module=module, suppress_detokenization=False)
    with ScopedDraftVocabulary(backend, path, quantized_linear_class=FakeQuantizedLinear):
        with observer.observe():
            assert backend._draft.draft_block(np.ones((1, 1, 1))).tolist() == [[7]]
    assert recorder.events[0]["token_ids"] == [7]
    assert "confidence" not in recorder.events[0]


def test_confidence_collection_rejected_before_override(tmp_path):
    backend, path = prepared(tmp_path)
    backend._confidence_records = []
    with pytest.raises(ValueError, match="confidence"):
        ScopedDraftVocabulary(backend, path, quantized_linear_class=FakeQuantizedLinear)
    assert "_greedy_token" not in vars(backend._draft)


def test_head_checkpoint_mismatch_and_invalid_vocab_rejected(tmp_path):
    backend, path = prepared(tmp_path)
    (tmp_path / "model/download-manifest.json").write_text("different")
    with pytest.raises(ValueError, match="different target"):
        ScopedDraftVocabulary(backend, path, quantized_linear_class=FakeQuantizedLinear)
    config = json.loads(path.read_text())
    config["token_ids"] = [3, 1, 7]
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="Malformed"):
        load_draft_vocabulary(path)


def test_trace_cli_accepts_shortlist_with_explicit_metadata(monkeypatch, tmp_path):
    from inference_lab.optimizations import mtp_sweep
    class FakeRunner:
        def __init__(self, config, *args, draft_vocab_path=None):
            assert config.trace is True
            assert draft_vocab_path == tmp_path / "vocab.json"
        def run(self):
            return 0
    monkeypatch.setattr(mtp_sweep, "MTPSweepRunner", FakeRunner)
    assert mtp_sweep.main(["--trace", "--draft-vocab", str(tmp_path / "vocab.json")]) == 0


def test_prefix_capacity_union_is_deterministic_and_does_not_use_evaluation_ids():
    tokenizer = {"added_tokens": [{"id": 7, "special": True}]}
    config = {"text_config": {"vocab_size": 8}}
    stock = build_draft_vocabulary(bundle(), tokenizer, config, "abc")
    assert "capacity" not in stock
    prefix = build_draft_vocabulary(bundle(), tokenizer, config, "abc", prefix_tokens=3)
    assert prefix["token_ids"] == [0, 1, 2, 3, 7]
    assert prefix["capacity"]["prefix_token_count"] == 3
    assert "not pristine validation" in prefix["development_tuning"]
    altered = bundle()
    altered["chains"][-1]["generated_token_ids"] = [4, 5, 6] * 20
    for chain in altered["chains"]:
        chain["prompt_token_ids"] = [4, 5, 6]
    other = build_draft_vocabulary(altered, tokenizer, config, "abc", prefix_tokens=3)
    assert other["token_ids"] == prefix["token_ids"]
    assert not {4, 5, 6} & set(other["token_ids"])


@pytest.mark.parametrize("prefix", [True, -1, 9, 3.5, None])
def test_prefix_capacity_validation(prefix):
    with pytest.raises(ValueError, match="prefix_tokens"):
        build_draft_vocabulary(bundle(), {"added_tokens": []}, {"vocab_size": 8}, "abc", prefix_tokens=prefix)
