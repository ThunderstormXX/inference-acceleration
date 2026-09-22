"""CPU-only tests for the scoped kernel policy; no MLX runtime is imported."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from inference_lab.optimizations.affine import ScopedAffineOptimization, load_affine_verifier


class Module(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None


class Linear(Module):
    def __call__(self, x=None):
        return ("original", id(self), x)


def fixture_models():
    head, projection, draft_projection = Linear(), Linear(), Linear()
    layer = Module(proj=projection)
    target = Module(language_model=Module(model=Module(layers=[layer]), lm_head=head))
    draft = Module(layers=[Module(proj=draft_projection)], lm_head=head)
    return target, draft, head, projection, draft_projection


def make_context(target, draft=None, *, policy="head", operation=None):
    verifier = SimpleNamespace(optimized_affine_linear=operation or (lambda module, x: ("fast", id(module), x)))
    return ScopedAffineOptimization(target, draft, policy=policy,
                                    verifier=verifier, quantized_linear_class=Linear)


def tensor(tokens=3):
    return SimpleNamespace(ndim=3, shape=(1, tokens, 4096))


def test_head_policy_is_scoped_to_shared_head_and_restores():
    target, draft, head, projection, draft_projection = fixture_models()
    original = Linear.__call__
    ctx = make_context(target, draft)
    with ctx:
        assert head(tensor())[0] == "fast"
        assert draft.lm_head(tensor(2))[0] == "fast"
        assert projection(tensor())[0] == "original"
        assert draft_projection(tensor())[0] == "original"
    assert Linear.__call__ is original
    assert ctx.metadata()["optimized_calls_by_tokens"] == {"3": 1, "2": 1}


def test_target_policy_finds_layers_behind_dflash_hooks():
    target, draft, head, projection, draft_projection = fixture_models()
    layers = target.language_model.model.layers
    layers[0] = SimpleNamespace(_layer=layers[0])
    with make_context(target, draft, policy="target") as ctx:
        assert projection(tensor())[0] == "fast"
        assert head(tensor())[0] == "fast"
        assert draft_projection(tensor())[0] == "original"
    assert ctx.metadata()["selected_module_count"] == 2


def test_all_policy_deduplicates_shared_head():
    target, draft, head, projection, draft_projection = fixture_models()
    with make_context(target, draft, policy="all") as ctx:
        assert all(module(tensor())[0] == "fast" for module in (head, projection, draft_projection))
    assert ctx.metadata()["selected_module_count"] == 3


def test_prefill_rank_and_keyword_calls_fall_back_without_kernel():
    target, _, head, _, _ = fixture_models()
    calls = []
    ctx = make_context(target, operation=lambda *args: calls.append(args))
    with ctx:
        assert head(tensor(128))[0] == "original"
        assert head(SimpleNamespace(ndim=2, shape=(1, 4096)))[0] == "original"
        assert head(x=tensor())[0] == "original"
    assert calls == []
    assert ctx.metadata()["fallback_calls_by_tokens"] == {"128": 1, "other": 2}
    assert ctx.metadata()["fallback_reasons"] == {"sequence_length": 1, "input_rank": 1, "call_signature": 1}


def test_unsupported_kernel_falls_back_and_counters_reset():
    target, _, head, _, _ = fixture_models()
    ctx = make_context(target, operation=lambda *args: None)
    with ctx:
        assert head(tensor(1))[0] == "original"
    assert ctx.metadata()["fallback_calls_by_tokens"] == {"1": 1}
    ctx.reset_counters()
    assert ctx.metadata()["fallback_calls_by_tokens"] == {}


def test_exception_restores_original_call():
    target, _, head, _, _ = fixture_models()
    original = Linear.__call__
    def fail(*args):
        raise RuntimeError("kernel failure")
    with pytest.raises(RuntimeError, match="kernel failure"):
        with make_context(target, operation=fail):
            head(tensor())
    assert Linear.__call__ is original


def test_nested_independent_contexts_restore_each_layer():
    target, _, head, projection, _ = fixture_models()
    original = Linear.__call__
    with make_context(target):
        outer = Linear.__call__
        with make_context(target, policy="target"):
            assert projection(tensor())[0] == "fast"
        assert Linear.__call__ is outer
        assert projection(tensor())[0] == "original"
    assert Linear.__call__ is original


def test_reenter_same_context_rejected_without_losing_outer_patch():
    target, _, head, _, _ = fixture_models()
    ctx = make_context(target)
    original = Linear.__call__
    with ctx:
        with pytest.raises(RuntimeError, match="entered twice"):
            with ctx:
                pass
        assert head(tensor())[0] == "fast"
    assert Linear.__call__ is original


@pytest.mark.parametrize("kwargs", [{"policy": "bad"}, {"max_tokens": 0}, {"max_tokens": True}, {"max_tokens": 9}])
def test_policy_validation_precedes_runtime_import(kwargs):
    with pytest.raises(ValueError):
        ScopedAffineOptimization({}, **kwargs)


def test_source_loader_uses_relative_import_without_site_packages(tmp_path):
    (tmp_path / "quantized_verifier.py").write_text("from .linear import VALUE\nRESULT = VALUE\n")
    (tmp_path / "linear.py").write_text("VALUE = 17\n")
    (tmp_path / "switch_layers.py").write_text("# unused test source\n")
    (tmp_path / "activations.py").write_text("# unused test source\n")
    paths = list(sys.path)
    module, metadata = load_affine_verifier(tmp_path)
    assert module.RESULT == 17
    assert sys.path == paths
    assert set(metadata["source_sha256"]) == {"quantized_verifier.py", "linear.py", "switch_layers.py", "activations.py"}
    assert all(len(value) == 64 for value in metadata["source_sha256"].values())
    (tmp_path / "linear.py").write_text("VALUE = 200\n")
    changed, changed_metadata = load_affine_verifier(tmp_path)
    assert changed.RESULT == 200
    assert changed is not module
    assert changed_metadata["combined_source_sha256"] != metadata["combined_source_sha256"]
