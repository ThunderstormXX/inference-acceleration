"""CPU-only checks for alias-preserving dispatch changes and restoration."""
import ast
import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest

from inference_lab.optimizations.streamed_affine import ScopedStreamedAffine, _threshold_code


GLOBAL_SENTINEL = object()


def optimized_affine_linear(linear, x):
    T = x
    streamed = linear.bits == 4 and 6 <= T <= 8
    token_tiled = linear.bits == 4 and T >= 6 and not streamed
    return streamed, token_tiled, GLOBAL_SENTINEL


def optimized_affine_argmax(linear, x, token_mask=None):
    T = x
    streamed = linear.bits == 4 and 6 <= T <= 8
    token_tiled = linear.bits == 4 and T >= 6 and not streamed
    return streamed, token_tiled, token_mask


def optimized_affine_linears(linears, x):
    bits, T = linears[0].bits, x
    streamed = bits == 4 and T >= 6
    return streamed


def verifier():
    return SimpleNamespace(optimized_affine_linear=optimized_affine_linear,
                           optimized_affine_argmax=optimized_affine_argmax,
                           optimized_affine_linears=optimized_affine_linears)


@pytest.mark.parametrize("threshold", [2, 3, 4, 5, 6])
def test_threshold_changes_all_imported_aliases_and_restores(threshold):
    module = verifier()
    aliases = [module.optimized_affine_linear, module.optimized_affine_argmax, module.optimized_affine_linears]
    codes = [function.__code__ for function in aliases]
    q4 = SimpleNamespace(bits=4)
    q8 = SimpleNamespace(bits=8)
    with ScopedStreamedAffine(threshold, module) as context:
        assert module.optimized_affine_linear is aliases[0]
        assert aliases[0].__globals__ is globals()
        for tokens in range(1, 10):
            expected = threshold <= tokens <= 8
            assert aliases[0](q4, tokens)[0] == expected
            assert aliases[1](q4, tokens)[0] == expected
            assert aliases[2]([q4], tokens) == (tokens >= threshold)
            assert aliases[0](q8, tokens)[0] is False
            assert aliases[1](q8, tokens)[0] is False
            assert aliases[2]([q8], tokens) is False
        assert aliases[0](q4, 4)[2] is GLOBAL_SENTINEL
        assert aliases[1](q4, 4, "mask")[2] == "mask"
        # Existing token-tiled eligibility must stay unchanged beyond T=8.
        assert aliases[0](q4, 9)[:2] == (False, True)
        metadata = context.metadata()
        assert metadata["streamed_from"] == threshold
        assert len(metadata["patched_functions"]) == 3
        for source in metadata["patched_functions"].values():
            assert len(source["original_source_sha256"]) == 64
            assert len(source["patched_source_sha256"]) == 64
    assert [function.__code__ for function in aliases] == codes
    assert aliases[0](q4, 4)[0] is False


def test_exception_restores_every_code_object():
    module = verifier()
    before = {name: function.__code__ for name, function in vars(module).items()}
    with pytest.raises(RuntimeError, match="benchmark failed"):
        with ScopedStreamedAffine(3, module):
            raise RuntimeError("benchmark failed")
    assert {name: function.__code__ for name, function in vars(module).items()} == before


def test_overlapping_contexts_fail_without_displacing_outer_patch():
    module = verifier()
    with ScopedStreamedAffine(4, module):
        with pytest.raises(RuntimeError, match="already patches"):
            with ScopedStreamedAffine(3, module):
                pass
        assert module.optimized_affine_linear(SimpleNamespace(bits=4), 3)[0] is False
        assert module.optimized_affine_linear(SimpleNamespace(bits=4), 4)[0] is True
    # A later independent context remains valid after the rejected nested call.
    with ScopedStreamedAffine(3, module):
        assert module.optimized_affine_linear(SimpleNamespace(bits=4), 3)[0] is True


def test_same_context_reentry_rejected():
    context = ScopedStreamedAffine(4, verifier())
    with context:
        with pytest.raises(RuntimeError, match="entered twice"):
            with context:
                pass


@pytest.mark.parametrize("threshold", [True, 1, 7, 3.5, None])
def test_invalid_threshold_rejected_without_runtime_import(threshold):
    with pytest.raises(ValueError):
        ScopedStreamedAffine(threshold)


@pytest.mark.parametrize("edit", [
    lambda source: source.replace("6 <= T <= 8", "5 <= T <= 8"),
    lambda source: source.replace("streamed =", "different_name ="),
    lambda source: source.replace("    return streamed,", "    streamed = False\n    return streamed,"),
])
def test_unexpected_source_fails_closed_before_any_function_is_patched(monkeypatch, edit):
    module = verifier()
    original_getsource = inspect.getsource
    codes = {name: function.__code__ for name, function in vars(module).items()}
    def changed_source(function):
        source = original_getsource(function)
        return edit(source) if function.__name__ == "optimized_affine_argmax" else source
    monkeypatch.setattr(inspect, "getsource", changed_source)
    with pytest.raises(ValueError, match="guard changed"):
        with ScopedStreamedAffine(4, module):
            pass
    assert {name: function.__code__ for name, function in vars(module).items()} == codes


def test_actual_installed_function_sources_compile_without_mlx_import(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    path = root / ".venv/lib/python3.13/site-packages/mlx_vlm/models/quantized_verifier.py"
    if not path.exists():
        pytest.skip("MLX-VLM source not installed")
    source = path.read_text()
    nodes = {node.name: ast.get_source_segment(source, node) for node in ast.parse(source).body
             if isinstance(node, ast.FunctionDef)}
    monkeypatch.setattr(inspect, "getsource", lambda function: nodes[function.__name__])
    # Only annotations evaluate while compiling a definition; no kernels run.
    monkeypatch.setitem(globals(), "mx", SimpleNamespace(array=object))
    for function in vars(verifier()).values():
        code, metadata = _threshold_code(function, 4)
        assert not code.co_freevars
        assert metadata["patched_guard"].count("4") >= 1


def test_mtp_cli_enters_scope_and_records_experiment_metadata(monkeypatch, tmp_path):
    from inference_lab.optimizations import mtp_sweep
    from inference_lab.optimizations import streamed_affine
    events = []
    class FakeOptimization:
        def __init__(self, threshold):
            assert threshold == 3
        def __enter__(self):
            events.append("enter")
            return self
        def metadata(self):
            return {"streamed_from": 3}
        def __exit__(self, *args):
            events.append("exit")
    class FakeRunner:
        def __init__(self, *args, optimization_metadata=None, draft_vocab_path=None):
            assert optimization_metadata == {"streamed_from": 3}
            assert events == ["enter"]
        def run(self):
            events.append("run")
            return 0
    monkeypatch.setattr(streamed_affine, "ScopedStreamedAffine", FakeOptimization)
    monkeypatch.setattr(mtp_sweep, "MTPSweepRunner", FakeRunner)
    assert mtp_sweep.main(["--streamed-from", "3", "--output", str(tmp_path)]) == 0
    assert events == ["enter", "run", "exit"]
