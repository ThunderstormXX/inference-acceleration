"""CPU-only contracts for short-block two-token tiled dispatch."""
import ast
import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest

from inference_lab.optimizations.tiled_affine import ScopedTiledAffine, _tiled_code


def optimized_affine_linear(linear, x):
    if getattr(linear, "unsupported", False):
        return None
    B, T, K = x.shape
    streamed = linear.bits == 4 and 6 <= T <= 8
    token_tiled = linear.bits == 4 and T >= 6 and not streamed
    return ("streamed" if streamed else "tiled" if token_tiled else "stock", T, id(linear))


def optimized_affine_argmax(linear, x, token_mask=None):
    B, T, K = x.shape
    streamed = linear.bits == 4 and 6 <= T <= 8
    token_tiled = linear.bits == 4 and T >= 6 and not streamed
    return ("streamed" if streamed else "tiled" if token_tiled else "stock", T, token_mask)


def optimized_affine_linears(linears, x):
    bits = getattr(linears[0], "bits", None) if linears else None
    if not 2 <= len(linears) <= 4 or not 1 < x.shape[1] <= 8:
        return None
    B, T, K = x.shape
    streamed = bits == 4 and T >= 6
    return ("fused streamed" if streamed else "fused", T)


def verifier():
    return SimpleNamespace(optimized_affine_linear=optimized_affine_linear,
                           optimized_affine_argmax=optimized_affine_argmax,
                           optimized_affine_linears=optimized_affine_linears)


def tensor(tokens):
    return SimpleNamespace(shape=(1, tokens, 4096))


@pytest.mark.parametrize("threshold", [2, 3, 4, 5])
def test_tiled_threshold_uses_existing_aliases_preserves_other_ranges(threshold):
    module = verifier()
    aliases = list(vars(module).values())
    codes = [function.__code__ for function in aliases]
    q4, other = SimpleNamespace(bits=4), SimpleNamespace(bits=4)
    with ScopedTiledAffine(threshold, module) as context:
        for tokens in range(1, 10):
            expected = ("streamed" if 6 <= tokens <= 8 else
                        "tiled" if tokens >= threshold else "stock")
            assert aliases[0](q4, tensor(tokens))[0] == expected
            assert aliases[1](q4, tensor(tokens), "mask") == (expected, tokens, "mask")
            # Formats other than 4-bit retain their old dispatch.
            assert aliases[0](SimpleNamespace(bits=8), tensor(tokens))[0] == "stock"
        for tokens in range(2, 6):
            result = aliases[2]([q4, other], tensor(tokens))
            if tokens >= threshold:
                assert result == (("tiled", tokens, id(q4)), ("tiled", tokens, id(other)))
            else:
                assert result == ("fused", tokens)
        assert aliases[2]([q4, other], tensor(6)) == ("fused streamed", 6)
        assert aliases[2]([q4, other], tensor(8)) == ("fused streamed", 8)
        assert aliases[2]([q4], tensor(4)) is None
        assert aliases[2]([q4, other], tensor(1)) is None
        assert aliases[2]([q4, other], tensor(9)) is None
        assert aliases[2]([SimpleNamespace(bits=8), SimpleNamespace(bits=8)], tensor(4)) == ("fused", 4)
        metadata = context.metadata()
        assert metadata["tiled_from"] == threshold
        assert len(metadata["patched_functions"]) == 3
        assert metadata["kernel_implementation_changed"] is False
    assert [function.__code__ for function in aliases] == codes
    assert aliases[0](q4, tensor(4))[0] == "stock"


def test_fused_projection_bypass_preserves_none_fallback():
    q4, unsupported = SimpleNamespace(bits=4), SimpleNamespace(bits=4, unsupported=True)
    with ScopedTiledAffine(4, verifier()):
        assert optimized_affine_linears([q4, unsupported], tensor(4)) is None


def test_exception_restores_original_dispatch():
    module = verifier()
    codes = [function.__code__ for function in vars(module).values()]
    with pytest.raises(RuntimeError, match="benchmark failure"):
        with ScopedTiledAffine(4, module):
            raise RuntimeError("benchmark failure")
    assert [function.__code__ for function in vars(module).values()] == codes


@pytest.mark.parametrize("threshold", [True, None, 1, 6, 3.5])
def test_threshold_validation_does_not_import_mlx(threshold):
    with pytest.raises(ValueError):
        ScopedTiledAffine(threshold)


@pytest.mark.parametrize("target_name, edit", [
    ("optimized_affine_linear", lambda source: source.replace("T >= 6 and not streamed", "T >= 5 and not streamed")),
    ("optimized_affine_argmax", lambda source: source.replace("6 <= T <= 8", "5 <= T <= 8")),
    ("optimized_affine_linears", lambda source: source.replace("B, T, K = x.shape", "B, T, K = different_shape")),
])
def test_changed_guards_fail_closed_without_partial_patch(monkeypatch, target_name, edit):
    module = verifier()
    codes = [function.__code__ for function in vars(module).values()]
    original = inspect.getsource
    monkeypatch.setattr(inspect, "getsource", lambda function:
                        edit(original(function)) if function.__name__ == target_name else original(function))
    with pytest.raises(ValueError):
        with ScopedTiledAffine(4, module):
            pass
    assert [function.__code__ for function in vars(module).values()] == codes


def test_installed_function_sources_compile_without_importing_mlx(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    path = root / ".venv/lib/python3.13/site-packages/mlx_vlm/models/quantized_verifier.py"
    if not path.exists():
        pytest.skip("MLX-VLM source not installed")
    source = path.read_text()
    nodes = {node.name: ast.get_source_segment(source, node) for node in ast.parse(source).body
             if isinstance(node, ast.FunctionDef)}
    monkeypatch.setattr(inspect, "getsource", lambda function: nodes[function.__name__])
    monkeypatch.setitem(globals(), "mx", SimpleNamespace(array=object))
    for function in vars(verifier()).values():
        code, metadata = _tiled_code(function, 4)
        assert not code.co_freevars
        assert len(metadata["patched_source_sha256"]) == 64


def test_cli_records_tiled_dispatch_and_excludes_streamed_flag(monkeypatch, tmp_path):
    from inference_lab.optimizations import mtp_sweep, tiled_affine
    events = []
    class FakeOptimization:
        def __init__(self, threshold):
            assert threshold == 4
        def __enter__(self):
            events.append("enter")
            return self
        def metadata(self):
            return {"tiled_from": 4}
        def __exit__(self, *args):
            events.append("exit")
    class FakeRunner:
        def __init__(self, *args, optimization_metadata=None, draft_vocab_path=None):
            assert optimization_metadata == {"tiled_from": 4}
        def run(self):
            events.append("run")
            return 0
    monkeypatch.setattr(tiled_affine, "ScopedTiledAffine", FakeOptimization)
    monkeypatch.setattr(mtp_sweep, "MTPSweepRunner", FakeRunner)
    assert mtp_sweep.main(["--tiled-from", "4", "--output", str(tmp_path)]) == 0
    assert events == ["enter", "run", "exit"]
    with pytest.raises(SystemExit) as error:
        mtp_sweep.main(["--tiled-from", "4", "--streamed-from", "4"])
    assert error.value.code == 2
