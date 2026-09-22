"""CPU-only regression tests for reversible ZMLX module patching."""
import builtins
from dataclasses import dataclass, field
import sys

import pytest

from inference_lab.optimizations import zmlx


class Module(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None

    def update(self, *args, **kwargs):
        raise AssertionError("nn.Module.update is a parameter update API, not dict restoration")

    def clear(self):
        raise AssertionError("Restore the raw mapping using dict.clear")

    def named_modules(self):
        def walk(value, path):
            if isinstance(value, Module):
                yield path, value
                for key, child in value.items():
                    yield from walk(child, f"{path}.{key}" if path else str(key))
            elif isinstance(value, (list, tuple)):
                for index, child in enumerate(value):
                    yield from walk(child, f"{path}.{index}")
        return list(walk(self, ""))


@dataclass
class PatchResult:
    patched_count: int = 1
    pattern_counts: dict = field(default_factory=lambda: {"deltanet": 1})


def model_fixture():
    weight = object()
    layer = Module(weight=weight)
    layer.ordinary_attribute = "original"
    model = Module(layers=[layer])
    return model, layer, weight


def fake_patch(model, patterns):
    layer = model["layers"][0]
    layer.__class__ = type("PatchedModule", (Module,), {})
    layer["temporary"] = object()
    layer.ordinary_attribute = "modified"
    layer._zmlx_state = {"new": True}
    model._zmlx_patch_result = PatchResult()


def assert_restored(model, layer, weight):
    assert type(layer) is Module
    assert set(layer) == {"weight"}
    assert layer["weight"] is weight
    assert vars(layer) == {"ordinary_attribute": "original"}
    assert vars(model) == {}
    assert model["layers"][0] is layer


def test_restores_classes_mappings_attributes_without_module_update(monkeypatch):
    model, layer, weight = model_fixture()
    monkeypatch.setattr(zmlx, "load_zmlx", lambda: fake_patch)
    with zmlx.scoped_zmlx(model, ["deltanet"]) as metadata:
        assert type(layer) is not Module
        assert layer._zmlx_state == {"new": True}
        assert metadata["patched_count"] == 1
        assert metadata["commit"] == zmlx.ZMLX_COMMIT
    assert_restored(model, layer, weight)


def test_restores_when_generation_raises(monkeypatch):
    model, layer, weight = model_fixture()
    monkeypatch.setattr(zmlx, "load_zmlx", lambda: fake_patch)
    with pytest.raises(RuntimeError, match="generation failed"):
        with zmlx.scoped_zmlx(model, ["swiglu_mlp"]):
            raise RuntimeError("generation failed")
    assert_restored(model, layer, weight)


def test_restores_partial_patch_before_enter_failure(monkeypatch):
    model, layer, weight = model_fixture()
    def fail(model, patterns):
        fake_patch(model, patterns)
        raise ValueError("patch failed")
    monkeypatch.setattr(zmlx, "load_zmlx", lambda: fail)
    with pytest.raises(ValueError, match="patch failed"):
        with zmlx.scoped_zmlx(model, ["deltanet"]):
            pytest.fail("Failed patch must not yield")
    assert_restored(model, layer, weight)


def test_zero_patch_count_is_not_a_valid_experiment(monkeypatch):
    model, layer, weight = model_fixture()
    def zero(model, patterns):
        model._zmlx_patch_result = PatchResult(patched_count=0)
    monkeypatch.setattr(zmlx, "load_zmlx", lambda: zero)
    with pytest.raises(RuntimeError, match="No modules patched"):
        with zmlx.scoped_zmlx(model, ["deltanet"]):
            pytest.fail("Unpatched variant must not yield")
    assert_restored(model, layer, weight)


@pytest.mark.parametrize("patterns", [[], ["rms_norm"], ["deltanet", "unknown"]])
def test_unsupported_patterns_rejected_before_loading(monkeypatch, patterns):
    monkeypatch.setattr(zmlx, "load_zmlx", lambda: pytest.fail("Unneeded import"))
    with pytest.raises(ValueError, match="Only DeltaNet and SwiGLU"):
        with zmlx.scoped_zmlx(Module(), patterns):
            pass


@pytest.mark.parametrize("head,dirty", [("wrong", ""), (zmlx.ZMLX_COMMIT, " M src/file.py")])
def test_loader_rejects_unpinned_or_dirty_checkout(monkeypatch, head, dirty):
    monkeypatch.setattr(zmlx.subprocess, "check_output", lambda cmd, **kwargs: head if cmd[-1] == "HEAD" else dirty)
    before = list(sys.path)
    with pytest.raises(ValueError, match="pinned clean"):
        zmlx.load_zmlx()
    assert sys.path == before


def test_loader_restores_sys_path_on_import_failure(monkeypatch):
    monkeypatch.setattr(zmlx.subprocess, "check_output", lambda cmd, **kwargs: zmlx.ZMLX_COMMIT if cmd[-1] == "HEAD" else "")
    original_import = builtins.__import__
    def failing_import(name, *args, **kwargs):
        if name == "zmlx.patch":
            raise ImportError("unavailable test dependency")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", failing_import)
    before = list(sys.path)
    with pytest.raises(ImportError, match="unavailable test dependency"):
        zmlx.load_zmlx()
    assert sys.path == before


def test_unexpected_list_replacement_is_rejected_and_restored(monkeypatch):
    model, layer, weight = model_fixture()
    def replace(model, patterns):
        fake_patch(model, patterns)
        model["layers"][0] = Module(weight=object())
    monkeypatch.setattr(zmlx, "load_zmlx", lambda: replace)
    with pytest.raises(RuntimeError, match="replaced modules"):
        with zmlx.scoped_zmlx(model, ["deltanet"]):
            pytest.fail("Unexpected replacement must not yield")
    assert_restored(model, layer, weight)
