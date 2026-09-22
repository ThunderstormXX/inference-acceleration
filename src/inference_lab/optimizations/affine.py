"""Scoped affine projection kernels borrowed from the installed MLX-VLM source.

This module does not import MLX until an optimization context is constructed.
The synthetic package avoids importing the MLX-VLM application and its optional
vision/audio dependencies into the independently pinned DFlash environment.
Kernels retain their upstream implementation; every imported source is hashed.
Numerical and token parity still require empirical checks against the baseline.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from hashlib import sha256
import importlib
import importlib.machinery
from pathlib import Path
import sys
from threading import RLock
from types import ModuleType
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODELS_PATH = ROOT / ".venv/lib/python3.13/site-packages/mlx_vlm/models"
_SOURCE_FILES = ("quantized_verifier.py", "switch_layers.py", "activations.py", "linear.py")
_PATCH_LOCK = RLock()
_IMPORT_LOCK = RLock()


def load_affine_verifier(models_path: str | Path | None = None):
    """Load the existing source without adding another environment to sys.path."""
    directory = Path(models_path or DEFAULT_MODELS_PATH).expanduser().resolve()
    sources = {name: sha256((directory / name).read_bytes()).hexdigest()
               for name in _SOURCE_FILES}
    digest = sha256("".join(name + sources[name] for name in _SOURCE_FILES).encode()).hexdigest()
    package_name = "_inference_lab_affine_" + digest
    with _IMPORT_LOCK:
        if package_name not in sys.modules:
            package = ModuleType(package_name)
            package.__path__ = [str(directory)]
            package.__package__ = package_name
            package.__spec__ = importlib.machinery.ModuleSpec(package_name, loader=None, is_package=True)
            package.__spec__.submodule_search_locations = [str(directory)]
            sys.modules[package_name] = package
        try:
            verifier = importlib.import_module(package_name + ".quantized_verifier")
        except BaseException:
            for name in tuple(sys.modules):
                if name == package_name or name.startswith(package_name + "."):
                    sys.modules.pop(name, None)
            raise
    return verifier, {"models_path": str(directory), "source_sha256": sources,
                      "combined_source_sha256": digest,
                      "import_method": "isolated synthetic package; no site-packages injection"}


def _quantized_modules(model: Any, quantized_class: type):
    """Traverse Module mappings, including DFlash's non-Module layer wrappers."""
    result = {}
    visited = set()

    def visit(value, path):
        identity = id(value)
        if identity in visited:
            return
        visited.add(identity)
        if isinstance(value, quantized_class):
            result[identity] = (path, value)
        if isinstance(value, Mapping):
            for key, child in value.items():
                visit(child, f"{path}.{key}" if path else str(key))
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                visit(child, f"{path}.{index}" if path else str(index))
        else:
            # _LayerHook is intentionally not an nn.Module. Do not traverse
            # arbitrary attributes or tensor internals; unwrap this one field.
            wrapped = getattr(value, "__dict__", {}).get("_layer")
            if wrapped is not None:
                visit(wrapped, path)

    visit(model, "")
    return result


class ScopedAffineOptimization:
    """Temporarily substitute eligible projection calls on selected modules.

    ``head`` selects the target LM head, which DFlash shares with its drafter.
    ``target`` selects all target projections. ``all`` additionally selects all
    draft projections. Calls longer than ``max_tokens`` keep the original
    prefill kernels. This adapter changes neither weights nor cache policy.
    The optional verifier/class arguments exist for CPU-only contract tests.
    """

    def __init__(self, target, draft=None, *, policy="head", max_tokens=8,
                 models_path=None, verifier=None, quantized_linear_class=None):
        if policy not in ("head", "target", "all"):
            raise ValueError("policy must be head, target, or all")
        if type(max_tokens) is not int or not 1 <= max_tokens <= 8:
            raise ValueError("max_tokens must be an integer between 1 and 8")
        if (verifier is None) != (quantized_linear_class is None):
            raise ValueError("test verifier and quantized class must be supplied together")
        if verifier is None:
            self.verifier, self.source_metadata = load_affine_verifier(models_path)
            import mlx.nn as nn
            self.quantized_class = nn.QuantizedLinear
        else:
            self.verifier = verifier
            self.quantized_class = quantized_linear_class
            self.source_metadata = {"import_method": "injected verifier"}
        self.policy = policy
        self.max_tokens = max_tokens
        target_modules = _quantized_modules(target, self.quantized_class)
        if policy == "head":
            language_model = getattr(target, "language_model", target)
            head = getattr(language_model, "lm_head", None)
            if head is None:
                head = getattr(target, "lm_head", None)
            if not isinstance(head, self.quantized_class):
                raise ValueError("head policy requires an explicit quantized target lm_head")
            self.modules = {id(head): ("target.lm_head", head)}
        else:
            self.modules = {key: ("target." + path, module)
                            for key, (path, module) in target_modules.items()}
            if policy == "all" and draft is not None:
                for key, (path, module) in _quantized_modules(draft, self.quantized_class).items():
                    self.modules.setdefault(key, ("draft." + path, module))
        if not self.modules:
            raise ValueError("no quantized projections found for requested policy")
        self.hits = Counter()
        self.fallbacks = Counter()
        self.fallback_reasons = Counter()
        self._active = False
        self._original = None

    def _call(self, original, module, args, kwargs):
        if id(module) not in self.modules:
            return original(module, *args, **kwargs)
        x = args[0] if len(args) == 1 and not kwargs else None
        tokens = int(x.shape[1]) if getattr(x, "ndim", None) == 3 else None
        token_key = str(tokens) if tokens is not None else "other"
        reason = None
        if x is None:
            reason = "call_signature"
        elif tokens is None:
            reason = "input_rank"
        elif not 1 <= tokens <= self.max_tokens:
            reason = "sequence_length"
        else:
            output = self.verifier.optimized_affine_linear(module, x)
            if output is not None:
                self.hits[token_key] += 1
                return output
            reason = "unsupported_runtime_or_format"
        self.fallbacks[token_key] += 1
        self.fallback_reasons[reason] += 1
        return original(module, *args, **kwargs)

    def __enter__(self):
        _PATCH_LOCK.acquire()
        try:
            if self._active:
                raise RuntimeError("the same optimization context cannot be entered twice")
            original = self.quantized_class.__call__
            self._original = original

            def replacement(module, *args, **kwargs):
                return self._call(original, module, args, kwargs)

            self.quantized_class.__call__ = replacement
            self._active = True
            return self
        except BaseException:
            _PATCH_LOCK.release()
            raise

    def __exit__(self, exc_type, exc, traceback):
        try:
            self.quantized_class.__call__ = self._original
            self._active = False
            self._original = None
        finally:
            _PATCH_LOCK.release()
        return False

    def reset_counters(self):
        self.hits.clear()
        self.fallbacks.clear()
        self.fallback_reasons.clear()

    def metadata(self):
        return {"policy": self.policy, "max_tokens": self.max_tokens,
                "selected_module_count": len(self.modules),
                "selected_modules": sorted(path for path, _ in self.modules.values()),
                "optimized_calls_by_tokens": dict(self.hits),
                "fallback_calls_by_tokens": dict(self.fallbacks),
                "fallback_reasons": dict(self.fallback_reasons),
                "head_shared_between_target_and_draft": True if self.policy == "head" else None,
                "numerical_parity": "not assumed; must be checked against unchanged baseline",
                **self.source_metadata}
