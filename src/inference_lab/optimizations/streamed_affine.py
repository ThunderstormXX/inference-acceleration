"""Scoped experiment: use existing streamed affine kernels for shorter blocks.

Only three Python dispatch thresholds change; Metal kernel code, weights,
mathematical operations, masks and cache handling are untouched. Function code
objects are swapped so aliases already imported by the MTP verifier participate.
"""
from __future__ import annotations

import ast
from hashlib import sha256
import importlib
import inspect
from threading import RLock
import textwrap
from types import FunctionType


_GUARDS = {
    "optimized_affine_linear": "linear.bits == 4 and 6 <= T <= 8",
    "optimized_affine_argmax": "linear.bits == 4 and 6 <= T <= 8",
    "optimized_affine_linears": "bits == 4 and T >= 6",
}
_DISPATCH_LOCK = RLock()
_ACTIVE_FUNCTIONS = set()


def _threshold_code(function, threshold):
    name = function.__name__
    expected = _GUARDS[name]
    source = textwrap.dedent(inspect.getsource(function))
    tree = ast.parse(source)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError(f"Expected one function definition for {name}")
    definition = tree.body[0]
    if definition.name != name or definition.decorator_list:
        raise ValueError(f"Unexpected decorated or renamed function {name}")
    if function.__code__.co_freevars:
        raise ValueError("Dispatch patch does not support closures")
    assignments = [node for node in ast.walk(definition) if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "streamed"
                           for target in node.targets)]
    if (len(assignments) != 1 or len(assignments[0].targets) != 1
            or ast.unparse(assignments[0].value) != expected):
        raise ValueError(f"Pinned streamed dispatch guard changed in {name}")
    rewritten = (f"bits == 4 and T >= {threshold}" if name.endswith("linears")
                 else f"linear.bits == 4 and {threshold} <= T <= 8")
    assignments[0].value = ast.parse(rewritten, mode="eval").body
    ast.fix_missing_locations(tree)
    namespace = dict(function.__globals__)
    # Compiled functions retain the original globals through __code__ swapping.
    # The temporary function is used only to obtain its code object.
    exec(compile(tree, inspect.getsourcefile(function) or "<streamed-affine>", "exec"), namespace)
    patched = namespace[name]
    if patched.__code__.co_freevars != function.__code__.co_freevars:
        raise ValueError("Dispatch patch unexpectedly changed closure requirements")
    return patched.__code__, {"original_source_sha256": sha256(source.encode()).hexdigest(),
                             "patched_source_sha256": sha256(ast.unparse(tree).encode()).hexdigest(),
                             "original_guard": expected, "patched_guard": rewritten}


class ScopedStreamedAffine:
    """Temporarily lower the 4-bit streamed-kernel threshold from six tokens."""
    def __init__(self, streamed_from=4, verifier_module=None):
        if type(streamed_from) is not int or not 2 <= streamed_from <= 6:
            raise ValueError("streamed_from must be an integer between 2 and 6")
        self.streamed_from = streamed_from
        self.verifier = verifier_module
        self.sources = {}
        self._originals = None

    def _prepare_function(self, function):
        return _threshold_code(function, self.streamed_from)

    def __enter__(self):
        if self._originals is not None:
            raise RuntimeError("The same streamed-affine context cannot be entered twice")
        _DISPATCH_LOCK.acquire()
        originals = []
        try:
            if self.verifier is None:
                self.verifier = importlib.import_module("mlx_vlm.models.quantized_verifier")
            functions = [getattr(self.verifier, name) for name in _GUARDS]
            if any(not isinstance(function, FunctionType) for function in functions):
                raise ValueError("Expected ordinary Python optimized affine functions")
            if any(id(function) in _ACTIVE_FUNCTIONS for function in functions):
                raise RuntimeError("A streamed-affine experiment already patches these functions")
            prepared = []
            for name, function in zip(_GUARDS, functions):
                if function.__name__ != name:
                    raise ValueError(f"Unexpected function alias for {name}")
                code, metadata = self._prepare_function(function)
                prepared.append((function, code))
                self.sources[name] = metadata
            # Prepare all three changes first: unexpected source means no patch.
            for function, code in prepared:
                originals.append((function, function.__code__))
                function.__code__ = code
                _ACTIVE_FUNCTIONS.add(id(function))
            self._originals = originals
            return self
        except BaseException:
            for function, code in reversed(originals):
                function.__code__ = code
                _ACTIVE_FUNCTIONS.discard(id(function))
            _DISPATCH_LOCK.release()
            raise

    def __exit__(self, exc_type, exc, traceback):
        try:
            for function, code in reversed(self._originals):
                function.__code__ = code
                _ACTIVE_FUNCTIONS.discard(id(function))
            self._originals = None
        finally:
            _DISPATCH_LOCK.release()
        return False

    def metadata(self):
        return {"optimization": "earlier streamed affine dispatch",
                "stock_streamed_from": 6, "streamed_from": self.streamed_from,
                "patched_functions": self.sources,
                "patch_method": "scoped function __code__; existing imported aliases preserved",
                "kernel_implementation_changed": False,
                "token_parity": "must be measured against stock AR",
                "hypothesis": "reduce per-thread live storage for T=3..5 on M2; register spilling is not directly measured"}
