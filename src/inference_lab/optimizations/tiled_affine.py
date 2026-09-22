"""Use upstream two-token tiles for short 4-bit verification blocks.

The existing tiled Metal kernels handle odd final tiles via token_count. This
experiment changes dispatch only for threshold <= T < 6. It retains the stock
streamed path at T=6..8 and the existing tiled path beyond that range.
"""
from __future__ import annotations

import ast
from hashlib import sha256
import inspect
import textwrap

from .streamed_affine import ScopedStreamedAffine


_TILE_GUARD = "linear.bits == 4 and T >= 6 and (not streamed)"
_STREAMED_GUARD = "linear.bits == 4 and 6 <= T <= 8"


def _tiled_code(function, threshold):
    source = textwrap.dedent(inspect.getsource(function))
    tree = ast.parse(source)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError("Expected one optimized affine function definition")
    definition = tree.body[0]
    if definition.name != function.__name__ or definition.decorator_list or function.__code__.co_freevars:
        raise ValueError("Unexpected optimized affine function wrapper or closure")
    assignments = [node for node in ast.walk(definition) if isinstance(node, ast.Assign)]

    def one_write(name, expected):
        matches = [node for node in assignments
                   if any(isinstance(target, ast.Name) and target.id == name for target in node.targets)]
        if (len(matches) != 1 or len(matches[0].targets) != 1
                or ast.unparse(matches[0].value) != expected):
            raise ValueError(f"Pinned {name} guard changed in {function.__name__}")
        return matches[0]

    if function.__name__ in ("optimized_affine_linear", "optimized_affine_argmax"):
        one_write("streamed", _STREAMED_GUARD)
        tile = one_write("token_tiled", _TILE_GUARD)
        rewritten = f"linear.bits == 4 and T >= {threshold} and not streamed"
        tile.value = ast.parse(rewritten, mode="eval").body
        changes = {"original_guard": _TILE_GUARD, "patched_guard": rewritten,
                   "streamed_guard_unchanged": _STREAMED_GUARD}
    elif function.__name__ == "optimized_affine_linears":
        one_write("streamed", "bits == 4 and T >= 6")
        locations = [index for index, node in enumerate(definition.body)
                     if isinstance(node, ast.Assign) and ast.unparse(node) == "B, T, K = x.shape"]
        if len(locations) != 1 or locations[0] < 1:
            raise ValueError("Pinned fused projection shape boundary changed")
        position = locations[0]
        guard = definition.body[position - 1]
        if not isinstance(guard, ast.If) or ast.unparse(guard.body[-1]) != "return None":
            raise ValueError("Expected original fused projection eligibility guard before shape boundary")
        # Preserve upstream's eligibility/fallback behavior. Only after all
        # linears pass that guard do we replace the fused multi-output kernel.
        dispatch = ast.parse(f"""
if bits == 4 and {threshold} <= T < 6:
    tiled_outputs = tuple(optimized_affine_linear(linear, x) for linear in linears)
    if any(output is None for output in tiled_outputs):
        return None
    return tiled_outputs
""").body[0]
        definition.body.insert(position + 1, dispatch)
        changes = {"original_guard": "bits == 4 and T >= 6",
                   "patched_guard": "unchanged; insert individual tiled projections before fused dispatch",
                   "individual_projection_guard": f"bits == 4 and {threshold} <= T < 6",
                   "none_fallback_preserved": True}
    else:
        raise ValueError("Unexpected optimized affine function")
    ast.fix_missing_locations(tree)
    namespace = dict(function.__globals__)
    exec(compile(tree, inspect.getsourcefile(function) or "<tiled-affine>", "exec"), namespace)
    patched = namespace[function.__name__]
    if patched.__code__.co_freevars != function.__code__.co_freevars:
        raise ValueError("Tiled dispatch unexpectedly changed closure requirements")
    return patched.__code__, {"original_source_sha256": sha256(source.encode()).hexdigest(),
                             "patched_source_sha256": sha256(ast.unparse(tree).encode()).hexdigest(),
                             **changes}


class ScopedTiledAffine(ScopedStreamedAffine):
    """Reversible dispatch experiment sharing the alias-safe patch machinery."""
    def __init__(self, tiled_from=4, verifier_module=None):
        if type(tiled_from) is not int or not 2 <= tiled_from <= 5:
            raise ValueError("tiled_from must be an integer between 2 and 5")
        super().__init__(6, verifier_module)
        self.tiled_from = tiled_from

    def _prepare_function(self, function):
        return _tiled_code(function, self.tiled_from)

    def metadata(self):
        return {"optimization": "earlier two-token tiled affine dispatch",
                "tiled_from": self.tiled_from, "short_block_upper_bound_exclusive": 6,
                "stock_streamed_range_preserved": [6, 8], "token_tile": 2,
                "fused_projections": "eligible short blocks use individual tiled projections",
                "patched_functions": self.sources,
                "patch_method": "scoped function __code__; existing imported aliases preserved",
                "kernel_implementation_changed": False,
                "token_parity": "must be measured against stock AR",
                "hypothesis": "reduce live per-thread input storage with two-token tiles; odd tails handled by existing token_count"}
