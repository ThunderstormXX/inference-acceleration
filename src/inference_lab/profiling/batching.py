"""Fixed-prefix target batching probe; no model loads and no draft generation.

This measures synchronized wall time, including Python submission and host token
transfer. It is a batching ceiling on a known continuation, not speculative
throughput: real drafting, rejected suffixes and rollback still have to be paid.
"""
from __future__ import annotations

from copy import copy
from hashlib import sha256
import json
from math import isfinite
from statistics import mean, median, stdev
from time import perf_counter
from typing import Any

from inference_lab.backends.apple.speculative.dflash_backend import _UPSTREAM_LOCK


CACHE_ATOL = 1e-3


def _tensor(value: Any) -> bool:
    return hasattr(value, "shape") and hasattr(value, "dtype")


def _leaves(value: Any, path: str = ""):
    if _tensor(value):
        yield path, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _leaves(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _leaves(item, f"{path}[{index}]")


def _clone_value(value: Any, mx: Any) -> Any:
    if _tensor(value):
        # New array wrapper: MLX assignment replaces its graph, not the source
        # wrapper. Copy every mutable container as well (ArraysCache.state is a
        # live list). Preserve KV allocation slack rather than using from_state.
        return mx.array(value)
    if isinstance(value, dict):
        return {key: _clone_value(item, mx) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_value(item, mx) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_value(item, mx) for item in value)
    return copy(value)


def _clone_cache(cache: list[Any], mx: Any) -> list[Any]:
    result = []
    for original in cache:
        cloned = copy(original)
        cloned.__dict__ = _clone_value(vars(original), mx)
        result.append(cloned)
    return result


def _cache_arrays(cache: list[Any]) -> list[Any]:
    return [tensor for layer in cache for _, tensor in _leaves(vars(layer))]


def _compare_caches(left: list[Any], right: list[Any], mx: Any) -> dict[str, Any]:
    if len(left) != len(right):
        raise RuntimeError("Compared caches have different layer counts")
    errors = []
    offsets_equal = True
    for index, (a, b) in enumerate(zip(left, right)):
        a_values, b_values = dict(_leaves(vars(a))), dict(_leaves(vars(b)))
        if a_values.keys() != b_values.keys():
            raise RuntimeError("Compared caches have different tensor fields")
        offsets_equal &= getattr(a, "offset", None) == getattr(b, "offset", None)
        for name, value in a_values.items():
            other = b_values[name]
            if value.shape != other.shape:
                raise RuntimeError(f"Cache shape mismatch at layer {index}{name}")
            error = mx.max(mx.abs(value.astype(mx.float32) - other.astype(mx.float32)))
            errors.append((f"{index}{name}", error))
    mx.eval([error for _, error in errors])
    values = [{"tensor": name, "max_abs_error": float(error.item())} for name, error in errors]
    if any(not isfinite(item["max_abs_error"]) for item in values):
        raise RuntimeError("Non-finite value in end-cache comparison")
    largest = max((item["max_abs_error"] for item in values), default=0.0)
    return {
        "max_abs_error": largest,
        "offsets_equal": bool(offsets_equal),
        "within_atol": bool(offsets_equal and all(item["max_abs_error"] <= CACHE_ATOL for item in values)),
        "tensor_errors": values,
    }


def _stats(values: list[float]) -> dict[str, float]:
    return {"mean": mean(values), "median": median(values), "stdev": stdev(values) if len(values) > 1 else 0.0,
            "min": min(values), "max": max(values)}


def _reset_hidden(model: Any) -> None:
    model._hidden_states[:] = [None] * len(model._hidden_states)


def _trial(backend: Any, snapshot: list[Any], token_ids: list[int], mode: str) -> tuple[float, list[int], list[Any]]:
    mx, upstream, model = backend._mx, backend._upstream, backend._model
    stream = upstream.generation_stream
    with mx.stream(stream):
        cache = _clone_cache(snapshot, mx)
        inputs = mx.array([token_ids], dtype=mx.uint32)
        mx.eval(inputs, _cache_arrays(cache))
    mx.synchronize(stream)
    _reset_hidden(model)
    # Production verifier captures GDN intermediates for potential rollback.
    # Installing/removing its Python patch is lifecycle overhead, not a forward.
    capture = upstream._GDNStateCapture() if mode == "block" else None
    try:
        with mx.stream(stream):
            started = perf_counter()
            output = []
            if mode == "block":
                logits = model(inputs, cache)
                hidden = mx.concatenate(model._hidden_states, axis=-1)
                predicted = mx.argmax(logits, axis=-1)
                mx.eval(predicted, hidden, _cache_arrays(cache))
                output.extend(int(item) for item in predicted[0].tolist())
            else:
                for position in range(len(token_ids)):
                    logits = model(inputs[:, position:position + 1], cache)
                    predicted = mx.argmax(logits, axis=-1)
                    # One completed ordinary decode step before the next input,
                    # despite teacher-forcing making the input known in advance.
                    mx.eval(predicted, _cache_arrays(cache))
                    output.extend(int(item) for item in predicted[0].tolist())
            mx.synchronize(stream)
            elapsed_ms = (perf_counter() - started) * 1000.0
        return elapsed_ms, output, cache
    finally:
        if capture is not None:
            capture.close()
        _reset_hidden(model)


def run_batching_probe(
    backend: Any,
    prompt_tokens: list[int],
    continuation_tokens: list[int],
    widths=(1, 2, 3, 5),
    repeats: int = 5,
    prefix_generated: int = 32,
) -> dict[str, Any]:
    """Compare K-wide verification to K sequential target steps at one prefix.

    ``continuation_tokens`` must be an existing baseline continuation. The first
    ``prefix_generated`` tokens join the prompt cache; following tokens are the
    identical teacher-forced inputs for both branches. End-state discrepancies
    are reported, not hidden behind a loose equality claim.
    """
    for name, tokens in (("prompt_tokens", prompt_tokens), ("continuation_tokens", continuation_tokens)):
        if not isinstance(tokens, list) or not tokens or any(type(token) is not int or token < 0 for token in tokens):
            raise ValueError(f"{name} must be a nonempty list of nonnegative integer token IDs")
    widths = tuple(widths)
    if not widths or any(type(width) is not int or width < 1 for width in widths) or len(set(widths)) != len(widths):
        raise ValueError("widths must contain distinct positive integers")
    if type(repeats) is not int or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    if type(prefix_generated) is not int or prefix_generated < 0:
        raise ValueError("prefix_generated must be a nonnegative integer")
    if len(continuation_tokens) < prefix_generated + max(widths):
        raise ValueError("Continuation is too short for the requested prefix and widths")
    if backend._model is None or backend._draft is None or backend._upstream is None:
        raise RuntimeError("Call DFlashBackend.load() before the batching probe")

    mx, upstream, model = backend._mx, backend._upstream, backend._model
    stream = upstream.generation_stream
    prefix = prompt_tokens + continuation_tokens[:prefix_generated]
    following = continuation_tokens[prefix_generated:prefix_generated + max(widths)]
    results = []
    previous_hidden = list(model._hidden_states) if hasattr(model, "_hidden_states") else None
    with _UPSTREAM_LOCK, upstream.wired_limit(model, [stream]):
        upstream._patch_model(model, backend._draft.config.target_layer_ids)
        try:
            with mx.stream(stream):
                snapshot = upstream.make_prompt_cache(model)
                for start in range(0, len(prefix), backend.prefill_step_size):
                    logits = model(mx.array([prefix[start:start + backend.prefill_step_size]], dtype=mx.uint32), snapshot)
                    # Prefix logits are unused; materialize only the states
                    # required by the subsequent decode and release its graph.
                    mx.eval(_cache_arrays(snapshot), model._hidden_states)
                    del logits
                reference = _clone_cache(snapshot, mx)
                mx.eval(_cache_arrays(reference))
            mx.synchronize(stream)
            _reset_hidden(model)
            cache_layout = [{"layer": index, "class": type(layer).__name__, "offset": getattr(layer, "offset", None),
                             "allocated_tensors": [{"field": name, "shape": list(value.shape), "dtype": str(value.dtype)}
                                                   for name, value in _leaves(vars(layer))]}
                            for index, layer in enumerate(snapshot)]
            for width in widths:
                inputs = following[:width]
                # Warm both exact shapes/paths before collecting paired samples.
                for mode in ("block", "sequential"):
                    _trial(backend, snapshot, inputs, mode)
                rows = []
                for repeat in range(repeats):
                    order = ("block", "sequential") if repeat % 2 == 0 else ("sequential", "block")
                    pair = {mode: _trial(backend, snapshot, inputs, mode) for mode in order}
                    with mx.stream(stream):
                        comparison = _compare_caches(pair["block"][2], pair["sequential"][2], mx)
                    mx.synchronize(stream)
                    block_ms, block_ids, _ = pair["block"]
                    seq_ms, seq_ids, _ = pair["sequential"]
                    rows.append({"repeat": repeat, "order": list(order), "block_ms": block_ms, "sequential_ms": seq_ms,
                                 "speedup": seq_ms / block_ms if block_ms > 0 else None,
                                 "block_argmax_ids": block_ids, "sequential_argmax_ids": seq_ids,
                                 "argmax_equal": block_ids == seq_ids, "end_cache": comparison})
                with mx.stream(stream):
                    unchanged = _compare_caches(snapshot, reference, mx)
                mx.synchronize(stream)
                if unchanged["max_abs_error"] != 0 or not unchanged["offsets_equal"]:
                    raise RuntimeError("A trial mutated the fixed-prefix snapshot")
                results.append({"width": width, "input_token_ids": inputs, "trials": rows,
                                "block_ms": _stats([row["block_ms"] for row in rows]),
                                "sequential_ms": _stats([row["sequential_ms"] for row in rows]),
                                "speedup": _stats([row["speedup"] for row in rows if row["speedup"] is not None]),
                                "all_argmax_equal": all(row["argmax_equal"] for row in rows),
                                "all_cache_within_atol": all(row["end_cache"]["within_atol"] for row in rows),
                                "max_cache_abs_error": max(row["end_cache"]["max_abs_error"] for row in rows)})
        finally:
            model._hidden_states[:] = previous_hidden if previous_hidden is not None else [None] * len(model._hidden_states)

    return {
        "schema_version": 1, "probe": "fixed_prefix_target_batching", "repeats": repeats,
        "prefix_generated": prefix_generated, "prompt_tokens": len(prompt_tokens), "prefix_tokens": len(prefix),
        "prefix_token_ids_sha256": sha256(json.dumps(prefix, separators=(",", ":")).encode()).hexdigest(),
        "cache_comparison_atol": CACHE_ATOL, "cache_layout": cache_layout,
        "methodology": {
            "time_kind": "synchronized wall-clock milliseconds, not GPU-only kernel time",
            "block": "K input positions; production GDN capture; target forward, hidden concatenation, argmax, cache evaluation and host token transfer",
            "sequential": "K completed one-token forwards; ordinary GDN; argmax, cache evaluation and host token transfer per step; hidden hooks remain but no concatenation",
            "excluded": ["model loading", "prefix prefill", "cache cloning", "input allocation", "GDN patch lifecycle", "warmup", "end-state comparison", "draft", "rollback"],
            "cache": "Each branch clones the same complete prefix cache, preserving allocated capacity and metadata; snapshots checked unchanged",
            "interpretation": "Teacher-forced batching ceiling on an actual continuation; real benefit must cover drafting plus rollback and use accepted+1 useful tokens, not all K positions",
            "warmup": "One block and one sequential trial per width, excluded from results",
            "order": "Block/sequential then sequential/block, alternating per repeat",
        },
        "widths": results,
    }
