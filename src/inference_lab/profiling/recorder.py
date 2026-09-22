"""Hierarchical timing for lazy array runtimes, without importing the runtime.

Durations are *synchronized wall times*, not individual GPU kernel durations.
Every span includes its own entry/exit synchronization. The synchronization and
materialization fields are subsets of its exclusive time, not extra additive
costs. Synchronization can wait for device work and is not pure profiler overhead.
Nested fences change scheduling, so compare a separately measured uninstrumented
run before treating a profile as a production latency estimate.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from math import prod
from time import perf_counter
from typing import Any, TypeVar

T = TypeVar("T")


def _is_tensor(value: Any) -> bool:
    return hasattr(value, "shape") and hasattr(value, "dtype")


def _tensors(value: Any, seen: set[int] | None = None) -> Iterator[Any]:
    """Collect array leaves without reading data or evaluating lazy expressions."""
    if seen is None:
        seen = set()
    if id(value) in seen:
        return
    seen.add(id(value))
    if _is_tensor(value):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _tensors(item, seen)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _tensors(item, seen)


def tensor_metadata(value: Any) -> Any:
    """Describe array shapes/dtypes/storage sizes, never array values.

    Lists, tuples and mappings preserve their structure. Other objects are
    represented by their type name. Cache objects should expose their state to
    the caller explicitly; inspecting arbitrary attributes could evaluate data.
    """
    def describe(item: Any, parents: set[int]) -> Any:
        if _is_tensor(item):
            shape = [int(dimension) for dimension in item.shape]
            nbytes = getattr(item, "nbytes", None)
            if nbytes is None:
                itemsize = getattr(item, "itemsize", None)
                if itemsize is not None:
                    nbytes = prod(shape) * int(itemsize)
            return {
                "shape": shape,
                "dtype": str(item.dtype),
                "nbytes": None if nbytes is None else int(nbytes),
            }
        if id(item) in parents:
            return {"type": type(item).__name__, "recursive_reference": True}
        if isinstance(item, Mapping):
            return {str(key): describe(child, parents | {id(item)})
                    for key, child in item.items()}
        if isinstance(item, (tuple, list)):
            return [describe(child, parents | {id(item)}) for child in item]
        return {"type": type(item).__name__}

    return describe(value, set())


@dataclass
class ProfileSpan:
    id: int
    parent_id: int | None
    name: str
    start_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)
    end_ms: float | None = None
    inclusive_ms: float | None = None
    exclusive_ms: float | None = None
    entry_sync_ms: float = 0.0
    exit_sync_ms: float = 0.0
    materialize_ms: float = 0.0
    status: str = "running"
    error: dict[str, str] | None = None
    cleanup_error: dict[str, str] | None = None
    _children_ms: float = field(default=0.0, repr=False)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result.pop("_children_ms")
        return result


class ProfileRecorder:
    """A bounded, synchronous tree recorder with injected runtime operations.

    Example::

        recorder = ProfileRecorder(synchronize=mx.synchronize, evaluate=mx.eval)
        recorder.materialize("inputs", tokens)
        with recorder.span("decode_round", {"round": 0}):
            logits = recorder.measured_call(
                "target", model, tokens,
                metadata={"input": tensor_metadata(tokens)},
                materialize=lambda: [cache.state for cache in caches],
            )

    ``measured_call`` evaluates returned array leaves and optional side-effect
    state. Its input arrays are deliberately not evaluated automatically: call
    ``materialize`` before the operation to prevent attributing upstream lazy
    work to it. A plain ``span`` fences submitted work but cannot discover lazy
    arrays; evaluate them inside the span or use ``measured_call`` instead.

    Spans must nest on one thread. Use one recorder per independent execution.
    All timing fields are milliseconds. Root inclusive durations or *all*
    exclusive durations are additive; all inclusive durations are not.
    """

    def __init__(
        self,
        *,
        synchronize: Callable[[], Any],
        evaluate: Callable[..., Any],
        clock: Callable[[], float] = perf_counter,
        max_spans: int = 100_000,
    ) -> None:
        if isinstance(max_spans, bool) or not isinstance(max_spans, int) or max_spans < 1:
            raise ValueError("max_spans must be a positive integer")
        self._synchronize = synchronize
        self._evaluate = evaluate
        self._clock = clock
        self._origin = clock()
        self._max_spans = max_spans
        self.spans: list[ProfileSpan] = []
        self._stack: list[ProfileSpan] = []

    def _now_ms(self) -> float:
        return (self._clock() - self._origin) * 1000.0

    @staticmethod
    def _error(exc: BaseException) -> dict[str, str]:
        return {"type": type(exc).__name__, "message": str(exc)}

    def _sync(self, event: ProfileSpan, field_name: str) -> None:
        start = self._clock()
        try:
            self._synchronize()
        finally:
            setattr(event, field_name,
                    getattr(event, field_name) + (self._clock() - start) * 1000.0)

    @contextmanager
    def span(self, name: str, metadata: Mapping[str, Any] | None = None) -> Iterator[ProfileSpan]:
        if len(self.spans) >= self._max_spans:
            raise RuntimeError(f"Profile limit of {self._max_spans} spans exceeded")
        parent = self._stack[-1] if self._stack else None
        event = ProfileSpan(len(self.spans), parent.id if parent else None,
                            str(name), self._now_ms(), dict(metadata or {}))
        self.spans.append(event)
        self._stack.append(event)
        body_error = None
        entry_finished = False
        try:
            self._sync(event, "entry_sync_ms")
            entry_finished = True
            yield event
        except BaseException as exc:
            body_error = exc
            event.error = self._error(exc)
            raise
        finally:
            try:
                if entry_finished:
                    try:
                        self._sync(event, "exit_sync_ms")
                    except BaseException as exc:
                        event.cleanup_error = self._error(exc)
                        if body_error is None:
                            event.error = self._error(exc)
                            raise
            finally:
                event.end_ms = self._now_ms()
                event.inclusive_ms = event.end_ms - event.start_ms
                event.exclusive_ms = max(0.0, event.inclusive_ms - event._children_ms)
                event.status = "error" if event.error else "ok"
                self._stack.pop()
                if parent is not None:
                    parent._children_ms += event.inclusive_ms

    def _materialize(self, event: ProfileSpan, values: Any) -> None:
        leaves = list(_tensors(values))
        if not leaves:
            return
        start = self._clock()
        try:
            self._evaluate(*leaves)
        finally:
            event.materialize_ms += (self._clock() - start) * 1000.0

    def evaluate(self, *values: Any) -> None:
        """Evaluate arrays in the current span, including cache side effects.

        This is useful with a manually scoped ``span``. Outside a span use
        ``materialize(name, values)`` so that the work has a visible owner.
        Stream-specific synchronization remains the injected caller's choice.
        """
        if not self._stack:
            raise RuntimeError("evaluate requires an active span; use materialize outside spans")
        self._materialize(self._stack[-1], values)

    def materialize(self, name: str, values: T,
                    metadata: Mapping[str, Any] | None = None) -> T:
        """Resolve lazy inputs in a separate, visible preparation span."""
        details = {"kind": "input_materialization", "tensors": tensor_metadata(values)}
        details.update(metadata or {})
        with self.span(name, details) as event:
            self._materialize(event, values)
        return values

    def measured_call(
        self,
        name: str,
        fn: Callable[..., T],
        *args: Any,
        metadata: Mapping[str, Any] | None = None,
        materialize: Callable[[], Any] | None = None,
        **kwargs: Any,
    ) -> T:
        """Time a call including evaluation of its output and declared state.

        ``materialize`` returns additional arrays (for example mutated cache
        state) after the call. It should not execute new model operations.
        """
        with self.span(name, metadata) as event:
            result = fn(*args, **kwargs)
            extra = materialize() if materialize is not None else None
            self._materialize(event, (result, extra))
            event.metadata.setdefault("output", tensor_metadata(result))
        return result

    def to_dict(self) -> dict[str, Any]:
        if self._stack:
            raise RuntimeError("Cannot export a profile while spans are active")
        roots = [event for event in self.spans if event.parent_id is None]
        return {
            "schema_version": 1,
            "timing_kind": "synchronized_wall",
            "units": "milliseconds",
            "boundary_sync_included": True,
            "root_total_ms": sum(event.inclusive_ms or 0.0 for event in roots),
            "exclusive_total_ms": sum(event.exclusive_ms or 0.0 for event in self.spans),
            "entry_sync_total_ms": sum(event.entry_sync_ms for event in self.spans),
            "exit_sync_total_ms": sum(event.exit_sync_ms for event in self.spans),
            "materialize_total_ms": sum(event.materialize_ms for event in self.spans),
            "spans": [event.to_dict() for event in self.spans],
        }
