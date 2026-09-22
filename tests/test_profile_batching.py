"""CPU-only coverage of cache isolation and fixed-prefix batching accounting."""
from contextlib import contextmanager
import json
from types import SimpleNamespace

import numpy as np
import pytest

from inference_lab.profiling.batching import _clone_cache, run_batching_probe


class ArrayRuntime:
    uint32 = np.uint32
    float32 = np.float32
    argmax = staticmethod(np.argmax)
    concatenate = staticmethod(np.concatenate)
    max = staticmethod(np.max)
    abs = staticmethod(np.abs)

    def __init__(self):
        self.events = []
        self.active = None

    def array(self, value, dtype=None):
        return np.array(value, dtype=dtype, copy=True)

    @contextmanager
    def stream(self, stream):
        before, self.active = self.active, stream
        try:
            yield
        finally:
            self.active = before

    def eval(self, *values):
        assert self.active is not None
        self.events.append("eval")

    def synchronize(self, stream):
        self.events.append(("sync", stream))


class KV:
    def __init__(self):
        self.keys = np.zeros((1, 1, 64, 1), dtype=np.float32)
        self.values = self.keys.copy()
        self.offset = 0


class Recurrent:
    def __init__(self):
        self.cache = [np.zeros((1, 3, 1), dtype=np.float32), np.zeros((1,), dtype=np.float32)]
        self.lengths = None
        self.left_padding = None


def backend_fixture(*, block_error=0.0, fail_block=False):
    mx = ArrayRuntime()
    status = {"capture": False, "created": 0, "closed": 0, "wired": False, "calls": []}

    class Capture:
        def __init__(self):
            assert not status["capture"]
            status["capture"] = True
            status["created"] += 1

        def close(self):
            status["capture"] = False
            status["closed"] += 1

    @contextmanager
    def wired_limit(model, streams):
        status["wired"] = True
        try:
            yield
        finally:
            status["wired"] = False

    class Model:
        def __init__(self):
            self._hidden_states = ["previous hidden"]

        def __call__(self, inputs, cache):
            assert status["wired"]
            if fail_block and status["capture"]:
                raise RuntimeError("injected forward failure")
            status["calls"].append((status["capture"], inputs.shape[1], cache[0].offset))
            logits = np.zeros((1, inputs.shape[1], 17), dtype=np.float32)
            for position, token in enumerate(inputs[0]):
                slot = cache[0].offset
                cache[0].keys[0, 0, slot, 0] = token
                cache[0].values[0, 0, slot, 0] = token * 2
                cache[0].offset += 1
                cache[1].cache[0] = np.concatenate([cache[1].cache[0][:, 1:], np.array([[[token]]])], axis=1)
                cache[1].cache[1] = cache[1].cache[1] + token
                prediction = int(cache[1].cache[1][0]) % 17
                logits[0, position, prediction] = 1
            if status["capture"]:
                cache[1].cache[1] += block_error
            self._hidden_states[0] = np.repeat(inputs[:, :, None], 2, axis=-1)
            return logits

    backend = SimpleNamespace(
        _mx=mx, _model=Model(), _draft=SimpleNamespace(config=SimpleNamespace(target_layer_ids=[0])),
        prefill_step_size=3,
        _upstream=SimpleNamespace(
            _GDNStateCapture=Capture, generation_stream=object(), wired_limit=wired_limit,
            _patch_model=lambda *args: None, make_prompt_cache=lambda model: [KV(), Recurrent()],
        ),
    )
    return backend, status


def test_clone_preserves_capacity_metadata_and_isolates_mutable_arrays_and_lists():
    mx = ArrayRuntime()
    original = [KV(), Recurrent()]
    original[0].offset = 2
    cloned = _clone_cache(original, mx)
    assert cloned[0].keys.shape == (1, 1, 64, 1)
    assert cloned[0].offset == 2
    cloned[0].keys[0, 0, 0, 0] = 99
    cloned[1].cache[1][0] = 10
    cloned[1].cache.append("new")
    assert original[0].keys[0, 0, 0, 0] == 0
    assert original[1].cache[1][0] == 0
    assert len(original[1].cache) == 2


def test_fixed_prefix_probe_parity_warmups_order_streams_and_json():
    backend, status = backend_fixture()
    result = run_batching_probe(backend, [1, 2], list(range(3, 30)), widths=(1, 3), repeats=3, prefix_generated=2)
    assert result["prefix_tokens"] == 4
    assert [row["width"] for row in result["widths"]] == [1, 3]
    for width_result in result["widths"]:
        assert width_result["all_argmax_equal"]
        assert width_result["all_cache_within_atol"]
        assert width_result["max_cache_abs_error"] == 0
        assert len(width_result["trials"]) == 3
        assert [row["order"] for row in width_result["trials"]] == [
            ["block", "sequential"], ["sequential", "block"], ["block", "sequential"]]
        assert width_result["block_ms"]["min"] > 0
    assert result["widths"][1]["input_token_ids"] == [5, 6, 7]
    assert status["created"] == status["closed"] == 8  # Per width: warmup + three repeats.
    assert not status["capture"] and not status["wired"]
    assert backend._model._hidden_states == ["previous hidden"]
    assert all(event[1] is backend._upstream.generation_stream for event in backend._mx.events if isinstance(event, tuple))
    assert all(offset == 4 for captured, width, offset in status["calls"] if captured)
    assert result["cache_layout"][0]["allocated_tensors"][0]["shape"] == [1, 1, 64, 1]
    json.dumps(result, allow_nan=False)


def test_state_error_is_reported_separately_from_argmax_parity():
    backend, _ = backend_fixture(block_error=0.01)
    result = run_batching_probe(backend, [1], list(range(2, 20)), widths=(3,), repeats=1, prefix_generated=2)
    width = result["widths"][0]
    assert width["all_argmax_equal"]
    assert not width["all_cache_within_atol"]
    assert width["max_cache_abs_error"] == pytest.approx(0.01, abs=1e-5)


def test_failure_restores_capture_wired_policy_and_hidden_state():
    backend, status = backend_fixture(fail_block=True)
    with pytest.raises(RuntimeError, match="injected"):
        run_batching_probe(backend, [1], list(range(2, 20)), widths=(1,), repeats=1, prefix_generated=2)
    assert status["created"] == status["closed"] == 1
    assert not status["capture"] and not status["wired"]
    assert backend._model._hidden_states == ["previous hidden"]


@pytest.mark.parametrize("kwargs,match", [
    ({"widths": (1, 1)}, "distinct"),
    ({"widths": (True,)}, "distinct"),
    ({"widths": ()}, "distinct"),
    ({"repeats": 0}, "repeats"),
    ({"prefix_generated": -1}, "prefix_generated"),
    ({"prefix_generated": 19}, "too short"),
])
def test_invalid_probe_inputs_fail_before_work(kwargs, match):
    backend, status = backend_fixture()
    options = {"widths": (1,), "repeats": 1, "prefix_generated": 2, **kwargs}
    with pytest.raises(ValueError, match=match):
        run_batching_probe(backend, [1], list(range(2, 20)), **options)
    assert not status["calls"]
