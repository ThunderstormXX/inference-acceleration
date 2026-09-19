"""CPU-only Mach clock tests; injected counters never put the Mac to sleep."""
import ctypes
from copy import deepcopy
import time

import pytest

from inference_lab.core.host_clock import MacSleepClock, _MachTimebaseInfo


def sample(absolute=1_000, continuous=2_000, *, numer=1, denom=1, span=0):
    return dict(schema_version=1, absolute_ticks=absolute, continuous_ticks=continuous,
                timebase_numer=numer, timebase_denom=denom, sampling_span_ticks=span)


def test_snapshot_reads_absolute_continuous_absolute_and_uses_integer_midpoint():
    reads, absolute = [], iter([2**63 + 10, 2**63 + 15])

    def abs_clock():
        reads.append("absolute")
        return next(absolute)

    def cont_clock():
        reads.append("continuous")
        return 2**63 + 200

    clock = MacSleepClock(absolute_time=abs_clock, continuous_time=cont_clock, timebase=(125, 3))
    assert clock.snapshot() == sample(2**63 + 12, 2**63 + 200, numer=125, denom=3, span=5)
    assert reads == ["absolute", "continuous", "absolute"]


@pytest.mark.parametrize("numer,denom,ticks_per_second", [(1, 1, 1_000_000_000), (125, 3, 24_000_000), (5, 2, 400_000_000)])
def test_timebase_conversion_and_sleep_delta(numer, denom, ticks_per_second):
    before = sample(numer=numer, denom=denom)
    after = sample(1000 + 2*ticks_per_second, 2000 + 5*ticks_per_second, numer=numer, denom=denom)
    assert MacSleepClock.assess(before, after) == {
        "continuous_elapsed_seconds": 5.0, "awake_elapsed_seconds": 2.0,
        "sleep_seconds": 3.0, "threshold_seconds": 1.0, "sleep_detected": True,
    }


def test_wall_clock_jumps_are_irrelevant(monkeypatch):
    monkeypatch.setattr(time, "time", lambda: (_ for _ in ()).throw(AssertionError("Wall time used")))
    monkeypatch.setattr(time, "perf_counter", lambda: (_ for _ in ()).throw(AssertionError("Python time used")))
    before = sample(2**63, 2**63 + 100)
    after = sample(2**63 + 11, 2**63 + 100 + 17)
    result = MacSleepClock.assess(before, after, threshold_seconds=0)
    assert result["awake_elapsed_seconds"] == 11e-9
    assert result["continuous_elapsed_seconds"] == 17e-9
    assert result["sleep_seconds"] == 6e-9
    assert result["sleep_detected"] is True


def test_detection_is_strictly_greater_than_threshold():
    before = sample()
    after = sample(1_000 + 1_000_000_000, 2_000 + 2_000_000_000)
    assert MacSleepClock.assess(before, after)["sleep_detected"] is False
    after["continuous_ticks"] += 1
    assert MacSleepClock.assess(before, after)["sleep_detected"] is True


def test_zero_interval_and_sampling_jitter_do_not_claim_sleep():
    before = sample(span=10)
    assert MacSleepClock.assess(before, before)["sleep_seconds"] == 0
    after = sample(1100, 2090, span=10)
    result = MacSleepClock.assess(before, after)
    assert result["sleep_seconds"] == 0
    assert result["sleep_detected"] is False
    after["continuous_ticks"] -= 2
    with pytest.raises(ValueError, match="sampling uncertainty"):
        MacSleepClock.assess(before, after)


@pytest.mark.parametrize("field,value", [
    ("schema_version", True), ("schema_version", 2), ("absolute_ticks", True),
    ("absolute_ticks", -1), ("continuous_ticks", 2.5), ("continuous_ticks", 2**64),
    ("sampling_span_ticks", -1), ("sampling_span_ticks", 10_000),
    ("timebase_numer", 0), ("timebase_denom", 0), ("timebase_numer", True),
    ("timebase_denom", 2**32),
])
def test_invalid_snapshot_values_are_rejected(field, value):
    corrupted = sample()
    corrupted[field] = value
    with pytest.raises(ValueError):
        MacSleepClock.assess(sample(), corrupted)


@pytest.mark.parametrize("threshold", [-1, float("nan"), float("inf"), True, "1", None, 10**1000])
def test_invalid_thresholds_are_rejected(threshold):
    with pytest.raises(ValueError, match="threshold_seconds"):
        MacSleepClock.assess(sample(), sample(), threshold)


@pytest.mark.parametrize("field", ["absolute_ticks", "continuous_ticks"])
def test_backwards_ticks_are_rejected(field):
    before, after = sample(), sample()
    after[field] -= 1
    with pytest.raises(ValueError, match="backwards"):
        MacSleepClock.assess(before, after)


def test_different_timebase_is_rejected_even_when_ratio_is_equivalent():
    with pytest.raises(ValueError, match="timebases differ"):
        MacSleepClock.assess(sample(), sample(numer=2, denom=2))


def test_snapshot_rejects_backwards_native_read():
    ticks = iter([11, 10])
    clock = MacSleepClock(absolute_time=lambda: next(ticks), continuous_time=lambda: 12, timebase=(1, 1))
    with pytest.raises(ValueError, match="backwards during sampling"):
        clock.snapshot()


@pytest.mark.parametrize("kwargs", [
    {"absolute_time": lambda: 1},
    {"absolute_time": 1, "continuous_time": lambda: 1, "timebase": (1, 1)},
    {"absolute_time": lambda: 1, "continuous_time": lambda: 1, "timebase": (0, 1)},
])
def test_incomplete_or_invalid_injection_is_rejected(kwargs):
    with pytest.raises(ValueError):
        MacSleepClock(**kwargs)


class NativeFunction:
    def __init__(self, function):
        self.function = function

    def __call__(self, *args):
        return self.function(*args)


@pytest.mark.parametrize("status", [0, 5])
def test_native_library_signatures_and_timebase_status(monkeypatch, status):
    from types import SimpleNamespace
    import inference_lab.core.host_clock as module
    absolute = iter([100, 104])

    def timebase(pointer):
        info = ctypes.cast(pointer, ctypes.POINTER(_MachTimebaseInfo)).contents
        info.numer, info.denom = 125, 3
        return status

    library = SimpleNamespace(
        mach_absolute_time=NativeFunction(lambda: next(absolute)),
        mach_continuous_time=NativeFunction(lambda: 200),
        mach_timebase_info=NativeFunction(timebase),
    )
    paths = []
    monkeypatch.setattr(module.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(module.ctypes, "CDLL", lambda path: paths.append(path) or library)
    if status:
        with pytest.raises(RuntimeError, match="kern_return_t=5"):
            MacSleepClock()
    else:
        clock = MacSleepClock()
        assert clock.snapshot() == sample(102, 200, numer=125, denom=3, span=4)
        assert clock._library is library
    assert paths == ["/usr/lib/libSystem.B.dylib"]
    assert library.mach_absolute_time.argtypes == []
    assert library.mach_absolute_time.restype is ctypes.c_uint64
    assert library.mach_continuous_time.restype is ctypes.c_uint64
    assert library.mach_timebase_info.restype is ctypes.c_int


def test_non_macos_native_clock_fails_before_library_load(monkeypatch):
    import inference_lab.core.host_clock as module
    monkeypatch.setattr(module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(module.ctypes, "CDLL", lambda path: pytest.fail("Unexpected library load"))
    with pytest.raises(RuntimeError, match="macOS"):
        MacSleepClock()
