"""Detect elapsed host sleep from Apple's two monotonic Mach clocks.

Apple documents the Mach interfaces at https://developer.apple.com/documentation/kernel/mach.
The local macOS SDK's mach/mach_time.h declares mach_continuous_time as the
variant of mach_absolute_time that advances during sleep. Their shared timebase
converts ticks to nanoseconds. Wall-clock time never participates in arithmetic.
"""
from __future__ import annotations

import ctypes
import math
import platform


_UINT64_MAX = (1 << 64) - 1
_UINT32_MAX = (1 << 32) - 1


class _MachTimebaseInfo(ctypes.Structure):
    _fields_ = [("numer", ctypes.c_uint32), ("denom", ctypes.c_uint32)]


def _integer(value, name, *, maximum=_UINT64_MAX, minimum=0):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in {minimum}..{maximum}")
    return value


def _timebase(numer, denom):
    return (_integer(numer, "timebase_numer", maximum=_UINT32_MAX, minimum=1),
            _integer(denom, "timebase_denom", maximum=_UINT32_MAX, minimum=1))


class MacSleepClock:
    """Read-only native clock, with explicit function injection for CPU tests.

    Each sample reads absolute/continuous/absolute and stores the integer
    midpoint of the absolute readings. Sampling uncertainty is bounded by half
    that interval (plus integer rounding), rather than a wall-clock tolerance.
    """

    def __init__(self, *, absolute_time=None, continuous_time=None, timebase=None):
        supplied = (absolute_time is not None, continuous_time is not None, timebase is not None)
        self._library = None
        if not any(supplied):
            if platform.system() != "Darwin":
                raise RuntimeError("MacSleepClock requires macOS Mach clocks")
            library = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
            absolute_time = library.mach_absolute_time
            continuous_time = library.mach_continuous_time
            for function in (absolute_time, continuous_time):
                function.argtypes = []
                function.restype = ctypes.c_uint64
            info_function = library.mach_timebase_info
            info_function.argtypes = [ctypes.POINTER(_MachTimebaseInfo)]
            info_function.restype = ctypes.c_int
            info = _MachTimebaseInfo()
            status = info_function(ctypes.byref(info))
            if status != 0:
                raise RuntimeError(f"mach_timebase_info failed with kern_return_t={status}")
            timebase = (int(info.numer), int(info.denom))
            self._library = library
        elif not all(supplied):
            raise ValueError("Inject absolute_time, continuous_time and timebase together")
        if not callable(absolute_time) or not callable(continuous_time):
            raise ValueError("Injected clocks must be callable")
        if not isinstance(timebase, (tuple, list)) or len(timebase) != 2:
            raise ValueError("timebase must contain its numerator and denominator")
        self._numer, self._denom = _timebase(*timebase)
        self._absolute_time = absolute_time
        self._continuous_time = continuous_time

    def snapshot(self) -> dict:
        first = _integer(self._absolute_time(), "first absolute tick")
        continuous = _integer(self._continuous_time(), "continuous tick")
        last = _integer(self._absolute_time(), "last absolute tick")
        if last < first:
            raise ValueError("Absolute clock moved backwards during sampling")
        return {
            "schema_version": 1,
            "absolute_ticks": (first + last) // 2,
            "continuous_ticks": continuous,
            "timebase_numer": self._numer,
            "timebase_denom": self._denom,
            "sampling_span_ticks": last - first,
        }

    @staticmethod
    def _validate(snapshot):
        if not isinstance(snapshot, dict) or type(snapshot.get("schema_version")) is not int or snapshot["schema_version"] != 1:
            raise ValueError("Unsupported Mach clock snapshot schema")
        absolute = _integer(snapshot.get("absolute_ticks"), "absolute_ticks")
        continuous = _integer(snapshot.get("continuous_ticks"), "continuous_ticks")
        span = _integer(snapshot.get("sampling_span_ticks"), "sampling_span_ticks")
        numer, denom = _timebase(snapshot.get("timebase_numer"), snapshot.get("timebase_denom"))
        if absolute - span // 2 < 0 or absolute + (span + 1) // 2 > _UINT64_MAX:
            raise ValueError("Sampling span cannot describe valid absolute-clock endpoints")
        return absolute, continuous, span, numer, denom

    @staticmethod
    def assess(before: dict, after: dict, threshold_seconds=1.0) -> dict:
        """Compare integer deltas, using continuous-minus-awake as sleep time.

        The estimate can be slightly negative because the reads are sequential.
        Clamp only a negative value within the two midpoint sampling intervals
        plus one tick of rounding; a stronger inconsistency is an error. A
        positive estimate must strictly exceed the threshold to mark sleep.
        Snapshots from different boots or machines must not be combined.
        """
        if type(threshold_seconds) not in (int, float):
            raise ValueError("threshold_seconds must be finite and nonnegative")
        try:
            threshold = float(threshold_seconds)
        except (ValueError, OverflowError) as error:
            raise ValueError("threshold_seconds must be finite and nonnegative") from error
        if not math.isfinite(threshold) or threshold < 0:
            raise ValueError("threshold_seconds must be finite and nonnegative")
        b_abs, b_cont, b_span, b_num, b_den = MacSleepClock._validate(before)
        a_abs, a_cont, a_span, a_num, a_den = MacSleepClock._validate(after)
        if (b_num, b_den) != (a_num, a_den):
            raise ValueError("Mach timebases differ between snapshots")
        awake_ticks, continuous_ticks = a_abs - b_abs, a_cont - b_cont
        if awake_ticks < 0 or continuous_ticks < 0:
            raise ValueError("Mach clocks moved backwards between snapshots")
        sleep_ticks = continuous_ticks - awake_ticks
        rounding_bound = (b_span + a_span + 1) // 2 + 1
        if sleep_ticks < -rounding_bound:
            raise ValueError("Continuous interval is shorter than awake interval beyond sampling uncertainty")
        sleep_ticks = max(0, sleep_ticks)

        def seconds(ticks):
            # Subtract large boot-time counters while still integers, then scale.
            return (ticks * b_num) / (b_den * 1_000_000_000)

        sleep_seconds = seconds(sleep_ticks)
        return {
            "continuous_elapsed_seconds": seconds(continuous_ticks),
            "awake_elapsed_seconds": seconds(awake_ticks),
            "sleep_seconds": sleep_seconds,
            "threshold_seconds": threshold,
            "sleep_detected": sleep_seconds > threshold,
        }
