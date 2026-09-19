"""Scoped MLX wired-memory policy, matching MLX-LM's generation helper."""

from typing import Any


class RecommendedWiredMemory:
    """Temporarily use Metal's recommended working-set size, then restore it.

    Only the current process's MLX allocator policy changes. Synchronization and
    limit changes surround the measured request, outside its phase timers.
    """

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime
        self._old_limit: int | None = None

    def __enter__(self):
        if self._old_limit is not None:
            raise RuntimeError("Wired-memory context is already active")
        recommended = self.runtime.device_info().get("max_recommended_working_set_size")
        if type(recommended) is not int or recommended <= 0:
            raise RuntimeError("MLX did not provide a positive recommended working-set size")
        self.runtime.synchronize()
        self._old_limit = self.runtime.set_wired_limit(recommended)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        old_limit, self._old_limit = self._old_limit, None
        if old_limit is not None:
            try:
                self.runtime.synchronize()
            finally:
                self.runtime.set_wired_limit(old_limit)
        return False
