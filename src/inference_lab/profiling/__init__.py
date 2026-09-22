"""Optional, framework-independent synchronized wall-time profiling."""

from .recorder import ProfileRecorder, ProfileSpan, tensor_metadata

__all__ = ["ProfileRecorder", "ProfileSpan", "tensor_metadata"]
