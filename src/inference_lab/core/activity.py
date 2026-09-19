"""Explicit, finite macOS user-initiated activities without a bridge dependency.

The local Foundation SDK declares NSActivityOptions as uint64_t and
NSActivityUserInitiated as 0x00FFFFFF. This includes idle-system-sleep prevention;
it does not include display-sleep or latency-critical options. This is an OS
activity hint, not a request for a particular GPU clock or thread QoS.
"""

from __future__ import annotations

import ctypes
import platform
from typing import Any, Callable


USER_INITIATED_OPTIONS = 0x00FFFFFF


class _FoundationActivityAPI:
    """Own Objective-C references across Python calls using the public API."""

    def __init__(self) -> None:
        if platform.system() != "Darwin":
            raise RuntimeError("--user-initiated requires macOS Foundation")
        self._foundation = ctypes.CDLL(
            "/System/Library/Frameworks/Foundation.framework/Foundation"
        )
        self._objc = ctypes.CDLL("/usr/lib/libobjc.A.dylib")
        self._objc.objc_getClass.argtypes = [ctypes.c_char_p]
        self._objc.objc_getClass.restype = ctypes.c_void_p
        self._objc.sel_registerName.argtypes = [ctypes.c_char_p]
        self._objc.sel_registerName.restype = ctypes.c_void_p
        # objc_msgSend must be cast to the exact non-variadic signature on arm64.
        def message(result, *arguments):
            signature = ctypes.CFUNCTYPE(result, ctypes.c_void_p, ctypes.c_void_p, *arguments)
            return signature(("objc_msgSend", self._objc))

        self._object = message(ctypes.c_void_p)
        self._string = message(ctypes.c_void_p, ctypes.c_char_p)
        self._begin = message(ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p)
        self._void = message(None)
        self._void_object = message(None, ctypes.c_void_p)
        self._selectors: dict[str, Any] = {}

    def _selector(self, name: str) -> Any:
        if name not in self._selectors:
            self._selectors[name] = self._objc.sel_registerName(name.encode("ascii"))
        return self._selectors[name]

    def _class(self, name: str) -> Any:
        value = self._objc.objc_getClass(name.encode("ascii"))
        if not value:
            raise RuntimeError(f"Foundation class is unavailable: {name}")
        return value

    def _pool(self) -> Any:
        pool = self._object(self._class("NSAutoreleasePool"), self._selector("new"))
        if not pool:
            raise RuntimeError("Could not create a Foundation autorelease pool")
        return pool

    def begin(self, options: int, reason: str) -> tuple[Any, Any]:
        pool = self._pool()
        process = token = None
        process_retained = token_retained = False
        try:
            process = self._object(self._class("NSProcessInfo"), self._selector("processInfo"))
            reason_object = self._string(
                self._class("NSString"), self._selector("stringWithUTF8String:"), reason.encode("utf-8")
            )
            if not process or not reason_object:
                raise RuntimeError("Could not create Foundation activity arguments")
            self._object(process, self._selector("retain"))
            process_retained = True
            token = self._begin(
                process, self._selector("beginActivityWithOptions:reason:"), options, reason_object
            )
            if not token:
                raise RuntimeError("NSProcessInfo returned no activity token")
            self._object(token, self._selector("retain"))
            token_retained = True
            return process, token
        except BaseException:
            try:
                if token:
                    self._void_object(process, self._selector("endActivity:"), token)
            finally:
                try:
                    if token_retained:
                        self._void(token, self._selector("release"))
                finally:
                    if process_retained:
                        self._void(process, self._selector("release"))
            raise
        finally:
            self._void(pool, self._selector("drain"))

    def end(self, handle: tuple[Any, Any]) -> None:
        process, token = handle
        # Explicit retains keep both objects alive after begin()'s pool drains.
        # Release them even if ending the activity reports an error.
        try:
            self._void_object(process, self._selector("endActivity:"), token)
        finally:
            try:
                self._void(token, self._selector("release"))
            finally:
                self._void(process, self._selector("release"))


class UserInitiatedActivity:
    """A single-use context manager; disabled mode never loads Foundation."""

    def __init__(
        self,
        enabled: bool = False,
        reason: str = "User-requested finite inference benchmark",
        *,
        _api_factory: Callable[[], Any] | None = None,
    ) -> None:
        if type(enabled) is not bool:
            raise ValueError("enabled must be a boolean")
        if not isinstance(reason, str) or not reason.strip() or "\0" in reason:
            raise ValueError("Activity reason must be a nonempty string without NUL bytes")
        self.enabled = enabled
        self.reason = reason
        self._api_factory = _api_factory or _FoundationActivityAPI
        self._api: Any = None
        self._handle: Any = None
        self._entered = False
        self._started = False
        self._ended = False
        self._cleanup_attempted = False

    def __enter__(self) -> UserInitiatedActivity:
        if self._entered:
            raise RuntimeError("An activity context cannot be entered more than once")
        self._entered = True
        if self.enabled:
            self._api = self._api_factory()
            self._handle = self._api.begin(USER_INITIATED_OPTIONS, self.reason)
            self._started = True
        return self

    def close(self) -> None:
        if self._handle is not None:
            handle, self._handle = self._handle, None
            self._cleanup_attempted = True
            self._api.end(handle)
            self._ended = True

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.close()
        return False

    def metadata(self) -> dict[str, Any]:
        return {
            "user_initiated": self.enabled,
            "api": "NSProcessInfo.beginActivityWithOptions:reason:",
            "options": USER_INITIATED_OPTIONS if self.enabled else 0,
            "options_hex": "0x00FFFFFF" if self.enabled else "0x00000000",
            "reason": self.reason if self.enabled else None,
            "started": self._started,
            "active": self._handle is not None,
            "cleanup_attempted": self._cleanup_attempted,
            "ended": self._ended,
        }
