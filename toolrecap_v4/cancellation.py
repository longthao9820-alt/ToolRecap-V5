"""Cancellation primitives for cooperative cancellation."""

from __future__ import annotations

import threading
from typing import Callable, List

from toolrecap_v4.errors import CancelledError


class CancellationToken:
    """Thread-safe cancellation token for cooperative cancellation."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._callbacks: List[Callable[[], None]] = []

    @property
    def is_cancelled(self) -> bool:
        """Return True if cancellation has been requested."""
        return self._event.is_set()

    def cancel(self) -> None:
        """Request cancellation and execute registered callbacks."""
        with self._lock:
            if self._event.is_set():
                return
            self._event.set()
            callbacks = list(self._callbacks)
            self._callbacks.clear()

        for callback in callbacks:
            try:
                callback()
            except Exception:
                # Do not allow callback failures to prevent cancellation
                pass

    def check_cancelled(self) -> None:
        """Raise CancelledError if cancellation has been requested."""
        if self.is_cancelled:
            raise CancelledError("Operation was cancelled.")

    def register_callback(self, callback: Callable[[], None]) -> None:
        """Register a callback to be called when cancellation is requested.
        
        If already cancelled, invokes callback immediately.
        """
        with self._lock:
            if not self._event.is_set():
                self._callbacks.append(callback)
                return

        # Already cancelled, invoke outside lock
        try:
            callback()
        except Exception:
            pass
