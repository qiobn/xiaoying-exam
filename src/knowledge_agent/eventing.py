from __future__ import annotations

import threading
import time

from .config import Settings
from .persistence import Store


class BufferedRunEventPublisher:
    """Persist audit events immediately and coalesce transient token deltas."""

    def __init__(self, store: Store, run_id: str, settings: Settings):
        self.store = store
        self.run_id = run_id
        self.batch_chars = max(1, settings.stream_token_batch_chars)
        self.batch_seconds = max(0.05, settings.stream_token_batch_ms / 1000)
        self._buffer: list[str] = []
        self._phase = "answer"
        self._last_flush = time.monotonic()
        self._lock = threading.Lock()

    def __call__(self, event_type: str, payload: dict) -> None:
        if event_type != "token":
            self.store.add_event(self.run_id, event_type, payload)
            return
        with self._lock:
            phase = str(payload.get("phase", "answer"))
            if phase != self._phase:
                self._flush_locked()
                self._phase = phase
            delta = str(payload.get("delta", ""))
            if delta:
                self._buffer.append(delta)
            size = sum(len(item) for item in self._buffer)
            elapsed = time.monotonic() - self._last_flush
            if size >= self.batch_chars or elapsed >= self.batch_seconds:
                self._flush_locked()

    def flush(self) -> None:
        with self._lock:
            self._flush_locked()

    def _flush_locked(self) -> None:
        if not self._buffer:
            self._last_flush = time.monotonic()
            return
        delta = "".join(self._buffer)
        self._buffer.clear()
        self.store.add_event(
            self.run_id,
            "token",
            {"delta": delta, "phase": self._phase, "batched": True},
        )
        self._last_flush = time.monotonic()
