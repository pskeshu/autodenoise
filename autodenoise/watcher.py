"""Directory watcher with stability check.

Uses ``watchdog`` for OS-agnostic filesystem events (inotify on Linux,
FSEvents on macOS, ReadDirectoryChangesW on Windows). On every create or
modify event we record the file's size + mtime; a background poller then
checks whether those values have held steady for ``quiet_seconds`` and, if
so, hands the path to a callback.

The stability check exists because a "file created" event fires the moment
the file appears, not when the writer finishes. Microscope acquisitions can
take seconds to minutes to flush a multi-GB stack; opening it too early
yields a truncated read.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Callable, Iterable, Optional

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

log = logging.getLogger(__name__)


class _PendingTracker:
    """Tracks files awaiting stability. Thread-safe."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # path -> (size, mtime, monotonic-time-when-these-values-were-recorded)
        self._pending: dict[Path, tuple[int, float, float]] = {}

    def touch(self, path: Path) -> None:
        with self._lock:
            try:
                stat = path.stat()
            except FileNotFoundError:
                self._pending.pop(path, None)
                return
            self._pending[path] = (stat.st_size, stat.st_mtime, time.monotonic())

    def stable_paths(self, quiet_seconds: float) -> list[Path]:
        ready: list[Path] = []
        now = time.monotonic()
        with self._lock:
            for path, (size, mtime, observed_at) in list(self._pending.items()):
                try:
                    stat = path.stat()
                except FileNotFoundError:
                    self._pending.pop(path, None)
                    continue
                if (stat.st_size, stat.st_mtime) != (size, mtime):
                    self._pending[path] = (stat.st_size, stat.st_mtime, now)
                    continue
                if now - observed_at >= quiet_seconds:
                    ready.append(path)
                    self._pending.pop(path, None)
        return ready


class _Handler(FileSystemEventHandler):
    def __init__(self, tracker: _PendingTracker, extensions: Optional[set[str]]) -> None:
        self._tracker = tracker
        self._extensions = extensions

    def _accept(self, path: Path) -> bool:
        if self._extensions is None:
            return True
        return path.suffix.lower() in self._extensions

    def on_created(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        path = Path(event.src_path)
        if self._accept(path):
            log.debug("created: %s", path)
            self._tracker.touch(path)

    def on_modified(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        path = Path(event.src_path)
        if self._accept(path):
            self._tracker.touch(path)

    def on_moved(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        dest = Path(getattr(event, "dest_path", ""))
        if dest and self._accept(dest):
            log.debug("moved: %s -> %s", event.src_path, dest)
            self._tracker.touch(dest)


class DirectoryWatcher:
    """Watch a directory for new files; call ``on_stable`` once each settles.

    A file is considered ready when its size and mtime have not changed for
    ``quiet_seconds`` consecutive seconds.
    """

    def __init__(
        self,
        watch_dir: Path,
        on_stable: Callable[[Path], None],
        extensions: Optional[Iterable[str]] = None,
        quiet_seconds: float = 3.0,
        poll_interval: float = 1.0,
        recursive: bool = False,
    ) -> None:
        self.watch_dir = Path(watch_dir).resolve()
        self.on_stable = on_stable
        self.extensions = (
            {e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions}
            if extensions is not None
            else None
        )
        self.quiet_seconds = quiet_seconds
        self.poll_interval = poll_interval
        self.recursive = recursive

        self._tracker = _PendingTracker()
        self._observer: Optional[Observer] = None
        self._stop = threading.Event()
        self._poller: Optional[threading.Thread] = None

    def start(self) -> None:
        if not self.watch_dir.is_dir():
            raise FileNotFoundError(f"watch dir does not exist: {self.watch_dir}")

        handler = _Handler(self._tracker, self.extensions)
        self._observer = Observer()
        self._observer.schedule(handler, str(self.watch_dir), recursive=self.recursive)
        self._observer.start()

        self._poller = threading.Thread(
            target=self._poll_loop, name="autodenoise-stability", daemon=True
        )
        self._poller.start()

        log.info(
            "watching %s (extensions=%s, quiet_seconds=%s, recursive=%s)",
            self.watch_dir,
            self.extensions if self.extensions else "any",
            self.quiet_seconds,
            self.recursive,
        )

    def stop(self) -> None:
        self._stop.set()
        if self._observer is not None:
            self._observer.stop()
            self._observer.join()
        if self._poller is not None:
            self._poller.join(timeout=self.poll_interval * 2)

    def _poll_loop(self) -> None:
        while not self._stop.is_set():
            for path in self._tracker.stable_paths(self.quiet_seconds):
                try:
                    self.on_stable(path)
                except Exception:
                    log.exception("on_stable failed for %s", path)
            self._stop.wait(self.poll_interval)
