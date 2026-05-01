"""User notifications. Always logs; best-effort desktop popup on Linux/macOS."""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


class Notifier:
    def __init__(self, app_name: str = "autodenoise") -> None:
        self.app_name = app_name
        self._desktop = self._detect_desktop()

    @staticmethod
    def _detect_desktop() -> Optional[str]:
        if sys.platform.startswith("linux") and shutil.which("notify-send"):
            return "notify-send"
        if sys.platform == "darwin":
            return "osascript"
        return None

    def notify(self, src: Path, dst: Path) -> None:
        msg = f"denoised: {src.name} -> {dst}"
        log.info(msg)
        try:
            if self._desktop == "notify-send":
                subprocess.run(
                    ["notify-send", self.app_name, msg],
                    check=False,
                    timeout=2,
                )
            elif self._desktop == "osascript":
                subprocess.run(
                    [
                        "osascript",
                        "-e",
                        f'display notification "{msg}" with title "{self.app_name}"',
                    ],
                    check=False,
                    timeout=2,
                )
        except Exception:
            log.debug("desktop notification failed", exc_info=True)
