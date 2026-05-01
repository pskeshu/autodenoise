"""Run the autodenoise pipeline against a watched directory."""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from pathlib import Path

from autodenoise import DirectoryWatcher, Notifier, PassthroughDenoiser


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Watch a directory for new images and denoise them automatically.",
    )
    parser.add_argument("watch_dir", type=Path, help="Directory to monitor.")
    parser.add_argument("output_dir", type=Path, help="Where to write denoised files.")
    parser.add_argument(
        "--extension",
        "-e",
        action="append",
        help="File extension to accept (e.g. .tif). May be repeated. Default: any file.",
    )
    parser.add_argument(
        "--quiet-seconds",
        type=float,
        default=3.0,
        help="Seconds a file must remain unchanged before processing. Default: 3.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=1.0,
        help="Stability poll interval, in seconds. Default: 1.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Watch subdirectories.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    denoiser = PassthroughDenoiser()
    notifier = Notifier()

    def on_stable(src: Path) -> None:
        dst = args.output_dir / src.name
        try:
            denoiser.denoise(src, dst)
        except Exception:
            logging.exception("denoise failed for %s", src)
            return
        notifier.notify(src, dst)

    watcher = DirectoryWatcher(
        watch_dir=args.watch_dir,
        on_stable=on_stable,
        extensions=args.extension,
        quiet_seconds=args.quiet_seconds,
        poll_interval=args.poll_interval,
        recursive=args.recursive,
    )

    stop = False

    def handle_signal(signum, frame):  # noqa: ARG001
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    watcher.start()
    try:
        while not stop:
            time.sleep(0.5)
    finally:
        watcher.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
