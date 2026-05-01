# install.md — agent onboarding

Instructions for a coding agent picking up this repo. Read this once before
making changes.

## What this project is

`autodenoise` is a long-lived process that watches a directory for new
microscopy images, runs them through a denoising backend, writes the result
to an output directory, and notifies the user. It is structured as a small
Python package plus a CLI runner.

The shipped denoiser is a passthrough copy. The expected next step is to
swap it for a real model (e.g. CAREamics n2v).

## Requirements

These are hard project constraints. Treat them as non-negotiable unless the
human explicitly relaxes one.

- **OS-agnostic.** Must run on Linux, macOS, and Windows. Users come from
  any platform — microscope rigs in particular are often Windows.
  Concretely:
  - Use `pathlib.Path` everywhere; never assemble paths with string
    concatenation or hardcoded `/` separators.
  - Filesystem events go through `watchdog`, which abstracts inotify
    (Linux), FSEvents (macOS), and ReadDirectoryChangesW (Windows). Do not
    bypass it.
  - Anything that shells out to a platform-specific binary
    (`notify-send`, `osascript`, etc.) must `shutil.which` first and
    degrade gracefully — see `notify.py` for the pattern.
  - File-extension matching is case-insensitive (Windows is
    case-insensitive on disk; users will type `.TIF` and `.tif`
    interchangeably).
  - Do not hardcode `/tmp`, `~`, or any POSIX-only path inside code. Use
    `tempfile.gettempdir()`, `Path.home()`, or take the path as an
    argument. Examples in docs may use `/tmp` for brevity; library code
    must not.
  - Signals: `SIGINT` is portable; `SIGTERM` is registered but Windows
    handling is best-effort. Do not add `SIGHUP`, `SIGUSR1`, or other
    POSIX-only signals.
  - Subprocess calls always pass an arg list and `shell=False`.
- **Python 3.9+.** Use `from __future__ import annotations` so newer typing
  syntax (e.g. `list[Path]`, `dict[str, int]`) works under 3.9.
- **GPU optional, detect at runtime.** A real denoiser should use the GPU
  when one is available and fall back to CPU when it isn't. Detect inside
  the `Denoiser` subclass — e.g. `torch.cuda.is_available()` for PyTorch,
  `tf.config.list_physical_devices('GPU')` for TensorFlow — and log which
  device was selected. CPU-only installs of the project must still work
  end-to-end; do not list `cuda` / `nvidia-*` packages as hard
  dependencies in `requirements.txt`.
- **Minimal dependencies.** The watcher pipeline depends only on
  `watchdog`. Heavy ML stacks (`torch`, `careamics`, `tifffile`, …) are
  pulled in lazily inside a `Denoiser` subclass, never at package
  top-level.

## Setup

```bash
cd autodenoise
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The only hard dependency is `watchdog`. Add new dependencies by appending to
`requirements.txt` (pinned with `>=`, not `==`, unless there is a known
incompatibility).

## Repo layout

```
autodenoise/                  # project root (git repo)
├── README.md                 # user-facing docs
├── install.md                # this file
├── requirements.txt
├── run.py                    # CLI entry point
└── autodenoise/              # Python package
    ├── __init__.py           # public API: DirectoryWatcher, Denoiser,
    │                         #   PassthroughDenoiser, Notifier
    ├── watcher.py            # DirectoryWatcher + stability check
    ├── denoiser.py           # Denoiser ABC + PassthroughDenoiser stub
    └── notify.py             # console + notify-send / osascript
```

The same-name nesting (`autodenoise/autodenoise/`) is intentional and
standard. `run.py` imports the package as `autodenoise`.

## How the watcher works (read before modifying)

`DirectoryWatcher` runs two threads:

1. **Observer thread** (from `watchdog`) — fires on `created` / `modified` /
   `moved` events and records `(size, mtime, monotonic_now)` for the path in
   a thread-safe `_PendingTracker`.
2. **Poller thread** — every `poll_interval` seconds, checks each pending
   path: if `(size, mtime)` has changed since the last record, the clock
   resets; if it has held steady for `quiet_seconds`, the path is handed to
   the `on_stable` callback and removed from the tracker.

The stability check exists because filesystem-create events fire when the
file appears, not when the writer finishes. Multi-GB stacks take seconds to
flush; opening too early yields a truncated read.

There is **no startup scan** of pre-existing files. Only files that arrive
after `watcher.start()` are processed. This is deliberate — keep it that way
unless explicitly asked to add a `--process-existing` flag.

## How to add a real denoiser

The single extension point is `autodenoise.Denoiser`. Subclass it, load
weights / config in `__init__`, run inference in `denoise(src, dst)`. Then
swap the instance in `run.py`.

Example skeleton (do not commit this — it is illustrative):

```python
from pathlib import Path
from autodenoise import Denoiser

class CAREamicsDenoiser(Denoiser):
    def __init__(self, weights: Path):
        from careamics import CAREamist
        self.model = CAREamist.from_pretrained(weights)

    def denoise(self, src: Path, dst: Path) -> None:
        import tifffile
        img = tifffile.imread(src)
        out = self.model.predict(img)
        tifffile.imwrite(dst, out)
```

Heavy imports (`careamics`, `tifffile`, `torch`, etc.) belong **inside**
`__init__` or `denoise`, not at module top level — this keeps the watcher's
import path light and the CLI snappy.

If you add a real backend, also:

- Add it to `requirements.txt`.
- Expose a CLI flag in `run.py` to select between backends, defaulting to
  `PassthroughDenoiser` so the existing smoke test keeps working.
- Update `README.md` (the "Plugging in a real denoiser" section) and add a
  one-line entry in `install.md` under "How to add a real denoiser".

## Verification before reporting a task done

Run all three:

```bash
# 1. Syntax / import sanity
python -m py_compile run.py autodenoise/*.py

# 2. CLI parses
python run.py --help

# 3. End-to-end smoke test (in two terminals)
mkdir -p /tmp/ad-in /tmp/ad-out
python run.py /tmp/ad-in /tmp/ad-out --quiet-seconds 2 --log-level DEBUG
# In another terminal:
cp some-image.tif /tmp/ad-in/
# Confirm the file appears in /tmp/ad-out/ within a few seconds.
```

If you change `watcher.py`, the smoke test is non-optional.

## Conventions

- **OS-agnostic by default.** Re-read the Requirements section above before
  touching anything that talks to the filesystem, the shell, or the OS.
  When in doubt, prefer the stdlib (`pathlib`, `tempfile`, `shutil`,
  `subprocess` with arg lists) over OS-specific commands.
- **Type hints** on all public signatures. Use `from __future__ import
  annotations` at the top of new modules so quoting is unnecessary.
- **Logging, not prints.** Each module uses `log = logging.getLogger(__name__)`.
  `run.py` configures the root logger; library code never calls
  `logging.basicConfig`.
- **Comments are scarce on purpose.** Only write a comment when the *why* is
  non-obvious. Do not narrate what the next line does.
- **No new top-level files** unless they are necessary. Prefer extending an
  existing module.
- **Do not** add a startup scan, a database, a web UI, or a config file
  format unless asked. Keep the surface area minimal.
- **Do not** commit. Stage and report changes; the human runs `git commit`.

## Out of scope (do not do without being asked)

- Replacing `watchdog` with a different watcher library.
- Adding async/await — the threaded design is intentional.
- Persisting tracker state across restarts.
- Integrating with a queue (Redis, RabbitMQ, etc.).
- Packaging as a `pip`-installable distribution (`pyproject.toml`, entry
  points). The CLI is invoked as `python run.py`.
