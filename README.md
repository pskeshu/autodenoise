# autodenoise

Watch a directory for new microscopy images, denoise them with a configurable
backend, and notify the user when each file is ready.

## Why

Acquisition writes raw frames to a folder; downstream analysis expects
denoised data. Doing this by hand breaks the loop. `autodenoise` runs as a
long-lived process: drop a file in, get a denoised copy out, get pinged.

## Layout

```
ht/
├── autodenoise/
│   ├── watcher.py    # watchdog-based directory watcher with stability check
│   ├── denoiser.py   # Denoiser ABC + PassthroughDenoiser placeholder
│   └── notify.py     # console + desktop notifications
├── run.py            # CLI entry point
├── requirements.txt
└── README.md
```

`DirectoryWatcher` uses [`watchdog`](https://pypi.org/project/watchdog/) for
OS-agnostic events (inotify on Linux, FSEvents on macOS, ReadDirectoryChangesW
on Windows), then debounces with a polling stability check: a file must hold
the same size and mtime for `--quiet-seconds` consecutive seconds before it is
handed to the denoiser. This avoids opening half-written multi-GB stacks.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python run.py /path/to/incoming /path/to/denoised \
    --extension .tif --extension .tiff \
    --quiet-seconds 5
```

`Ctrl-C` to stop.

Useful flags:

- `--extension / -e` — repeatable; restrict by suffix. Omit to accept any file.
- `--quiet-seconds` — how long a file must stop changing before it's processed.
- `--poll-interval` — how often the stability check runs.
- `--recursive` — watch subdirectories too.
- `--log-level` — `DEBUG` shows every filesystem event.

## Plugging in a real denoiser

The shipped `PassthroughDenoiser` just copies the file. To wire in a real
model, subclass `autodenoise.Denoiser`:

```python
from pathlib import Path
from autodenoise import Denoiser

class CAREamicsDenoiser(Denoiser):
    def __init__(self, weights_path: Path):
        from careamics import CAREamist
        self.model = CAREamist.from_pretrained(weights_path)

    def denoise(self, src: Path, dst: Path) -> None:
        import tifffile
        img = tifffile.imread(src)
        out = self.model.predict(img)
        tifffile.imwrite(dst, out)
```

Then swap the instance in `run.py`:

```python
denoiser = CAREamicsDenoiser(Path("weights/n2v.ckpt"))
```

## Limits

- Stability is judged by size + mtime. A writer that pre-allocates the file
  and patches it in place won't trigger the size-change branch; in that case
  raise `--quiet-seconds`.
- Network filesystems sometimes drop watchdog events. Raise `--quiet-seconds`
  and treat the tool as a best-effort layer over a periodic full sweep.
- A startup scan of pre-existing files is intentionally **not** performed —
  only files that arrive after the watcher starts are processed.
