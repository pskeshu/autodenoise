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

`autodenoise` ships with a CAREamics-based **N2V2** backend (self-supervised,
no clean ground truth required). It is opt-in — install only if you want it,
otherwise the watcher runs on the lean `watchdog`-only base.

```bash
pip install -r requirements-careamics.txt   # adds careamics + tifffile, Python 3.11+
```

### 1. Train on representative noisy data

`train.py` accepts a TIFF file or a directory of TIFFs. For 3D stacks pass
`--axes ZYX`; checkpoints land under `models/n2v2/checkpoints/`.

```bash
python train.py /path/to/training/tiffs --axes ZYX --epochs 30
```

### 2. Run the watcher with the trained model

```bash
python run.py /path/to/incoming /path/to/denoised \
    --backend careamics-n2v \
    --weights models/n2v2/checkpoints/<best>.ckpt \
    --axes ZYX \
    --tile-size 16 256 256 \
    --quiet-seconds 5
```

GPU is used automatically when available (logged at startup); CPU works
too, just slower.

### Adding your own backend

Subclass `autodenoise.Denoiser`, do heavy imports in `__init__` or
`denoise`, then add a branch to `_build_denoiser` in `run.py`:

```python
from pathlib import Path
from autodenoise import Denoiser

class MyDenoiser(Denoiser):
    def __init__(self, weights: Path):
        import my_lib
        self.model = my_lib.load(weights)

    def denoise(self, src: Path, dst: Path) -> None:
        import tifffile
        img = tifffile.imread(src)
        tifffile.imwrite(dst, self.model.predict(img))
```

## Limits

- Stability is judged by size + mtime. A writer that pre-allocates the file
  and patches it in place won't trigger the size-change branch; in that case
  raise `--quiet-seconds`.
- Network filesystems sometimes drop watchdog events. Raise `--quiet-seconds`
  and treat the tool as a best-effort layer over a periodic full sweep.
- A startup scan of pre-existing files is intentionally **not** performed —
  only files that arrive after the watcher starts are processed.
