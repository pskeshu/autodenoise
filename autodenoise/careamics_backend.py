"""CAREamics-based denoising backends.

Heavy imports (``careamics``, ``torch``, ``tifffile``) are deferred to
method bodies so the watcher pipeline starts without the ML stack present.
Importing this module on a Python install without ``careamics`` works;
constructing :class:`CAREamicsN2VDenoiser` is what triggers the import.

Optional dependency: install with

    pip install -r requirements-careamics.txt

Requires Python 3.11+ (a CAREamics constraint).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

from .denoiser import Denoiser

log = logging.getLogger(__name__)


class CAREamicsN2VDenoiser(Denoiser):
    """Self-supervised denoiser backed by CAREamics N2V / N2V2.

    Parameters
    ----------
    checkpoint:
        Path to a CAREamics-saved model. Both ``.ckpt`` checkpoints and
        ``.yaml`` configuration files are accepted (CAREamist auto-detects).
        Produced by ``train.py``.
    axes:
        Axis ordering of input arrays passed to ``predict``: ``"YX"`` for
        2D, ``"ZYX"`` for 3D, ``"SYX"`` for stacks of 2D images, etc.
        Should normally match the axes used at training time.
    tile_size:
        If set, prediction is tiled. Recommended for large 3D stacks; pick
        values that fit in GPU memory and divide the data cleanly.
    tile_overlap:
        Overlap between adjacent tiles. Defaults to half of ``tile_size``
        when only ``tile_size`` is given.
    """

    def __init__(
        self,
        checkpoint: Path,
        axes: str = "YX",
        tile_size: Optional[Sequence[int]] = None,
        tile_overlap: Optional[Sequence[int]] = None,
    ) -> None:
        from careamics import CAREamist
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info("CAREamics N2V denoiser loading on %s: %s", device, checkpoint)

        self._checkpoint = Path(checkpoint)
        self._axes = axes
        self._tile_size = list(tile_size) if tile_size is not None else None
        self._tile_overlap = list(tile_overlap) if tile_overlap is not None else None
        self._careamist = CAREamist(self._checkpoint)

    def denoise(self, src: Path, dst: Path) -> None:
        import numpy as np
        import tifffile

        log.info("CAREamics N2V denoising %s -> %s", src, dst)
        image = tifffile.imread(src)

        # data_type="array" overrides the training-time data_type stored in
        # the checkpoint config, since we feed predict() a numpy array.
        kwargs = {"source": image, "axes": self._axes, "data_type": "array"}
        if self._tile_size is not None:
            kwargs["tile_size"] = self._tile_size
            kwargs["tile_overlap"] = (
                self._tile_overlap
                if self._tile_overlap is not None
                else [s // 2 for s in self._tile_size]
            )

        prediction = self._careamist.predict(**kwargs)

        # CAREamist.predict returns a list of arrays for batched inputs and
        # an ndarray otherwise. Normalize to one array per call.
        if isinstance(prediction, list):
            prediction = (
                prediction[0] if len(prediction) == 1 else np.stack(prediction)
            )
        prediction = np.squeeze(prediction)

        # Match input dtype for integer types so downstream tools see the
        # same bit depth they wrote.
        if np.issubdtype(image.dtype, np.integer):
            info = np.iinfo(image.dtype)
            prediction = np.clip(prediction, info.min, info.max).astype(image.dtype)
        else:
            prediction = prediction.astype(np.float32, copy=False)

        tifffile.imwrite(dst, prediction)
