"""Denoising backends.

The pipeline is intentionally agnostic about *how* denoising happens. Subclass
:class:`Denoiser`, load weights / config in ``__init__``, and run inference in
``denoise``.
"""

from __future__ import annotations

import logging
import shutil
from abc import ABC, abstractmethod
from pathlib import Path

log = logging.getLogger(__name__)


class Denoiser(ABC):
    """Interface for denoising backends."""

    @abstractmethod
    def denoise(self, src: Path, dst: Path) -> None:
        """Read ``src``, write the denoised result to ``dst``."""


class PassthroughDenoiser(Denoiser):
    """Copies input to output unchanged. Lets the rest of the pipeline run
    end-to-end while a real backend is being wired in."""

    def denoise(self, src: Path, dst: Path) -> None:
        log.info("passthrough denoiser: %s -> %s", src, dst)
        shutil.copyfile(src, dst)
