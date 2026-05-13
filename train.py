"""Train a CAREamics N2V2 model on a directory of TIFF files.

The trained model and logs land under ``--work-dir`` (default
``models/n2v2``). After training, point ``run.py`` at the best checkpoint:

    python train.py /path/to/training/tiffs --axes ZYX --epochs 30

    python run.py incoming/ outgoing/ \\
        --backend careamics-n2v \\
        --weights models/n2v2/checkpoints/<best>.ckpt \\
        --axes ZYX
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a CAREamics N2V2 model.")
    parser.add_argument(
        "input",
        type=Path,
        help="TIFF file or directory of TIFF stacks to train on.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("models") / "n2v2",
        help="Where to save checkpoints and logs. Default: models/n2v2.",
    )
    parser.add_argument(
        "--axes",
        default="YX",
        help="Axis ordering of input TIFFs (e.g. YX, ZYX, SYX). Default: YX.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs="+",
        default=None,
        help="Training patch size; length must match `axes` dim. "
        "Default: [64, 64] for 2D, [16, 64, 64] when 'Z' is in axes.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument(
        "--algorithm",
        choices=["n2v2", "n2v"],
        default="n2v2",
        help="Use N2V2 (default) or vanilla N2V.",
    )
    parser.add_argument(
        "--val-percentage",
        type=float,
        default=0.1,
        help="Fraction of training data held out for validation. Default: 0.1.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers. Default 0 (safe on Windows). Bump on Linux "
        "for faster I/O if the disk is the bottleneck.",
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

    # Heavy imports kept lazy.
    import torch
    from careamics import CAREamist
    from careamics.config import create_n2v_configuration

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info("Training device: %s", device)

    if args.patch_size is None:
        args.patch_size = [16, 64, 64] if "Z" in args.axes.upper() else [64, 64]

    args.work_dir.mkdir(parents=True, exist_ok=True)

    config = create_n2v_configuration(
        experiment_name=f"autodenoise_{args.algorithm}",
        data_type="tiff",
        axes=args.axes,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        use_n2v2=(args.algorithm == "n2v2"),
        train_dataloader_params={"num_workers": args.num_workers},
    )

    careamist = CAREamist(config, work_dir=args.work_dir)
    careamist.train(
        train_source=args.input,
        val_percentage=args.val_percentage,
    )

    ckpt_dir = args.work_dir / "checkpoints"
    ckpts = sorted(ckpt_dir.rglob("*.ckpt")) if ckpt_dir.exists() else []
    if ckpts:
        logging.info("Trained checkpoints under %s:", ckpt_dir)
        for c in ckpts:
            logging.info("  %s", c.relative_to(args.work_dir))
    else:
        logging.warning("No checkpoints found under %s.", ckpt_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
