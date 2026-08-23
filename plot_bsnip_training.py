"""plot_bsnip_training.py

Plot BSNIP 3D CNN training curves from training_log_bsnip.csv (produced by
train_3dcnn.py), mirroring plot_training_log.py.

Two subplots:
    Left:  Train vs Validation Loss over epochs.
    Right: Train vs Validation AUC-ROC over epochs, with the best
           validation AUC epoch highlighted.

Usage:
    python plot_bsnip_training.py \\
        --log-csv training_log_bsnip.csv \\
        --output-png training_curves_bsnip.png
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from logging_utils import setup_logging

logger = logging.getLogger("plot_bsnip_training")

DEFAULT_LOG_CSV = Path("training_log_bsnip.csv")
DEFAULT_OUTPUT_PNG = Path("training_curves_bsnip.png")


def load_training_log(log_csv: Path) -> pd.DataFrame:
    """Read the per-epoch metrics CSV written by train_3dcnn.py."""
    logger.info("Reading training log %s", log_csv)
    df = pd.read_csv(log_csv)
    logger.info("Loaded %d epochs", len(df))
    return df


def plot_loss(ax: plt.Axes, df: pd.DataFrame, palette: Sequence[tuple]) -> None:
    """Left subplot: Train Loss vs Validation Loss over epochs."""
    ax.plot(df["epoch"], df["train_loss"], label="Train Loss", linewidth=2.2, color=palette[0])
    ax.plot(df["epoch"], df["val_loss"], label="Val Loss", linewidth=2.2, color=palette[1])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training vs Validation Loss")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)


def plot_auc(ax: plt.Axes, df: pd.DataFrame, palette: Sequence[tuple]) -> None:
    """Right subplot: Train AUC vs Validation AUC, best val AUC epoch highlighted."""
    ax.plot(df["epoch"], df["train_auc"], label="Train AUC", linewidth=2.2, color=palette[0])
    ax.plot(df["epoch"], df["val_auc"], label="Val AUC", linewidth=2.2, color=palette[1])

    best_idx = df["val_auc"].idxmax()
    best_epoch = df.loc[best_idx, "epoch"]
    best_val_auc = df.loc[best_idx, "val_auc"]

    ax.scatter(
        [best_epoch], [best_val_auc],
        color="crimson", zorder=5, s=80, edgecolor="white", linewidth=1.2,
        label=f"Best Val AUC (epoch {int(best_epoch)}: {best_val_auc:.3f})",
    )
    ax.axvline(best_epoch, color="crimson", linestyle="--", linewidth=1, alpha=0.5)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("AUC-ROC")
    ax.set_title("Training vs Validation AUC-ROC")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, loc="lower right")
    ax.grid(alpha=0.3)


def make_figure(df: pd.DataFrame) -> plt.Figure:
    """Build the two-subplot publication-quality training curves figure."""
    sns.set_theme(style="whitegrid", context="talk")
    palette = sns.color_palette("deep")

    fig, (ax_loss, ax_auc) = plt.subplots(1, 2, figsize=(14, 6))
    plot_loss(ax_loss, df, palette)
    plot_auc(ax_auc, df, palette)

    fig.suptitle("BSNIP 3D CNN Training Curves (HC vs SZ)", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot BSNIP training/validation loss and AUC-ROC curves from training_log_bsnip.csv.",
    )
    parser.add_argument("--log-csv", type=Path, default=DEFAULT_LOG_CSV,
                         help="Path to training_log_bsnip.csv (default: %(default)s)")
    parser.add_argument("--output-png", type=Path, default=DEFAULT_OUTPUT_PNG,
                         help="Output figure path (default: %(default)s)")
    parser.add_argument("--dpi", type=int, default=300, help="Output figure DPI (default: %(default)s)")
    parser.add_argument("--log-level", type=str, default="INFO",
                         choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                         help="Logging verbosity (default: %(default)s)")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    setup_logging(args.log_level)

    df = load_training_log(args.log_csv)
    fig = make_figure(df)
    fig.savefig(args.output_png, dpi=args.dpi, bbox_inches="tight")
    logger.info("Saved figure to %s", args.output_png)

    best_idx = df["val_auc"].idxmax()
    logger.info(
        "Best val_auc=%.4f at epoch %d", df.loc[best_idx, "val_auc"], int(df.loc[best_idx, "epoch"]),
    )


if __name__ == "__main__":
    main()
