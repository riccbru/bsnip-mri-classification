"""run_grid_search.py

2D grid search over (learning rate, weight decay) for the BSNIP 3D CNN
(GAP architecture), each configuration evaluated via 5-fold cross-
validation. Launches train_3dcnn.py once per configuration as a
subprocess (no torch import needed here), is safe to resume — a
configuration whose cv_summary.json already exists is skipped, not
re-run — and aggregates every completed configuration's Out-Of-Fold
results into a sorted CSV plus a comparison heatmap.

Grid: lr in [5e-5, 1e-4, 2e-4] x weight_decay in [1e-5, 1e-4, 1e-3]
      (9 configurations total, deterministic order: lr outer, wd inner)
Fixed: --cv-folds 5 --epochs 50 --patience 15 --batch-size 4
       --num-workers 4 --use-gap --augment --scheduler cosine
       --seed 42 --device cuda

Layout:
    runs/bsnip_3dcnn_v7_grid/
        lr_{lr}_wd_{wd}/                 one train_3dcnn.py --exp-name run
            cv_summary.json                  (per sklearn OOF aggregation)
            fold_1/ ... fold_5/
        grid_search_summary.csv          all completed configs, auc_mean desc
        grid_search_heatmap.png          Mean AUC-ROC + Mean Tuned Bal. Acc

Usage:
    python run_grid_search.py
    python run_grid_search.py --dry-run   # print commands/skip-status only
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence

import matplotlib
matplotlib.use("Agg")  # headless-safe (SLURM/cluster nodes have no display)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from logging_utils import setup_logging

logger = logging.getLogger("run_grid_search")

TRAIN_SCRIPT = Path("train_3dcnn.py")
DEFAULT_METADATA_CSV = Path("bsnip_preprocessed_npy_metadata.csv")
GRID_BASE_DIR = Path("runs/bsnip_3dcnn_v7_grid")
SUMMARY_CSV_FILENAME = "grid_search_summary.csv"
HEATMAP_FILENAME = "grid_search_heatmap.png"

GRID_LR: list[float] = [5e-5, 1e-4, 2e-4]
GRID_WEIGHT_DECAY: list[float] = [1e-5, 1e-4, 1e-3]

# Fixed hyperparameters shared by every configuration in the grid.
FIXED_VALUE_ARGS: dict[str, str] = {
    "--cv-folds": "5",
    "--epochs": "50",
    "--patience": "15",
    "--batch-size": "4",
    "--num-workers": "4",
    "--seed": "42",
    "--device": "cuda",
    "--scheduler": "cosine",
}
FIXED_FLAGS: list[str] = ["--use-gap", "--augment"]


def format_value(value: float) -> str:
    """1e-4 -> "1e-04", matching the lr_{lr}_wd_{wd} directory-name convention."""
    return f"{value:.0e}"


def run_name(lr: float, weight_decay: float) -> str:
    return f"lr_{format_value(lr)}_wd_{format_value(weight_decay)}"


def build_command(exp_name: str, lr: float, weight_decay: float, metadata_csv: Path) -> list[str]:
    cmd = [
        sys.executable, str(TRAIN_SCRIPT),
        "--exp-name", exp_name,
        "--metadata-csv", str(metadata_csv),
        "--lr", str(lr),
        "--weight-decay", str(weight_decay),
    ]
    for flag, value in FIXED_VALUE_ARGS.items():
        cmd.extend([flag, value])
    cmd.extend(FIXED_FLAGS)
    return cmd


def run_configuration(lr: float, weight_decay: float, metadata_csv: Path, dry_run: bool) -> Optional[Path]:
    """Run (or skip, if already done) one (lr, weight_decay) configuration.

    Returns the path to its cv_summary.json if that configuration is (or
    already was) complete, else None — a fresh run that's pending in
    dry-run mode, or one that failed / didn't produce a summary.
    """
    name = run_name(lr, weight_decay)
    exp_name = f"{GRID_BASE_DIR.name}/{name}"
    run_dir = GRID_BASE_DIR / name
    cv_summary_path = run_dir / "cv_summary.json"

    if cv_summary_path.exists():
        logger.info("Skipping %s: %s already exists (resumption)", name, cv_summary_path)
        return cv_summary_path

    cmd = build_command(exp_name, lr, weight_decay, metadata_csv)
    logger.info("Config %s: %s", name, " ".join(cmd))

    if dry_run:
        logger.info("[dry-run] not executing %s", name)
        return None

    result = subprocess.run(cmd)
    if result.returncode != 0:
        logger.error("%s failed (exit code %d); excluded from the summary", name, result.returncode)
        return None

    if not cv_summary_path.exists():
        logger.error("%s exited 0 but %s wasn't created; excluded from the summary", name, cv_summary_path)
        return None

    return cv_summary_path


def _get_mean_std(container: dict, key: str) -> tuple[float, float]:
    sub = container.get(key, {})
    return float(sub.get("mean", float("nan"))), float(sub.get("std", float("nan")))


def extract_oof_row(name: str, lr: float, weight_decay: float, cv_summary_path: Path) -> dict[str, object]:
    """Pull the OOF metrics this grid search reports out of one cv_summary.json."""
    with cv_summary_path.open() as f:
        cv_summary = json.load(f)
    oof = cv_summary["oof"]
    by_threshold = oof.get("by_threshold", {})
    default_m = by_threshold.get("Default (p=0.50)", {})
    tuned_m = by_threshold.get("Tuned (p*)")
    if tuned_m is None:
        logger.warning("%s: no tuned-threshold OOF metrics (was --tune-threshold disabled?)", name)
        tuned_m = {}

    auc_mean, auc_std = _get_mean_std(oof, "auc")
    tuned_bal_mean, tuned_bal_std = _get_mean_std(tuned_m, "balanced_acc")
    tuned_sz_recall_mean, tuned_sz_recall_std = _get_mean_std(tuned_m, "sz_recall")
    tuned_hc_recall_mean, tuned_hc_recall_std = _get_mean_std(tuned_m, "hc_recall")
    default_bal_mean, default_bal_std = _get_mean_std(default_m, "balanced_acc")

    return {
        "run_name": name,
        "lr": lr,
        "weight_decay": weight_decay,
        "n_folds": oof.get("n_folds"),
        "auc_mean": auc_mean, "auc_std": auc_std,
        "tuned_balanced_acc_mean": tuned_bal_mean, "tuned_balanced_acc_std": tuned_bal_std,
        "tuned_sz_recall_mean": tuned_sz_recall_mean, "tuned_sz_recall_std": tuned_sz_recall_std,
        "tuned_hc_recall_mean": tuned_hc_recall_mean, "tuned_hc_recall_std": tuned_hc_recall_std,
        "default_balanced_acc_mean": default_bal_mean, "default_balanced_acc_std": default_bal_std,
    }


def build_value_grid(
    df: pd.DataFrame, lr_values: Sequence[float], wd_values: Sequence[float], value_col: str,
) -> np.ndarray:
    """lr x weight_decay grid of `value_col`; NaN for any missing/failed configuration."""
    grid = np.full((len(lr_values), len(wd_values)), np.nan)
    for i, lr in enumerate(lr_values):
        for j, wd in enumerate(wd_values):
            match = df[np.isclose(df["lr"], lr) & np.isclose(df["weight_decay"], wd)]
            if not match.empty:
                grid[i, j] = match.iloc[0][value_col]
    return grid


def plot_heatmaps(
    df: pd.DataFrame, lr_values: Sequence[float], wd_values: Sequence[float], output_path: Path,
) -> None:
    """Side-by-side heatmaps: Mean AUC-ROC and Mean Tuned Balanced Accuracy, LR x Weight Decay."""
    specs = [
        ("auc_mean", "Mean AUC-ROC (OOF)"),
        ("tuned_balanced_acc_mean", "Mean Tuned Balanced Accuracy (OOF)"),
    ]
    lr_labels = [format_value(v) for v in lr_values]
    wd_labels = [format_value(v) for v in wd_values]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, (value_col, title) in zip(axes, specs):
        grid = build_value_grid(df, lr_values, wd_values, value_col)
        finite = grid[~np.isnan(grid)]
        vmin, vmax = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)

        im = ax.imshow(grid, cmap="viridis", aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(wd_values)))
        ax.set_xticklabels(wd_labels)
        ax.set_yticks(range(len(lr_values)))
        ax.set_yticklabels(lr_labels)
        ax.set_xlabel("Weight Decay")
        ax.set_ylabel("Learning Rate")
        ax.set_title(title)

        midpoint = (vmin + vmax) / 2
        for i in range(len(lr_values)):
            for j in range(len(wd_values)):
                val = grid[i, j]
                text = f"{val:.3f}" if not np.isnan(val) else "N/A"
                color = "white" if (not np.isnan(val) and val < midpoint) else "black"
                ax.text(j, i, text, ha="center", va="center", color=color, fontsize=10)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("BSNIP 3D CNN Grid Search — 5-Fold CV, GAP Architecture", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved grid search heatmap to %s", output_path)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="2D (learning rate x weight decay) grid search for the BSNIP 3D CNN via 5-fold CV.",
    )
    parser.add_argument("--metadata-csv", type=Path, default=DEFAULT_METADATA_CSV,
                         help="Passed through to train_3dcnn.py --metadata-csv (default: %(default)s)")
    parser.add_argument("--dry-run", action="store_true",
                         help="Print each configuration's command and skip/pending status; run nothing "
                              "and write no summary/heatmap files (default: %(default)s)")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                         help="Logging verbosity (default: %(default)s)")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    setup_logging(args.log_level)

    GRID_BASE_DIR.mkdir(parents=True, exist_ok=True)
    configs = list(itertools.product(GRID_LR, GRID_WEIGHT_DECAY))
    logger.info("Grid search: %d configurations (lr x weight_decay) -> %s", len(configs), GRID_BASE_DIR)

    rows: list[dict[str, object]] = []
    for lr, weight_decay in configs:
        name = run_name(lr, weight_decay)
        cv_summary_path = run_configuration(lr, weight_decay, args.metadata_csv, args.dry_run)
        if cv_summary_path is None:
            continue
        try:
            rows.append(extract_oof_row(name, lr, weight_decay, cv_summary_path))
        except (KeyError, json.JSONDecodeError) as exc:
            logger.error("Failed to parse %s: %s", cv_summary_path, exc)

    if args.dry_run:
        logger.info("Dry run complete; no summary/heatmap written.")
        return

    if not rows:
        logger.warning("No completed configurations found; nothing to summarize.")
        return

    df = pd.DataFrame(rows).sort_values("auc_mean", ascending=False).reset_index(drop=True)
    summary_csv_path = GRID_BASE_DIR / SUMMARY_CSV_FILENAME
    df.to_csv(summary_csv_path, index=False)
    logger.info("Saved grid search summary to %s", summary_csv_path)
    print("\n" + df.to_string(index=False))

    plot_heatmaps(df, GRID_LR, GRID_WEIGHT_DECAY, GRID_BASE_DIR / HEATMAP_FILENAME)

    best = df.iloc[0]
    logger.info(
        "Best configuration: %s (auc_mean=%.4f +/- %.4f)",
        best["run_name"], best["auc_mean"], best["auc_std"],
    )


if __name__ == "__main__":
    main()
