"""train_3dcnn.py

Train the 3D CNN on BSNIP binary classification (HC vs SZ).

Mirrors the ADNI training scripts (train_3dcnn.py / model_3dcnn_gradcam.py),
adapted to:
    - bsnip_dataset.get_bsnip_dataloaders() for the stratified,
      subject-level 70/15/15 train/val/test split.
    - model_3dcnn.Simple3DCNN as the base architecture, subclassed here with
      a shape-adaptive fc1: unlike ADNI, BSNIP volumes are not resized to a
      fixed 128^3 (see preprocessing.py), so the base class's hardcoded
      32*16*16*16 flatten size would break on BSNIP's native volume shape.

Everything about a run is keyed off --exp-name: every output goes to
runs/<exp_name>/ (auto-created) with no other path flags to manage:
    - best_model.pth           best checkpoint (by --checkpoint-metric)
    - training_log.csv         per-epoch Loss/Accuracy/Balanced Acc/AUC-ROC
    - training_curves.png      Loss + AUC curves, plotted at the end
    - experiment_summary.json  hyperparams + best val metrics + test metrics

Four boolean switches (--augment, --weighted-loss, --use-gap all default
False; --tune-threshold defaults True) toggle: on-the-fly 3D train
augmentation, inverse-frequency class-weighted loss, a GAP head instead of
flatten, and val-tuned decision-threshold evaluation on top of the default
p=0.5. Each also accepts a --no-<flag> form (e.g. --no-tune-threshold).

Usage:
    python train_3dcnn.py --exp-name v3_gap --epochs 50 --batch-size 4 --lr 2e-5 \\
        --seed 42 --device cuda --augment --weighted-loss --use-gap
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import matplotlib
matplotlib.use("Agg")  # headless-safe (SLURM/cluster nodes have no display)
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.backends.cudnn.enabled = False
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from bsnip_dataset import LABEL_NAMES, get_bsnip_dataloaders
from model_3dcnn import Simple3DCNN

logger = logging.getLogger("train_3dcnn")

DEFAULT_METADATA_CSV = Path("bsnip_preprocessed_npy_metadata.csv")
RUNS_DIR = Path("runs")
BEST_MODEL_FILENAME = "best_model.pth"
LOG_CSV_FILENAME = "training_log.csv"
CURVES_FILENAME = "training_curves.png"
SUMMARY_FILENAME = "experiment_summary.json"

NUM_CLASSES = 2  # (HC, SZ) = (0, 1)
LOG_COLUMNS: list[str] = [
    "epoch",
    "train_loss", "train_acc", "train_balanced_acc", "train_auc",
    "val_loss", "val_acc", "val_balanced_acc", "val_auc",
]


class BSNIP3DCNN(Simple3DCNN):
    """Simple3DCNN with a shape-adaptive fc1, and an optional GAP head.

    ADNI volumes are resized to a fixed 128^3, so the base class's fc1 is
    hardcoded for that shape. BSNIP volumes keep their native preprocessed
    shape (see preprocessing.py), so fc1's input size is computed here from
    an actual `input_shape` via a dummy forward pass, then fc1 is replaced.

    Also overrides the base class's dropout (p=0.5) with p=0.4, applied
    between fc1 and fc2 by the inherited forward() — the base's dropout
    rate was too aggressive for BSNIP's smaller dataset and contributed to
    val_balanced_acc collapsing to 0.5 (majority-class-only predictions).

    If `use_gap=True`, conv3's output is global-average-pooled to
    (C, 1, 1, 1) before fc1 instead of flattened — far fewer fc1 parameters
    (32 vs. flatten's spatial-dim-dependent size) and less sensitivity to
    where in the volume a feature appears, at the cost of discarding
    positional information entirely. Off by default to preserve the
    existing (flatten-based) architecture and its trained checkpoints.
    """

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        num_classes: int = NUM_CLASSES,
        use_gap: bool = False,
    ) -> None:
        super().__init__(num_classes=num_classes)
        self.use_gap = use_gap

        if use_gap:
            self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))
            flat_dim = 32  # conv3's output channel count
        else:
            self.gap = None
            with torch.no_grad():
                dummy = torch.zeros(1, 1, *input_shape)
                x = self.pool1(F.relu(self.bn1(self.conv1(dummy))))
                x = self.pool2(F.relu(self.bn2(self.conv2(x))))
                x = self.pool3(F.relu(self.bn3(self.conv3(x))))
                flat_dim = x.view(1, -1).shape[1]

        self.fc1 = nn.Linear(flat_dim, 128)
        self.dropout = nn.Dropout(p=0.4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        if self.use_gap:
            x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


def set_seed(seed: int) -> None:
    """Seed all RNGs used for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def infer_input_shape(loader: DataLoader) -> tuple[int, int, int]:
    """Read the (D, H, W) shape of one sample from a DataLoader's dataset."""
    sample_img, _ = loader.dataset[0]
    return tuple(sample_img.shape[1:])  # drop channel dim: (1, D, H, W) -> (D, H, W)


def compute_class_weights(labels: Sequence[int], num_classes: int = NUM_CLASSES) -> torch.Tensor:
    """Inverse class-frequency weights for nn.CrossEntropyLoss(weight=...).

    weight[c] = n_samples / (n_classes * n_samples_in_class_c), the standard
    "balanced" weighting — rarer classes get proportionally larger weight.
    """
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=num_classes).astype(np.float64)
    counts = np.where(counts == 0, 1.0, counts)  # avoid div-by-zero if a class is absent
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def compute_metrics(y_true: Sequence[int], y_pred: Sequence[int], y_prob: Sequence[float]) -> dict[str, float]:
    """Accuracy, balanced accuracy, and AUC-ROC (prob. of the SZ=1 class)."""
    metrics = {
        "acc": accuracy_score(y_true, y_pred),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
    }
    try:
        metrics["auc"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        # Only one class present in y_true (can happen on a tiny/unlucky split).
        logger.warning("AUC-ROC undefined for this epoch (single class present); logging NaN")
        metrics["auc"] = float("nan")
    return metrics


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> tuple[float, dict[str, float]]:
    """Run one train (optimizer given) or eval (optimizer=None) epoch."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    y_true: list[int] = []
    y_pred: list[int] = []
    y_prob: list[float] = []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            if is_train:
                optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)

            y_true.extend(labels.detach().cpu().tolist())
            y_pred.extend(preds.detach().cpu().tolist())
            y_prob.extend(probs.detach().cpu().tolist())

    avg_loss = total_loss / len(loader)
    metrics = compute_metrics(y_true, y_pred, y_prob)
    return avg_loss, metrics


def is_better(current: float, best: float, metric: str) -> bool:
    """Whether `current` improves on `best` for the given checkpoint metric."""
    if metric == "val_auc":
        return current > best
    return current < best  # val_loss: lower is better


def init_log_csv(log_csv: Path) -> None:
    with log_csv.open("w", newline="") as f:
        csv.writer(f).writerow(LOG_COLUMNS)


def append_log_row(log_csv: Path, row: dict[str, float]) -> None:
    with log_csv.open("a", newline="") as f:
        csv.writer(f).writerow([row[col] for col in LOG_COLUMNS])


def make_and_save_curves(history: list[dict[str, float]], output_path: Path) -> None:
    """Pure-matplotlib Loss + AUC-ROC training curves, saved to output_path.

    Left: Train vs Val Loss. Right: Train vs Val AUC, with the best-val-AUC
    epoch marked. Self-contained (no seaborn) so it has no extra dependency
    beyond what training itself already needs, and runs headless via the
    Agg backend set at module import time (SLURM/cluster-safe).
    """
    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["val_loss"] for row in history]
    train_auc = [row["train_auc"] for row in history]
    val_auc = [row["val_auc"] for row in history]

    fig, (ax_loss, ax_auc) = plt.subplots(1, 2, figsize=(14, 6))

    ax_loss.plot(epochs, train_loss, label="Train Loss", linewidth=2, color="tab:blue")
    ax_loss.plot(epochs, val_loss, label="Val Loss", linewidth=2, color="tab:orange")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Training vs Validation Loss")
    ax_loss.legend(frameon=False)
    ax_loss.grid(alpha=0.3)

    ax_auc.plot(epochs, train_auc, label="Train AUC", linewidth=2, color="tab:blue")
    ax_auc.plot(epochs, val_auc, label="Val AUC", linewidth=2, color="tab:orange")
    try:
        best_idx = int(np.nanargmax(val_auc))
        ax_auc.scatter(
            [epochs[best_idx]], [val_auc[best_idx]],
            color="crimson", zorder=5, s=80, edgecolor="white", linewidth=1.2,
            label=f"Best Val AUC (epoch {epochs[best_idx]}: {val_auc[best_idx]:.3f})",
        )
        ax_auc.axvline(epochs[best_idx], color="crimson", linestyle="--", linewidth=1, alpha=0.5)
    except ValueError:
        logger.warning("Could not determine best-val-AUC epoch for plotting (all NaN)")

    ax_auc.set_xlabel("Epoch")
    ax_auc.set_ylabel("AUC-ROC")
    ax_auc.set_title("Training vs Validation AUC-ROC")
    ax_auc.set_ylim(0, 1.05)
    ax_auc.legend(frameon=False, loc="lower right")
    ax_auc.grid(alpha=0.3)

    fig.suptitle("BSNIP 3D CNN Training Curves (HC vs SZ)", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved training curves to %s", output_path)


def collect_probs(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[list[int], list[float]]:
    """Collect true labels and predicted P(SZ) probabilities for every sample in `loader`."""
    model.eval()
    y_true: list[int] = []
    y_prob: list[float] = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)[:, 1]
            y_true.extend(labels.tolist())
            y_prob.extend(probs.cpu().tolist())
    return y_true, y_prob


def find_optimal_threshold(
    y_true: Sequence[int], y_prob: Sequence[float], metric: str = "balanced_accuracy",
) -> tuple[float, float]:
    """Grid-search over [0.1, 0.9] for the decision threshold maximizing `metric` on (y_true, y_prob)."""
    def score_fn(yt: Sequence[int], preds: np.ndarray) -> float:
        if metric == "macro_f1":
            return f1_score(yt, preds, average="macro", zero_division=0)
        return balanced_accuracy_score(yt, preds)

    y_prob_arr = np.asarray(y_prob)
    best_threshold, best_score = 0.5, -1.0
    for threshold in np.arange(0.1, 0.91, 0.01):
        preds = (y_prob_arr >= threshold).astype(int)
        score = score_fn(y_true, preds)
        if score > best_score:
            best_score, best_threshold = score, float(threshold)
    return best_threshold, best_score


def evaluate_at_threshold(y_true: Sequence[int], y_prob: Sequence[float], threshold: float) -> dict[str, float]:
    """Precision/recall/F1 (macro), balanced accuracy, and overall accuracy at a threshold."""
    preds = (np.asarray(y_prob) >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, preds, average="macro", zero_division=0,
    )
    return {
        "threshold": float(threshold),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "balanced_acc": float(balanced_accuracy_score(y_true, preds)),
        "acc": float(accuracy_score(y_true, preds)),
    }


def print_comparison_table(results: dict[str, dict[str, float]]) -> None:
    """Print a compact Default (p=0.5) vs Tuned (p*) metrics table to stdout."""
    header = f"{'Threshold':<22}{'Precision':>10}{'Recall':>10}{'F1':>10}{'Balanced Acc':>14}{'Overall Acc':>13}"
    width = len(header)
    print("\n" + "=" * width)
    print("Test Set Threshold Comparison")
    print("=" * width)
    print(header)
    print("-" * width)
    for label, m in results.items():
        print(
            f"{label:<22}{m['precision_macro']:>10.4f}{m['recall_macro']:>10.4f}{m['f1_macro']:>10.4f}"
            f"{m['balanced_acc']:>14.4f}{m['acc']:>13.4f}"
        )
    print("=" * width + "\n")


def evaluate_test_set(
    model: nn.Module,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    tune_threshold: bool = True,
    threshold_metric: str = "balanced_accuracy",
) -> dict[str, object]:
    """Evaluate test at p=0.5, and (if tune_threshold) also at a val-tuned p*.

    Logs a per-class classification_report for each threshold evaluated,
    prints the compact comparison table, and returns a JSON-able summary
    dict for experiment_summary.json.
    """
    test_true, test_prob = collect_probs(model, test_loader, device)

    thresholds: list[tuple[float, str]] = [(0.5, "Default (p=0.50)")]
    optimal_threshold: Optional[float] = None
    optimal_threshold_val_score: Optional[float] = None

    if tune_threshold:
        val_true, val_prob = collect_probs(model, val_loader, device)
        optimal_threshold, optimal_threshold_val_score = find_optimal_threshold(
            val_true, val_prob, metric=threshold_metric,
        )
        logger.info(
            "Optimal threshold p*=%.2f (val %s=%.4f)",
            optimal_threshold, threshold_metric, optimal_threshold_val_score,
        )
        thresholds.append((optimal_threshold, f"Tuned (p*={optimal_threshold:.2f})"))

    results: dict[str, dict[str, float]] = {}
    for threshold, label in thresholds:
        metrics = evaluate_at_threshold(test_true, test_prob, threshold)
        results[label] = metrics

        preds = (np.asarray(test_prob) >= threshold).astype(int)
        target_names = [LABEL_NAMES[i] for i in sorted(set(test_true) | set(preds.tolist()))]
        logger.info(
            "Test set @ %s -> acc=%.4f balanced_acc=%.4f macro_f1=%.4f\n%s",
            label, metrics["acc"], metrics["balanced_acc"], metrics["f1_macro"],
            classification_report(test_true, preds, target_names=target_names),
        )

    print_comparison_table(results)

    return {
        "tune_threshold": tune_threshold,
        "threshold_metric": threshold_metric if tune_threshold else None,
        "optimal_threshold": optimal_threshold,
        "optimal_threshold_val_score": optimal_threshold_val_score,
        "metrics_by_threshold": results,
    }


def _json_safe(value: object) -> object:
    """Coerce argparse.Namespace values (Path, etc.) into JSON-serializable types."""
    if isinstance(value, Path):
        return str(value)
    return value


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a 3D CNN on BSNIP HC vs SZ classification.")
    parser.add_argument("--exp-name", type=str, default=None,
                         help="Experiment name; all outputs go to runs/<exp_name>/. "
                              "Defaults to a timestamp (run_YYYYmmdd_HHMMSS) if omitted.")
    parser.add_argument("--metadata-csv", type=Path, default=DEFAULT_METADATA_CSV,
                         help="Path to bsnip_preprocessed_npy_metadata.csv (default: %(default)s)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs (default: %(default)s)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size (default: %(default)s)")
    parser.add_argument("--lr", type=float, default=2e-5, help="Adam learning rate (default: %(default)s)")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker count (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed, also used for the data split (default: %(default)s)")
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"],
                         help="Device to train on (default: cuda if available, else cpu)")
    parser.add_argument("--checkpoint-metric", type=str, default="val_auc", choices=["val_auc", "val_loss"],
                         help="Metric used to select the best checkpoint (default: %(default)s)")
    parser.add_argument("--threshold-metric", type=str, default="balanced_accuracy",
                         choices=["balanced_accuracy", "macro_f1"],
                         help="Metric to maximize when tuning the test-set decision threshold on val "
                              "(default: %(default)s)")

    # Feature switches. Each also accepts a --no-<flag> form.
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=False,
                         help="Enable on-the-fly 3D train augmentation (flip, translation, noise) (default: %(default)s)")
    parser.add_argument("--weighted-loss", action=argparse.BooleanOptionalAction, default=False,
                         help="Use inverse-frequency class weights in CrossEntropyLoss (default: %(default)s)")
    parser.add_argument("--use-gap", action=argparse.BooleanOptionalAction, default=False,
                         help="Use global average pooling before fc1 instead of flattening conv3's "
                              "output (default: %(default)s)")
    parser.add_argument("--tune-threshold", action=argparse.BooleanOptionalAction, default=True,
                         help="Grid-search a val-tuned decision threshold p* and evaluate test at both "
                              "0.5 and p* (default: %(default)s)")

    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                         help="Logging verbosity (default: %(default)s)")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s [%(levelname)s] %(message)s")

    exp_name = args.exp_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    exp_dir = RUNS_DIR / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = exp_dir / BEST_MODEL_FILENAME
    log_csv_path = exp_dir / LOG_CSV_FILENAME
    curves_path = exp_dir / CURVES_FILENAME
    summary_path = exp_dir / SUMMARY_FILENAME
    logger.info("Experiment '%s' -> outputs in %s", exp_name, exp_dir)

    set_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info("Using device: %s", device)

    train_loader, val_loader, test_loader = get_bsnip_dataloaders(
        metadata_csv=args.metadata_csv,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        random_state=args.seed,
        augment=args.augment,
    )
    logger.info(
        "Split sizes -> train: %d, val: %d, test: %d",
        len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset),
    )
    logger.info("Train augmentation: %s", args.augment)

    input_shape = infer_input_shape(train_loader)
    logger.info("Inferred volume input shape (D, H, W): %s", input_shape)

    if args.weighted_loss:
        train_labels = train_loader.dataset.data["label"].tolist()
        class_weights = compute_class_weights(train_labels, NUM_CLASSES)
        logger.info(
            "Weighted loss enabled — class weights (inverse frequency): %s",
            {LABEL_NAMES[i]: round(w, 4) for i, w in enumerate(class_weights.tolist())},
        )
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        logger.info("Weighted loss disabled — using unweighted CrossEntropyLoss")
        criterion = nn.CrossEntropyLoss()

    model = BSNIP3DCNN(input_shape=input_shape, num_classes=NUM_CLASSES, use_gap=args.use_gap).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    init_log_csv(log_csv_path)
    history: list[dict[str, float]] = []
    best_metric = float("-inf") if args.checkpoint_metric == "val_auc" else float("inf")
    best_val_summary: dict[str, float] = {}

    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_metrics = run_epoch(model, val_loader, criterion, device, optimizer=None)

        row = {
            "epoch": epoch,
            "train_loss": train_loss, "train_acc": train_metrics["acc"],
            "train_balanced_acc": train_metrics["balanced_acc"], "train_auc": train_metrics["auc"],
            "val_loss": val_loss, "val_acc": val_metrics["acc"],
            "val_balanced_acc": val_metrics["balanced_acc"], "val_auc": val_metrics["auc"],
        }
        append_log_row(log_csv_path, row)
        history.append(row)

        logger.info(
            "Epoch [%d/%d] train_loss=%.4f train_acc=%.4f train_bal_acc=%.4f train_auc=%.4f | "
            "val_loss=%.4f val_acc=%.4f val_bal_acc=%.4f val_auc=%.4f",
            epoch, args.epochs, train_loss, train_metrics["acc"], train_metrics["balanced_acc"], train_metrics["auc"],
            val_loss, val_metrics["acc"], val_metrics["balanced_acc"], val_metrics["auc"],
        )

        if not np.isnan(val_metrics["auc"]):
            scheduler.step(val_metrics["auc"])
        else:
            logger.warning("Skipping LR scheduler step: val_auc is NaN this epoch")

        current_metric = val_metrics["auc"] if args.checkpoint_metric == "val_auc" else val_loss
        if not np.isnan(current_metric) and is_better(current_metric, best_metric, args.checkpoint_metric):
            best_metric = current_metric
            best_val_summary = {"epoch": epoch, "val_loss": val_loss, **val_metrics}
            torch.save(model.state_dict(), checkpoint_path)
            logger.info("New best checkpoint (%s=%.4f) saved to %s", args.checkpoint_metric, best_metric, checkpoint_path)

    logger.info("Training complete. Best %s: %.4f", args.checkpoint_metric, best_metric)

    make_and_save_curves(history, curves_path)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    test_summary = evaluate_test_set(
        model, val_loader, test_loader, device,
        tune_threshold=args.tune_threshold, threshold_metric=args.threshold_metric,
    )

    summary = {
        "exp_name": exp_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "hyperparameters": {k: _json_safe(v) for k, v in vars(args).items()},
        "best_val": best_val_summary,
        "test": test_summary,
    }
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved experiment summary to %s", summary_path)


if __name__ == "__main__":
    main()
