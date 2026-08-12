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

Per epoch, both train and val: Loss, Accuracy, Balanced Accuracy, AUC-ROC
(probability of the SZ=1 class) are computed and logged to
training_log_bsnip.csv. The best checkpoint (by val AUC-ROC or val loss,
--checkpoint-metric) is saved to best_bsnip_3dcnn.pth. After training, the
best checkpoint is reloaded and evaluated once on the held-out test split.

Usage:
    python train_3dcnn.py --epochs 50 --batch-size 8 --lr 1e-4 --seed 42 --device cuda
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.backends.cudnn.enabled = False
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, roc_auc_score
from torch.utils.data import DataLoader

from bsnip_dataset import LABEL_NAMES, get_bsnip_dataloaders
from model_3dcnn import Simple3DCNN

logger = logging.getLogger("train_3dcnn")

DEFAULT_METADATA_CSV = Path("bsnip_preprocessed_npy_metadata.csv")
DEFAULT_CHECKPOINT_PATH = Path("best_bsnip_3dcnn.pth")
DEFAULT_LOG_CSV = Path("training_log_bsnip.csv")
NUM_CLASSES = 2  # 0 = HC, 1 = SZ
LOG_COLUMNS: list[str] = [
    "epoch",
    "train_loss", "train_acc", "train_balanced_acc", "train_auc",
    "val_loss", "val_acc", "val_balanced_acc", "val_auc",
]


class BSNIP3DCNN(Simple3DCNN):
    """Simple3DCNN with a shape-adaptive fc1.

    ADNI volumes are resized to a fixed 128^3, so the base class's fc1 is
    hardcoded for that shape. BSNIP volumes keep their native preprocessed
    shape (see preprocessing.py), so fc1's input size is computed here from
    an actual `input_shape` via a dummy forward pass, then fc1 is replaced.
    """

    def __init__(self, input_shape: tuple[int, int, int], num_classes: int = NUM_CLASSES) -> None:
        super().__init__(num_classes=num_classes)
        with torch.no_grad():
            dummy = torch.zeros(1, 1, *input_shape)
            x = self.pool1(F.relu(self.bn1(self.conv1(dummy))))
            x = self.pool2(F.relu(self.bn2(self.conv2(x))))
            x = self.pool3(F.relu(self.bn3(self.conv3(x))))
            flat_dim = x.view(1, -1).shape[1]
        self.fc1 = nn.Linear(flat_dim, 128)


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


def evaluate_test_set(model: nn.Module, test_loader: DataLoader, device: torch.device) -> None:
    """Final evaluation of the best checkpoint on the held-out test split."""
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            y_true.extend(labels.tolist())
            y_pred.extend(preds.cpu().tolist())

    target_names = [LABEL_NAMES[i] for i in sorted(set(y_true) | set(y_pred))]
    logger.info("Test set classification report:\n%s", classification_report(y_true, y_pred, target_names=target_names))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a 3D CNN on BSNIP HC vs SZ classification.")
    parser.add_argument("--metadata-csv", type=Path, default=DEFAULT_METADATA_CSV,
                         help="Path to bsnip_preprocessed_npy_metadata.csv (default: %(default)s)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs (default: %(default)s)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (default: %(default)s)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Adam learning rate (default: %(default)s)")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker count (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed, also used for the data split (default: %(default)s)")
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"],
                         help="Device to train on (default: cuda if available, else cpu)")
    parser.add_argument("--checkpoint-metric", type=str, default="val_auc", choices=["val_auc", "val_loss"],
                         help="Metric used to select the best checkpoint (default: %(default)s)")
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT_PATH,
                         help="Where to save the best model checkpoint (default: %(default)s)")
    parser.add_argument("--log-csv", type=Path, default=DEFAULT_LOG_CSV,
                         help="Where to write per-epoch metrics (default: %(default)s)")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                         help="Logging verbosity (default: %(default)s)")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s [%(levelname)s] %(message)s")

    set_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info("Using device: %s", device)

    train_loader, val_loader, test_loader = get_bsnip_dataloaders(
        metadata_csv=args.metadata_csv,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        random_state=args.seed,
    )
    logger.info(
        "Split sizes -> train: %d, val: %d, test: %d",
        len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset),
    )

    input_shape = infer_input_shape(train_loader)
    logger.info("Inferred volume input shape (D, H, W): %s", input_shape)

    model = BSNIP3DCNN(input_shape=input_shape, num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    init_log_csv(args.log_csv)
    best_metric = float("-inf") if args.checkpoint_metric == "val_auc" else float("inf")

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
        append_log_row(args.log_csv, row)

        logger.info(
            "Epoch [%d/%d] train_loss=%.4f train_acc=%.4f train_bal_acc=%.4f train_auc=%.4f | "
            "val_loss=%.4f val_acc=%.4f val_bal_acc=%.4f val_auc=%.4f",
            epoch, args.epochs, train_loss, train_metrics["acc"], train_metrics["balanced_acc"], train_metrics["auc"],
            val_loss, val_metrics["acc"], val_metrics["balanced_acc"], val_metrics["auc"],
        )

        current_metric = val_metrics["auc"] if args.checkpoint_metric == "val_auc" else val_loss
        if not np.isnan(current_metric) and is_better(current_metric, best_metric, args.checkpoint_metric):
            best_metric = current_metric
            torch.save(model.state_dict(), args.checkpoint_path)
            logger.info("New best checkpoint (%s=%.4f) saved to %s", args.checkpoint_metric, best_metric, args.checkpoint_path)

    logger.info("Training complete. Best %s: %.4f", args.checkpoint_metric, best_metric)

    model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    evaluate_test_set(model, test_loader, device)


if __name__ == "__main__":
    main()
