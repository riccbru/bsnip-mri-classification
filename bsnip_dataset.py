"""bsnip_dataset.py

PyTorch Dataset and DataLoader factory for the BSNIP HC vs SZ classification
task. Mirrors adni_dataset.py, adapted to bsnip_preprocessed_npy_metadata.csv
(columns: subject_id, site, label, nii_path, npy_path) with a stratified,
subject-level 70/15/15 train/val/test split.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

DEFAULT_METADATA_CSV = Path("bsnip_preprocessed_npy_metadata.csv")
LABEL_NAMES: dict[int, str] = {0: "HC", 1: "SZ"}


class BSNIPDataset(Dataset):
    """PyTorch Dataset over preprocessed BSNIP Grey Matter volumes (.npy).

    Each item is a (1, D, H, W) float32 tensor paired with an int64
    (torch.long) label: 0 = Healthy Control (HC), 1 = Schizophrenia (SZ).
    """

    def __init__(
        self,
        metadata: Union[str, Path, pd.DataFrame],
        subject_ids: Optional[Sequence[str]] = None,
    ) -> None:
        """
        Args:
            metadata: Path to bsnip_preprocessed_npy_metadata.csv, or an
                already-loaded DataFrame with subject_id, site, label,
                nii_path, npy_path columns.
            subject_ids: If given, restrict the dataset to these subject_id
                values (e.g. one side of a subject-level train/val/test
                split). If None, the full table is used.
        """
        if isinstance(metadata, (str, Path)):
            self.data = pd.read_csv(metadata)
        else:
            self.data = metadata.copy()

        if subject_ids is not None:
            self.data = self.data[self.data["subject_id"].isin(subject_ids)]

        self.data = self.data.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.data.iloc[idx]
        img = np.load(row["npy_path"]).astype(np.float32)
        img = np.expand_dims(img, axis=0)  # channel-first: (1, D, H, W)

        tensor = torch.from_numpy(img).float()
        label = torch.tensor(int(row["label"]), dtype=torch.long)
        return tensor, label


def _split_subject_ids(
    subjects: pd.DataFrame,
    val_size: float,
    test_size: float,
    random_state: int,
) -> tuple[list[str], list[str], list[str]]:
    """Stratified split of unique subject_ids into train/val/test id lists."""
    trainval_ids, test_ids = train_test_split(
        subjects["subject_id"],
        test_size=test_size,
        stratify=subjects["label"],
        random_state=random_state,
    )

    trainval_labels = subjects.set_index("subject_id").loc[trainval_ids, "label"]
    relative_val_size = val_size / (1.0 - test_size)
    train_ids, val_ids = train_test_split(
        trainval_ids,
        test_size=relative_val_size,
        stratify=trainval_labels,
        random_state=random_state,
    )

    return list(train_ids), list(val_ids), list(test_ids)


def get_bsnip_dataloaders(
    metadata_csv: Union[str, Path] = DEFAULT_METADATA_CSV,
    batch_size: int = 8,
    num_workers: int = 4,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build stratified, subject-level train/val/test DataLoaders for BSNIP.

    Splits unique subject_ids (not rows) 70/15/15 by default, stratified on
    `label`, via sklearn.model_selection.train_test_split with a fixed
    random_state — so a given subject never appears in more than one split.

    Args:
        metadata_csv: Path to bsnip_preprocessed_npy_metadata.csv.
        batch_size: DataLoader batch size (same for all three splits).
        num_workers: DataLoader worker count (same for all three splits).
        val_size: Fraction of subjects held out for validation.
        test_size: Fraction of subjects held out for test.
        random_state: Seed for reproducible splits.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    df = pd.read_csv(metadata_csv)
    subjects = df[["subject_id", "label"]].drop_duplicates(subset="subject_id").reset_index(drop=True)

    train_ids, val_ids, test_ids = _split_subject_ids(subjects, val_size, test_size, random_state)

    train_dataset = BSNIPDataset(df, subject_ids=train_ids)
    val_dataset = BSNIPDataset(df, subject_ids=val_ids)
    test_dataset = BSNIPDataset(df, subject_ids=test_ids)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


def _label_distribution(labels: Sequence[int]) -> dict[str, int]:
    """Map a sequence of int labels to a {class_name: count} dict."""
    counts = Counter(int(label) for label in labels)
    return {LABEL_NAMES.get(label, str(label)): count for label, count in sorted(counts.items())}


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_bsnip_dataloaders()

    print(f"Train subjects: {len(train_loader.dataset)} | {_label_distribution(train_loader.dataset.data['label'])}")
    print(f"Val subjects:   {len(val_loader.dataset)} | {_label_distribution(val_loader.dataset.data['label'])}")
    print(f"Test subjects:  {len(test_loader.dataset)} | {_label_distribution(test_loader.dataset.data['label'])}")

    images, labels = next(iter(train_loader))
    print(f"\nBatch images shape: {tuple(images.shape)} (dtype={images.dtype})")
    print(f"Batch labels shape: {tuple(labels.shape)} (dtype={labels.dtype})")
    print(f"Batch label distribution: {_label_distribution(labels.tolist())}")
