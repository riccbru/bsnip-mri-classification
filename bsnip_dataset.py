"""bsnip_dataset.py

PyTorch Dataset and DataLoader factory for the BSNIP HC vs SZ classification
task. Mirrors adni_dataset.py, adapted to bsnip_preprocessed_npy_metadata.csv
(columns: subject_id, site, label, nii_path, npy_path).

Two ways to get data:
    - get_bsnip_dataloaders(): a single stratified, subject-level 70/15/15
      train/val/test split (the default, backward-compatible path).
    - get_bsnip_cv_folds(): stratified K-fold cross-validation. Each fold's
      outer test portion is held out entirely for that fold; the remaining
      K-1/K "trainval" pool is further split 80/20 into Train/Inner-Val.

Note: this metadata schema only has a numeric `label` column (0=HC, 1=SZ),
not a `diagnosis` string column — the composite CV stratum below is built
from `label` (as a string) + `site`, not `diagnosis` + `site`.
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
from scipy import ndimage
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger("bsnip_dataset")

DEFAULT_METADATA_CSV = Path("bsnip_preprocessed_npy_metadata.csv")
LABEL_NAMES: dict[int, str] = {0: "HC", 1: "SZ"}
DEFAULT_INNER_VAL_SIZE = 0.2

AUGMENT_FLIP_PROB = 0.5
AUGMENT_MAX_SHIFT_VOXELS = 2
AUGMENT_NOISE_SIGMA = 0.02


def _augment_volume(img: np.ndarray) -> np.ndarray:
    """On-the-fly 3D augmentation applied only to the training split.

    - Random sagittal (left-right) flip, p=0.5. Assumes axis 0 = sagittal,
      the same (X, Y, Z) SPM/MNI convention documented in gradcam_bsnip.py's
      AXIS_VIEW_MAP for these (121, 145, 121)-shaped volumes.
    - Random +/-2 voxel translation per axis (order=1 interpolation,
      "nearest" edge padding to avoid introducing zero-value seams).
    - Additive Gaussian noise, N(0, 0.02), then re-clipped to [0, 1] since
      the volumes are min-max normalized there.

    Note: uses numpy's global RNG. get_bsnip_dataloaders passes
    worker_init_fn=_seed_worker to each DataLoader so multi-worker runs get
    distinct, epoch-varying augmentation instead of correlated/repeated
    transforms (a standard PyTorch multiprocessing gotcha — see
    _seed_worker).
    """
    if np.random.rand() < AUGMENT_FLIP_PROB:
        img = np.flip(img, axis=0)

    shift = np.random.randint(-AUGMENT_MAX_SHIFT_VOXELS, AUGMENT_MAX_SHIFT_VOXELS + 1, size=3)
    img = ndimage.shift(img, shift=shift, order=1, mode="nearest")

    img = img + np.random.normal(0.0, AUGMENT_NOISE_SIGMA, size=img.shape).astype(np.float32)
    img = np.clip(img, 0.0, 1.0)

    return np.ascontiguousarray(img, dtype=np.float32)


class BSNIPDataset(Dataset):
    """PyTorch Dataset over preprocessed BSNIP Grey Matter volumes (.npy).

    Each item is a (1, D, H, W) float32 tensor paired with an int64
    (torch.long) label: 0 = Healthy Control (HC), 1 = Schizophrenia (SZ).
    """

    def __init__(
        self,
        metadata: Union[str, Path, pd.DataFrame],
        subject_ids: Optional[Sequence[str]] = None,
        is_train: bool = False,
    ) -> None:
        """
        Args:
            metadata: Path to bsnip_preprocessed_npy_metadata.csv, or an
                already-loaded DataFrame with subject_id, site, label,
                nii_path, npy_path columns.
            subject_ids: If given, restrict the dataset to these subject_id
                values (e.g. one side of a subject-level train/val/test
                split). If None, the full table is used.
            is_train: If True, apply on-the-fly 3D augmentation (random
                sagittal flip, +/-2 voxel translation, additive Gaussian
                noise) in __getitem__. Leave False for validation/test
                splits so evaluation always runs on unaugmented volumes.
        """
        if isinstance(metadata, (str, Path)):
            self.data = pd.read_csv(metadata)
        else:
            self.data = metadata.copy()

        if subject_ids is not None:
            self.data = self.data[self.data["subject_id"].isin(subject_ids)]

        self.data = self.data.reset_index(drop=True)
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.data.iloc[idx]
        img = np.load(row["npy_path"]).astype(np.float32)
        if self.is_train:
            img = _augment_volume(img)
        img = np.expand_dims(img, axis=0)  # channel-first: (1, D, H, W)

        tensor = torch.from_numpy(img).float()
        label = torch.tensor(int(row["label"]), dtype=torch.long)
        return tensor, label


def _seed_worker(worker_id: int) -> None:
    """DataLoader worker_init_fn: seed numpy's global RNG per worker, per epoch.

    PyTorch reseeds its own RNG per worker automatically, but not numpy's
    global RNG that _augment_volume uses — without this, forked worker
    processes can start from identical/stale RNG state and produce
    correlated or repeated "random" augmentations across workers and
    epochs. See https://pytorch.org/docs/stable/notes/randomness.html
    """
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)


def _load_subjects_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per subject_id, keeping label and site for stratification."""
    return df[["subject_id", "label", "site"]].drop_duplicates(subset="subject_id").reset_index(drop=True)


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


def _build_composite_strata(subjects: pd.DataFrame) -> pd.Series:
    """label__site composite stratum, e.g. "1_Chicago", for StratifiedKFold."""
    return subjects["label"].astype(str) + "_" + subjects["site"].astype(str)


def _split_cv_folds(
    subjects: pd.DataFrame,
    cv_folds: int,
    inner_val_size: float,
    random_state: int,
) -> list[tuple[list[str], list[str], list[str]]]:
    """Stratified K-fold subject-id split: (train_ids, inner_val_ids, outer_test_ids) per fold.

    Outer split: StratifiedKFold over a composite label+site stratum, so
    each fold's outer test portion is balanced on both diagnosis and
    acquisition site — falls back to label-only stratification if any
    label+site group has fewer members than cv_folds. This is checked
    proactively (not via try/except): StratifiedKFold only *warns*, it
    doesn't raise, when a stratum is smaller than n_splits (it only raises
    for far more extreme cases, e.g. n_splits exceeding even the largest
    class's count) — so relying on catching a ValueError would silently
    never trigger the fallback for the realistic case this is meant to
    guard against.

    Inner split: each fold's K-1/K "trainval" pool is further split into
    Train ((1-inner_val_size)) / Inner-Val (inner_val_size), stratified by
    label. The outer test portion is never touched until final evaluation.
    """
    subject_ids = subjects["subject_id"].to_numpy()
    labels = subjects["label"].to_numpy()
    composite_strata = _build_composite_strata(subjects)

    min_stratum_size = int(composite_strata.value_counts().min())
    if min_stratum_size < cv_folds:
        logger.warning(
            "Smallest label+site stratum has only %d member(s), fewer than cv_folds=%d; "
            "falling back to label-only stratification for the outer K-fold split",
            min_stratum_size, cv_folds,
        )
        strata_for_split = labels
    else:
        logger.info("Outer %d-fold split stratified by label+site composite", cv_folds)
        strata_for_split = composite_strata.to_numpy()

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    splits = list(skf.split(subject_ids, strata_for_split))

    folds: list[tuple[list[str], list[str], list[str]]] = []
    for trainval_pos, outer_test_pos in splits:
        trainval_ids = subject_ids[trainval_pos]
        trainval_labels = labels[trainval_pos]
        outer_test_ids = subject_ids[outer_test_pos].tolist()

        train_ids, inner_val_ids = train_test_split(
            trainval_ids,
            test_size=inner_val_size,
            stratify=trainval_labels,
            random_state=random_state,
        )
        folds.append((list(train_ids), list(inner_val_ids), outer_test_ids))

    return folds


def get_bsnip_dataloaders(
    metadata_csv: Union[str, Path] = DEFAULT_METADATA_CSV,
    batch_size: int = 8,
    num_workers: int = 4,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
    augment: bool = False,
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
        augment: If True, enable on-the-fly 3D augmentation (see
            BSNIPDataset / _augment_volume) on the train split only.
            val/test are always unaugmented regardless of this flag.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    df = pd.read_csv(metadata_csv)
    subjects = _load_subjects_table(df)

    train_ids, val_ids, test_ids = _split_subject_ids(subjects, val_size, test_size, random_state)

    train_dataset = BSNIPDataset(df, subject_ids=train_ids, is_train=augment)
    val_dataset = BSNIPDataset(df, subject_ids=val_ids, is_train=False)
    test_dataset = BSNIPDataset(df, subject_ids=test_ids, is_train=False)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, worker_init_fn=_seed_worker,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=_seed_worker,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=_seed_worker,
    )

    return train_loader, val_loader, test_loader


def get_bsnip_cv_folds(
    metadata_csv: Union[str, Path] = DEFAULT_METADATA_CSV,
    cv_folds: int = 5,
    batch_size: int = 8,
    num_workers: int = 4,
    inner_val_size: float = DEFAULT_INNER_VAL_SIZE,
    random_state: int = 42,
    augment: bool = False,
) -> list[tuple[DataLoader, DataLoader, DataLoader]]:
    """Build stratified K-fold (Train, Inner-Val, Outer-Test) DataLoaders for BSNIP.

    Each of the cv_folds folds gets its own outer test portion (via
    StratifiedKFold over a label+site composite stratum — see
    _split_cv_folds), held out entirely from that fold's training and
    threshold tuning. The remaining K-1/K "trainval" pool for that fold is
    further split into Train / Inner-Val (stratified by label,
    inner_val_size fraction). A subject never appears in more than one role
    within a fold, and every subject is some fold's outer test exactly once.

    Args:
        metadata_csv: Path to bsnip_preprocessed_npy_metadata.csv.
        cv_folds: Number of stratified folds (K).
        batch_size: DataLoader batch size (same for all loaders/folds).
        num_workers: DataLoader worker count (same for all loaders/folds).
        inner_val_size: Fraction of each fold's trainval pool held out for
            Inner-Val (default 0.2, i.e. Train 80% / Inner-Val 20%).
        random_state: Seed for reproducible splits.
        augment: If True, enable on-the-fly 3D augmentation on each fold's
            Train split only. Inner-Val and Outer-Test are always
            unaugmented regardless of this flag.

    Returns:
        A list of length cv_folds of (train_loader, inner_val_loader,
        outer_test_loader) tuples, one per fold.
    """
    df = pd.read_csv(metadata_csv)
    subjects = _load_subjects_table(df)

    fold_ids = _split_cv_folds(subjects, cv_folds, inner_val_size, random_state)

    loaders: list[tuple[DataLoader, DataLoader, DataLoader]] = []
    for fold_idx, (train_ids, inner_val_ids, outer_test_ids) in enumerate(fold_ids, start=1):
        train_dataset = BSNIPDataset(df, subject_ids=train_ids, is_train=augment)
        inner_val_dataset = BSNIPDataset(df, subject_ids=inner_val_ids, is_train=False)
        outer_test_dataset = BSNIPDataset(df, subject_ids=outer_test_ids, is_train=False)

        logger.info(
            "Fold %d/%d -> train: %d, inner_val: %d, outer_test: %d",
            fold_idx, cv_folds, len(train_dataset), len(inner_val_dataset), len(outer_test_dataset),
        )

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, worker_init_fn=_seed_worker,
        )
        inner_val_loader = DataLoader(
            inner_val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=_seed_worker,
        )
        outer_test_loader = DataLoader(
            outer_test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=_seed_worker,
        )

        loaders.append((train_loader, inner_val_loader, outer_test_loader))

    return loaders


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
