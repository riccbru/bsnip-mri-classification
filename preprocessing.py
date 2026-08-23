"""preprocessing.py

Convert BSNIP 3D NIfTI Grey Matter volumes into normalized NumPy arrays.

Mirrors the ADNI pipeline's `preprocessing.py`, adapted to BSNIP's metadata
CSV (produced by generate_bsnip_metadata.py) and a simple min-max
normalization (no resizing).

Steps:
    1. Read data/bsnip_binary_metadata.csv (columns: subject_id, site,
       label, nii_path, npy_path).
    2. Ensure the target directory data/bsnip_npy exists.
    3. For each subject (tqdm progress bar):
         - Load the 3D volume from nii_path via nibabel.
         - Cast to np.float32.
         - Min-max normalize: (img - img.min()) / (img.max() - img.min() + 1e-8)
         - Save to npy_path.
       Errors (corrupted/unreadable files) are logged and the subject is
       skipped, not fatal to the run.
    4. Write bsnip_preprocessed_npy_metadata.csv containing only the
       subjects that converted successfully (subject_id, site, label,
       nii_path, npy_path), and print the array shape (D, H, W) of a
       sample volume for verification.

Usage:
    python preprocessing.py \\
        --input-csv data/bsnip_binary_metadata.csv \\
        --output-dir data/bsnip_npy \\
        --output-csv bsnip_preprocessed_npy_metadata.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional, Sequence

import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm

from logging_utils import setup_logging

logger = logging.getLogger("preprocessing")

DEFAULT_INPUT_CSV = Path("data/bsnip_binary_metadata.csv")
DEFAULT_OUTPUT_DIR = Path("data/bsnip_npy")
DEFAULT_OUTPUT_CSV = Path("bsnip_preprocessed_npy_metadata.csv")

OUTPUT_COLUMNS: list[str] = ["subject_id", "site", "label", "nii_path", "npy_path"]


def load_metadata(input_csv: Path) -> pd.DataFrame:
    """Read the BSNIP metadata CSV listing subjects and their nii_path."""
    logger.info("Reading metadata %s", input_csv)
    df = pd.read_csv(input_csv)
    logger.info("Loaded %d subjects", len(df))
    return df


def resolve_npy_path(row: pd.Series, output_dir: Path) -> Path:
    """Use the row's npy_path if present, else fall back to output_dir/{subject_id}.npy."""
    npy_path = row.get("npy_path")
    if isinstance(npy_path, str) and npy_path.strip():
        return Path(npy_path)
    return output_dir / f"{row['subject_id']}.npy"


def normalize_min_max(img: np.ndarray) -> np.ndarray:
    """Min-max normalize a volume to roughly [0, 1]."""
    return (img - img.min()) / (img.max() - img.min() + 1e-8)


def load_and_normalize_volume(nii_path: Path) -> np.ndarray:
    """Load a NIfTI volume and apply float32 cast + min-max normalization."""
    volume = nib.load(str(nii_path))
    img = volume.get_fdata().astype(np.float32)
    return normalize_min_max(img)


def process_subjects(df: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, Optional[tuple[int, int, int]]]:
    """Convert every subject's .nii to a normalized .npy, skipping failures.

    Returns the metadata for successfully converted subjects, plus the
    (D, H, W) shape of the first successfully processed volume (if any).
    """
    successes: list[dict[str, object]] = []
    sample_shape: Optional[tuple[int, int, int]] = None
    failures = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing volumes"):
        subject_id = row["subject_id"]
        nii_path = Path(row["nii_path"])
        npy_path = resolve_npy_path(row, output_dir)

        try:
            img = load_and_normalize_volume(nii_path)
            npy_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(npy_path, img)
        except Exception as exc:  # noqa: BLE001 - log and continue past bad files
            logger.error("Failed to preprocess subject %s (%s): %s", subject_id, nii_path, exc)
            failures += 1
            continue

        if sample_shape is None:
            sample_shape = img.shape  # type: ignore[assignment]

        successes.append(
            {
                "subject_id": subject_id,
                "site": row["site"],
                "label": row["label"],
                "nii_path": str(nii_path),
                "npy_path": str(npy_path),
            }
        )

    logger.info("Preprocessing done: %d succeeded, %d failed", len(successes), failures)
    return pd.DataFrame(successes, columns=OUTPUT_COLUMNS), sample_shape


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert BSNIP NIfTI volumes to min-max normalized NumPy arrays.",
    )
    parser.add_argument(
        "--input-csv", type=Path, default=DEFAULT_INPUT_CSV,
        help="Path to the BSNIP metadata CSV (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Directory .npy volumes are saved under (default: %(default)s)",
    )
    parser.add_argument(
        "--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV,
        help="Output CSV listing successfully converted subjects (default: %(default)s)",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: %(default)s)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    setup_logging(args.log_level)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_metadata(args.input_csv)
    df_out, sample_shape = process_subjects(df, args.output_dir)

    df_out.to_csv(args.output_csv, index=False)
    logger.info("Saved %d entries to %s", len(df_out), args.output_csv)

    if sample_shape is not None:
        logger.info("Sample volume shape (D, H, W): %s", sample_shape)
        print(f"Sample volume shape (D, H, W): {sample_shape}")
    else:
        logger.warning("No volumes were successfully processed; no sample shape to report")


if __name__ == "__main__":
    main()
