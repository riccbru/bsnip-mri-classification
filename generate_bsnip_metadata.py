"""generate_bsnip_metadata.py

Build the BSNIP binary (HC vs SZ) metadata CSV from the pre-extracted dataset
and manifest on Midway2 at /project/bonnietfleming/wenjie/BSNIP2_FINAL_wmr/,
replacing the earlier raw-Drive / master_bsnip.xlsx workflow.

Steps:
    1. Read manifest_master.csv (columns: subject_id, site, rel_path,
       final_dx, prep_rating).
    2. Keep only rows that passed quality control (prep_rating == 1).
    3. Map final_dx to a binary label (HC -> 0, SZ -> 1); drop any other
       diagnosis (e.g. SAD, BPP).
    4. Build nii_path = {raw_data_root}/{rel_path} and verify each file
       actually exists on disk, dropping rows whose file is missing.
    5. Add npy_path = {npy_dir}/{subject_id}.npy.
    6. Write data/bsnip_binary_metadata.csv with columns:
         subject_id, site, label, nii_path, npy_path
       and print HC vs SZ class distribution.

Usage:
    python generate_bsnip_metadata.py \\
        --manifest-csv /project/bonnietfleming/wenjie/BSNIP2_FINAL_wmr/manifest_master.csv \\
        --raw-data-root /project/bonnietfleming/wenjie/BSNIP2_FINAL_wmr \\
        --output-csv data/bsnip_binary_metadata.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

logger = logging.getLogger("generate_bsnip_metadata")

DEFAULT_RAW_DATA_ROOT = Path("/project/bonnietfleming/wenjie/BSNIP2_FINAL_wmr")
DEFAULT_MANIFEST_CSV = DEFAULT_RAW_DATA_ROOT / "manifest_master.csv"
DEFAULT_NPY_DIR = Path("data/bsnip_npy")
DEFAULT_OUTPUT_CSV = Path("data/bsnip_binary_metadata.csv")

# manifest_master.csv column names.
SUBJECT_COL = "subject_id"
SITE_COL = "site"
REL_PATH_COL = "rel_path"
DIAGNOSIS_COL = "final_dx"
QC_COL = "prep_rating"

LABEL_MAP: dict[str, int] = {"HC": 0, "SZ": 1}

OUTPUT_COLUMNS: list[str] = ["subject_id", "site", "label", "nii_path", "npy_path"]


def load_manifest(manifest_csv: Path) -> pd.DataFrame:
    """Read manifest_master.csv."""
    logger.info("Reading manifest %s", manifest_csv)
    df = pd.read_csv(manifest_csv)
    df.columns = [str(c).strip() for c in df.columns]
    logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))
    return df


def filter_quality_control(df: pd.DataFrame, qc_col: str = QC_COL) -> pd.DataFrame:
    """Keep only rows with prep_rating == 1 (clean quality control)."""
    before = len(df)
    qc_numeric = pd.to_numeric(df[qc_col], errors="coerce")
    filtered = df.loc[qc_numeric == 1].copy()
    logger.info("QC filter '%s == 1': %d -> %d rows", qc_col, before, len(filtered))
    return filtered


def filter_and_label_diagnosis(
    df: pd.DataFrame,
    diagnosis_col: str = DIAGNOSIS_COL,
    label_map: dict[str, int] = LABEL_MAP,
) -> pd.DataFrame:
    """Keep only HC/SZ rows and add a binary `label` column."""
    before = len(df)
    df = df.copy()
    diagnosis_norm = df[diagnosis_col].astype(str).str.strip()
    keep_mask = diagnosis_norm.isin(label_map.keys())

    dropped = diagnosis_norm[~keep_mask].value_counts()
    if not dropped.empty:
        logger.info("Dropping diagnoses outside %s:\n%s", list(label_map.keys()), dropped.to_string())

    df = df.loc[keep_mask].copy()
    df["label"] = diagnosis_norm.loc[keep_mask].map(label_map).astype(int)
    logger.info("Diagnosis filter/label: %d -> %d rows", before, len(df))
    return df


def build_nii_path(df: pd.DataFrame, raw_data_root: Path, rel_path_col: str = REL_PATH_COL) -> pd.DataFrame:
    """Add nii_path = raw_data_root / rel_path."""
    df = df.copy()
    df["nii_path"] = df[rel_path_col].apply(lambda rel: str(raw_data_root / str(rel)))
    return df


def verify_files_exist(df: pd.DataFrame, nii_path_col: str = "nii_path") -> pd.DataFrame:
    """Drop rows whose nii_path doesn't actually exist on disk."""
    before = len(df)
    exists_mask = df[nii_path_col].apply(lambda p: Path(p).exists())
    missing = df.loc[~exists_mask, nii_path_col]
    if not missing.empty:
        logger.warning("Missing %d NIfTI file(s) on disk, e.g.:\n%s", len(missing), missing.head(10).to_string())
    df = df.loc[exists_mask].copy()
    logger.info("File-existence check: %d -> %d rows", before, len(df))
    return df


def add_npy_path(df: pd.DataFrame, npy_dir: Path, subject_col: str = SUBJECT_COL) -> pd.DataFrame:
    """Add npy_path = npy_dir / {subject_id}.npy."""
    df = df.copy()
    df["npy_path"] = df[subject_col].apply(lambda sid: str(npy_dir / f"{sid}.npy"))
    return df


def log_class_distribution(df: pd.DataFrame, label_col: str = "label", site_col: str = SITE_COL) -> None:
    """Print HC vs SZ class counts, and a per-site breakdown."""
    label_names = {v: k for k, v in LABEL_MAP.items()}
    counts = df[label_col].map(label_names).value_counts()
    logger.info("Class distribution (HC vs SZ):\n%s", counts.to_string())
    logger.info(
        "Site x class counts:\n%s",
        pd.crosstab(df[site_col], df[label_col].map(label_names)).to_string(),
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate BSNIP binary (HC vs SZ) metadata CSV from manifest_master.csv "
        "on Midway2 (/project/bonnietfleming/wenjie/BSNIP2_FINAL_wmr/).",
    )
    parser.add_argument(
        "--manifest-csv", type=Path, default=DEFAULT_MANIFEST_CSV,
        help="Path to manifest_master.csv (default: %(default)s)",
    )
    parser.add_argument(
        "--raw-data-root", type=Path, default=DEFAULT_RAW_DATA_ROOT,
        help="Base directory rel_path is relative to, used to build nii_path (default: %(default)s)",
    )
    parser.add_argument(
        "--npy-dir", type=Path, default=DEFAULT_NPY_DIR,
        help="Directory npy_path is constructed under (default: %(default)s)",
    )
    parser.add_argument(
        "--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV,
        help="Output CSV path (default: %(default)s)",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: %(default)s)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    df = load_manifest(args.manifest_csv)
    df = filter_quality_control(df)
    df = filter_and_label_diagnosis(df)
    df = build_nii_path(df, args.raw_data_root)
    df = verify_files_exist(df)
    df = add_npy_path(df, args.npy_dir)

    log_class_distribution(df)

    df_out = df[OUTPUT_COLUMNS]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(args.output_csv, index=False)
    logger.info("Saved %d entries to %s", len(df_out), args.output_csv)


if __name__ == "__main__":
    main()
