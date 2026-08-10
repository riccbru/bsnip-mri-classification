"""generate_bsnip_metadata.py

Build the BSNIP binary (HC vs SZ) metadata CSV from Wenjie's
`manifest_master.csv`, translating that manifest's site-specific column
names into this project's standardized schema via an external column-mapping
config (`config/col_map.json`) rather than hard-coding names in the script.

Standardized schema (post-rename):
    subject_id, site, rel_path, diagnosis, qc_rating

Steps:
    1. Read manifest_master.csv.
    2. Rename columns per the mapping in config/col_map.json.
    3. Filter for qc_rating == 1.
    4. Filter for diagnosis in {HC, SZ} and add a binary `label` column
       (HC -> 0, SZ -> 1).
    5. Add `nii_path`  = {raw_data_root}/{rel_path}
       Add `npy_path`  = {npy_dir}/{subject_id}.npy
    6. Save to data/bsnip_binary_metadata.csv, logging subject counts per
       class and per site.

Usage:
    python src/data/generate_bsnip_metadata.py \\
        --manifest-csv /project/bonnietfleming/wenjie/BSNIP2_FINAL_wmr/manifest_master.csv \\
        --raw-data-root /project/bonnietfleming/wenjie/BSNIP2_FINAL_wmr \\
        --output-csv data/bsnip_binary_metadata.csv
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path, PurePosixPath
from typing import Optional, Sequence

import pandas as pd

logger = logging.getLogger("generate_bsnip_metadata")

# Manifest lives on the RCC cluster, alongside the preprocessed volumes it
# points to via `rel_path`.
DEFAULT_MANIFEST_CSV = Path(
    "/project/bonnietfleming/wenjie/BSNIP2_FINAL_wmr/manifest_master.csv"
)
DEFAULT_RAW_DATA_ROOT = Path("/project/bonnietfleming/wenjie/BSNIP2_FINAL_wmr")
DEFAULT_NPY_DIR = Path("data/bsnip_npy")
DEFAULT_OUTPUT_CSV = Path("data/bsnip_binary_metadata.csv")

# Column-mapping config: {source_manifest_key: {source_col: internal_col}, "label_mapping": {...}}
DEFAULT_COL_MAP_JSON = Path("config/col_map.json")
DEFAULT_MANIFEST_KEY = "bsnip_wenjie_manifest"

# Used only if config/col_map.json is missing or malformed, so the script
# still runs; keep in sync with config/col_map.json.
FALLBACK_COLUMN_MAP: dict[str, str] = {
    "subject_id": "subject_id",
    "site": "site",
    "rel_path": "rel_path",
    "final_dx": "diagnosis",
    "prep_rating": "qc_rating",
}
FALLBACK_LABEL_MAP: dict[str, int] = {"HC": 0, "SZ": 1}


def load_column_config(
    col_map_json: Path, manifest_key: str
) -> tuple[dict[str, str], dict[str, int]]:
    """Load the {source_col: internal_col} rename map and diagnosis->label map."""
    if not col_map_json.exists():
        logger.warning(
            "%s not found; using in-code fallback column/label mapping", col_map_json
        )
        return dict(FALLBACK_COLUMN_MAP), dict(FALLBACK_LABEL_MAP)

    with col_map_json.open() as f:
        config = json.load(f)

    column_map = config.get(manifest_key)
    if column_map is None:
        raise KeyError(
            f"'{manifest_key}' not found in {col_map_json}; "
            f"available keys: {list(config.keys())}"
        )
    label_map = config.get("label_mapping", FALLBACK_LABEL_MAP)

    logger.info("Loaded column map from %s (key='%s'): %s", col_map_json, manifest_key, column_map)
    logger.info("Loaded label map: %s", label_map)
    return column_map, label_map


def load_manifest(manifest_csv: Path) -> pd.DataFrame:
    """Read the raw BSNIP manifest CSV."""
    logger.info("Reading manifest %s", manifest_csv)
    df = pd.read_csv(manifest_csv)
    df.columns = [str(c).strip() for c in df.columns]
    logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))
    return df


def rename_to_standard_schema(df: pd.DataFrame, column_map: dict[str, str]) -> pd.DataFrame:
    """Rename manifest columns to the internal standardized schema and select them."""
    missing = [src for src in column_map if src not in df.columns]
    if missing:
        raise KeyError(
            f"Manifest is missing expected source columns {missing}; "
            f"available columns: {list(df.columns)}"
        )
    renamed = df.rename(columns=column_map)
    standardized_cols = list(column_map.values())
    logger.info("Renamed columns to standardized schema: %s", standardized_cols)
    return renamed[standardized_cols].copy()


def filter_quality_control(df: pd.DataFrame, qc_col: str = "qc_rating") -> pd.DataFrame:
    """Keep only rows with qc_rating == 1 (clean quality control)."""
    before = len(df)
    qc_numeric = pd.to_numeric(df[qc_col], errors="coerce")
    filtered = df.loc[qc_numeric == 1].copy()
    logger.info("QC filter '%s == 1': %d -> %d rows", qc_col, before, len(filtered))
    return filtered


def filter_and_label_diagnosis(
    df: pd.DataFrame, label_map: dict[str, int], diagnosis_col: str = "diagnosis"
) -> pd.DataFrame:
    """Keep only rows whose diagnosis is a label_map key, and add a `label` column."""
    before = len(df)
    df = df.copy()
    diagnosis_norm = df[diagnosis_col].astype(str).str.strip()
    keep_mask = diagnosis_norm.isin(label_map.keys())

    dropped = diagnosis_norm[~keep_mask].value_counts()
    if not dropped.empty:
        logger.info("Dropping diagnoses outside %s:\n%s", list(label_map.keys()), dropped.to_string())

    df = df.loc[keep_mask].copy()
    df[diagnosis_col] = diagnosis_norm.loc[keep_mask]
    df["label"] = df[diagnosis_col].map(label_map).astype(int)
    logger.info("Diagnosis filter/label: %d -> %d rows", before, len(df))
    return df


def add_derived_paths(df: pd.DataFrame, raw_data_root: Path, npy_dir: Path) -> pd.DataFrame:
    """Add absolute `nii_path` and `npy_path` columns."""
    df = df.copy()
    # Use PurePosixPath: raw_data_root/rel_path are RCC (Linux) cluster paths,
    # independent of the host OS this script is authored/run on.
    raw_root_posix = PurePosixPath(str(raw_data_root))
    npy_dir_posix = PurePosixPath(str(npy_dir))

    df["nii_path"] = df["rel_path"].apply(lambda rel: str(raw_root_posix / str(rel)))
    df["npy_path"] = df["subject_id"].apply(lambda sid: str(npy_dir_posix / f"{sid}.npy"))
    return df


def log_class_and_site_counts(df: pd.DataFrame) -> None:
    """Log subject counts per class (HC vs SZ) and per site."""
    logger.info("Class counts (diagnosis):\n%s", df["diagnosis"].value_counts().to_string())
    logger.info(
        "Site x diagnosis counts:\n%s",
        pd.crosstab(df["site"], df["diagnosis"]).to_string(),
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate BSNIP binary (HC vs SZ) metadata CSV from manifest_master.csv.",
    )
    parser.add_argument(
        "--manifest-csv", type=Path, default=DEFAULT_MANIFEST_CSV,
        help="Path to manifest_master.csv (default: %(default)s)",
    )
    parser.add_argument(
        "--col-map-json", type=Path, default=DEFAULT_COL_MAP_JSON,
        help="Path to the column-mapping config JSON (default: %(default)s)",
    )
    parser.add_argument(
        "--manifest-key", type=str, default=DEFAULT_MANIFEST_KEY,
        help="Key in col-map-json holding this manifest's column map (default: %(default)s)",
    )
    parser.add_argument(
        "--raw-data-root", type=Path, default=DEFAULT_RAW_DATA_ROOT,
        help="Root directory rel_path is relative to, used to build nii_path (default: %(default)s)",
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

    column_map, label_map = load_column_config(args.col_map_json, args.manifest_key)

    df = load_manifest(args.manifest_csv)
    df = rename_to_standard_schema(df, column_map)
    df = filter_quality_control(df)
    df = filter_and_label_diagnosis(df, label_map)
    df = add_derived_paths(df, args.raw_data_root, args.npy_dir)

    log_class_and_site_counts(df)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    logger.info("Saved %d entries to %s", len(df), args.output_csv)


if __name__ == "__main__":
    main()
