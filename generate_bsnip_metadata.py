"""generate_bsnip_metadata.py

Build a clean metadata CSV mapping BSNIP subjects to their SPM12-preprocessed
Grey Matter NIfTI volumes and a binary Control/Schizophrenia label.

Mirrors the ADNI pipeline's `match_mri_to_diagnosis.py` / `generate_metadata_csv.py`,
adapted to the BSNIP `master_bsnip.xlsx` "MASTER" sheet and BSNIP's on-disk layout:

    [Site]/subjects/completed/[Subject_ID]/Preprocessed/sm0wrp1*.nii

Steps:
    1. Read the MASTER sheet from master_bsnip.xlsx.
    2. Keep only rows that passed quality control (EI Rating == 1).
    3. Map free-text diagnosis to a binary label:
         Control / HC  -> 0
         Schizophrenia / SZ -> 1
       Any other diagnosis (e.g. Bipolar, Schizoaffective) is dropped.
    4. Resolve the preprocessed Grey Matter NIfTI path for each subject.
    5. Write bsnip_matched_nii_with_labels.csv with columns:
         subject_id, site, label, nii_path

Usage:
    python generate_bsnip_metadata.py \\
        --xlsx-path master_bsnip.xlsx \\
        --data-root /project/aereditato/abhat/BSNIP \\
        --output-csv bsnip_matched_nii_with_labels.csv
"""

from __future__ import annotations

import argparse
import logging
from glob import glob
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

logger = logging.getLogger("generate_bsnip_metadata")

# Preprocessed Grey Matter NIfTI files live at:
#   [data_root]/[Site]/subjects/completed/[Subject_ID]/Preprocessed/sm0wrp1*.nii
NII_GLOB_PATTERN = "sm0wrp1*.nii"

# Candidate column names to look for in the master sheet, tried in order,
# case-insensitively. The first entries match the actual master_bsnip.xlsx
# ("Clean_Imaging_Status" sheet: nidb_id, site, T1_QC_Rating, dxgroup); the
# rest are fallbacks in case a differently-formatted export is used.
SUBJECT_COL_CANDIDATES: tuple[str, ...] = (
    "nidb_id",
    "Subject_ID",
    "subject_id",
    "Subject",
    "SubjectID",
)
SITE_COL_CANDIDATES: tuple[str, ...] = ("site", "Site", "Site_ID", "SiteID")
EI_RATING_COL_CANDIDATES: tuple[str, ...] = (
    "T1_QC_Rating",
    "EI Rating",
    "EI_Rating",
    "EIRating",
)
DIAGNOSIS_COL_CANDIDATES: tuple[str, ...] = (
    "dxgroup",
    "dx_group",
    "Diagnosis",
    "Dx",
    "Clinical Diagnosis",
    "Group",
)

# Diagnoses are short codes in the real sheet (e.g. "HC", "SZ", "SAD", "BDP",
# "OTH"), so we match those exactly first to avoid 2-letter codes like "hc"
# colliding as substrings of unrelated text. The substring fallback below
# then also catches longer free-text diagnoses (e.g. "Healthy Control").
CONTROL_EXACT: frozenset[str] = frozenset({"hc", "control", "nc"})
SCHIZOPHRENIA_EXACT: frozenset[str] = frozenset({"sz", "scz"})
CONTROL_SUBSTRINGS: tuple[str, ...] = ("healthy control", "normal control")
SCHIZOPHRENIA_SUBSTRINGS: tuple[str, ...] = ("schizophrenia",)

# Diagnoses explicitly dropped per spec (not exhaustive — anything that
# doesn't match Control/SZ above is dropped regardless of this list).
# Kept here only to document intent: e.g. "SAD" (Schizoaffective Disorder),
# "BDP" (Bipolar Disorder), "OTH" (Other).


def load_master_sheet(xlsx_path: Path, sheet_name: str) -> pd.DataFrame:
    """Load the MASTER sheet from the BSNIP metadata workbook."""
    logger.info("Reading sheet '%s' from %s", sheet_name, xlsx_path)
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name, engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]
    logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))
    return df


def resolve_column(df: pd.DataFrame, candidates: Sequence[str], description: str) -> str:
    """Find the first column in `df` matching any of `candidates` (case-insensitive)."""
    lookup = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        match = lookup.get(candidate.lower())
        if match is not None:
            logger.info("Using column '%s' for %s", match, description)
            return match
    raise KeyError(
        f"Could not find a column for {description}. "
        f"Tried {list(candidates)}; available columns: {list(df.columns)}"
    )


def filter_quality_control(df: pd.DataFrame, ei_rating_col: str) -> pd.DataFrame:
    """Keep only rows with EI Rating == 1 (clean quality control)."""
    before = len(df)
    ei_numeric = pd.to_numeric(df[ei_rating_col], errors="coerce")
    filtered = df.loc[ei_numeric == 1].copy()
    logger.info(
        "QC filter '%s == 1': %d -> %d rows", ei_rating_col, before, len(filtered)
    )
    return filtered


def map_diagnosis_to_label(diagnosis: object) -> Optional[int]:
    """Map a diagnosis code/free-text value to a binary label, or None to drop it."""
    if pd.isna(diagnosis):
        return None
    text = str(diagnosis).strip().lower()

    if text in SCHIZOPHRENIA_EXACT:
        return 1
    if text in CONTROL_EXACT:
        return 0
    if any(phrase in text for phrase in SCHIZOPHRENIA_SUBSTRINGS):
        return 1
    if any(phrase in text for phrase in CONTROL_SUBSTRINGS):
        return 0
    return None


def add_binary_labels(df: pd.DataFrame, diagnosis_col: str) -> pd.DataFrame:
    """Add a `label` column and drop rows whose diagnosis isn't Control/SZ."""
    before = len(df)
    df = df.copy()
    df["label"] = df[diagnosis_col].apply(map_diagnosis_to_label)
    dropped = df[df["label"].isna()][diagnosis_col].value_counts()
    if not dropped.empty:
        logger.info("Dropping non Control/SZ diagnoses:\n%s", dropped.to_string())
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)
    logger.info("Diagnosis mapping: %d -> %d rows", before, len(df))
    return df


def find_nii_path(data_root: Path, site: str, subject_id: str) -> Optional[str]:
    """Resolve the sm0wrp1*.nii Grey Matter file for one subject, if present."""
    subject_dir = data_root / str(site) / "subjects" / "completed" / str(subject_id) / "Preprocessed"
    matches = sorted(glob(str(subject_dir / NII_GLOB_PATTERN)))
    if not matches:
        logger.debug("No match for subject %s (site %s) in %s", subject_id, site, subject_dir)
        return None
    if len(matches) > 1:
        logger.warning(
            "Multiple Grey Matter files for subject %s (site %s); using first: %s",
            subject_id, site, matches,
        )
    return matches[0]


def build_metadata(
    df: pd.DataFrame,
    data_root: Path,
    subject_col: str,
    site_col: str,
) -> pd.DataFrame:
    """Resolve NIfTI paths and assemble the final metadata table."""
    entries: list[dict[str, object]] = []
    missing = 0
    for _, row in df.iterrows():
        subject_id = str(row[subject_col]).strip()
        site = str(row[site_col]).strip()
        nii_path = find_nii_path(data_root, site, subject_id)
        if nii_path is None:
            missing += 1
            continue
        entries.append(
            {
                "subject_id": subject_id,
                "site": site,
                "label": int(row["label"]),
                "nii_path": nii_path,
            }
        )

    if missing:
        logger.warning("Skipped %d subjects with no matching .nii file", missing)

    return pd.DataFrame(entries, columns=["subject_id", "site", "label", "nii_path"])


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate BSNIP metadata CSV mapping subjects to preprocessed "
        "Grey Matter NIfTI files and binary Control/Schizophrenia labels.",
    )
    parser.add_argument(
        "--xlsx-path",
        type=Path,
        default=Path("master_bsnip.xlsx"),
        help="Path to master_bsnip.xlsx (default: %(default)s)",
    )
    parser.add_argument(
        "--sheet-name",
        type=str,
        # master_bsnip.xlsx has no sheet literally named "MASTER"; its master
        # subject table (QC + diagnosis + VBM prep status) is
        # "Clean_Imaging_Status". Override with --sheet-name if your workbook
        # differs.
        default="Clean_Imaging_Status",
        help="Sheet name to read (default: %(default)s)",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("."),
        help="Root directory containing the [Site]/subjects/completed/... tree "
        "(default: current directory)",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("bsnip_matched_nii_with_labels.csv"),
        help="Output CSV path (default: %(default)s)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
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

    df = load_master_sheet(args.xlsx_path, args.sheet_name)

    subject_col = resolve_column(df, SUBJECT_COL_CANDIDATES, "subject ID")
    site_col = resolve_column(df, SITE_COL_CANDIDATES, "site")
    ei_rating_col = resolve_column(df, EI_RATING_COL_CANDIDATES, "EI Rating")
    diagnosis_col = resolve_column(df, DIAGNOSIS_COL_CANDIDATES, "diagnosis")

    df = filter_quality_control(df, ei_rating_col)
    df = add_binary_labels(df, diagnosis_col)
    df_out = build_metadata(df, args.data_root, subject_col, site_col)

    df_out.to_csv(args.output_csv, index=False)
    logger.info("Saved %d entries to %s", len(df_out), args.output_csv)
    if not df_out.empty:
        logger.info("Label counts:\n%s", df_out["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
