"""Ground-truth masks and benign/malignant labels for IMA++."""

import os
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .config import MASK_EXTENSIONS


# One annotator's mask, for example:
# ISIC_0000000_A04_T3_S1_5463934bbae47821f88025ad.png
# re.IGNORECASE makes matching robust to ISIC/isic filename capitalization.
_SINGLE = re.compile(
    r"^(ISIC_\d+)_A\d+_T\d+_S\d+_\w+$",
    flags=re.IGNORECASE,
)

# Consensus masks, for example:
# ISIC_0000004_MV_MV_MV_MV.png
# ISIC_0000004_ST_ST_ST_ST.png
_CONSENSUS = re.compile(
    r"^(ISIC_\d+)_(MV|ST)_(?:MV|ST)_(?:MV|ST)_(?:MV|ST)$",
    flags=re.IGNORECASE,
)


def index_masks(gt_dir: Path) -> dict:
    """Index mask files directly inside ``gt_dir``.

    Returns a dictionary in this format::

        {
            "ISIC_0000004": {
                "single": [Path(...), ...],
                "MV": Path(...) or None,
                "ST": Path(...) or None,
            }
        }

    This function scans only the immediate contents of ``gt_dir``; it does
    not search nested subdirectories.
    """
    gt_dir = Path(gt_dir)
    if not gt_dir.exists():
        raise FileNotFoundError(f"Ground-truth folder does not exist: {gt_dir}")
    if not gt_dir.is_dir():
        raise NotADirectoryError(f"Ground-truth path is not a directory: {gt_dir}")

    index = defaultdict(lambda: {"single": [], "MV": None, "ST": None})

    with os.scandir(gt_dir) as entries:
        for entry in entries:
            if not entry.is_file():
                continue

            mask_path = Path(entry.path)
            if mask_path.suffix.lower() not in MASK_EXTENSIONS:
                continue

            # Consensus masks first, then individual annotator masks.
            match = _CONSENSUS.fullmatch(mask_path.stem)
            if match:
                image_id = match.group(1).upper()
                consensus_type = match.group(2).upper()
                index[image_id][consensus_type] = mask_path
                continue

            match = _SINGLE.fullmatch(mask_path.stem)
            if match:
                image_id = match.group(1).upper()
                index[image_id]["single"].append(mask_path)

    # Stable ordering makes fallback selection reproducible.
    for item in index.values():
        item["single"].sort(key=lambda path: path.name.lower())

    return dict(index)


def pick_ground_truth(entry: dict, consensus_tag: str):
    """Choose one reference mask for an image.

    Selection order:
      1. Requested consensus mask (MV or ST), if available.
      2. The sole individual annotator mask, if exactly one exists.
      3. First alphabetically sorted individual mask if multiple exist but
         the requested consensus mask is unavailable.
      4. Missing, if no usable mask exists.

    Returns ``(path, source, n_annotators)``.
    """
    consensus_tag = str(consensus_tag).upper()
    if consensus_tag not in {"MV", "ST"}:
        raise ValueError(
            f"consensus_tag must be 'MV' or 'ST', got {consensus_tag!r}"
        )

    individual_masks = entry.get("single", [])
    n_annotators = len(individual_masks)

    consensus_path = entry.get(consensus_tag)
    if consensus_path is not None:
        return consensus_path, f"consensus_{consensus_tag}", n_annotators

    if n_annotators == 1:
        return individual_masks[0], "single", n_annotators

    if n_annotators > 1:
        ordered_masks = sorted(individual_masks, key=lambda path: path.name.lower())
        return ordered_masks[0], "single_first_of_many", n_annotators

    return None, "missing", 0


def read_mask(path: Path):
    """Read a ground-truth mask as a boolean array (True = lesion)."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None

    threshold = 0 if image.max() <= 1 else 127
    return image > threshold


def load_labels(path: Path, column: str) -> dict:
    """Return ``{ISIC_id: 'benign' | 'malignant'}``.

    Other diagnosis values, including indeterminate and missing values, are
    excluded from the benign-versus-malignant analysis.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Label file not found: {path}")

    if path.suffix.lower() in {".xlsx", ".xls"}:
        dataframe = pd.read_excel(path)
    else:
        dataframe = pd.read_csv(path)

    # Be tolerant of accidental whitespace in CSV column headers.
    dataframe.columns = dataframe.columns.astype(str).str.strip()

    if "isic_id" not in dataframe.columns:
        raise KeyError(
            f"No 'isic_id' column in {path}. Columns: {list(dataframe.columns)}"
        )
    if column not in dataframe.columns:
        raise KeyError(
            f"No {column!r} column in {path}. "
            f"Set LABEL_COLUMN in config.py. Columns: {list(dataframe.columns)}"
        )

    labels = dataframe[["isic_id", column]].drop_duplicates("isic_id").copy()
    labels["isic_id"] = labels["isic_id"].astype(str).str.strip().str.upper()
    labels[column] = labels[column].astype(str).str.strip().str.lower()

    print(f"Label file: {len(dataframe)} rows | values in {column!r}:")
    print(labels[column].value_counts().to_string())

    eligible_labels = labels[labels[column].isin(["benign", "malignant"])]
    return dict(zip(eligible_labels["isic_id"], eligible_labels[column]))
