"""Ground-truth masks and benign/malignant labels for IMA++."""
import os
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .config import MASK_EXTENSIONS

# One annotator's mask, e.g. ISIC_0000000_A04_T3_S1_5463934bbae47821f88025ad
_SINGLE = re.compile(r"^(ISIC_\d+)_A\d+_T\d+_S\d+_\w+$")
# Consensus mask, e.g. ISIC_0000004_MV_MV_MV_MV or ISIC_0000004_ST_ST_ST_ST
_CONSENSUS = re.compile(r"^(ISIC_\d+)_(MV|ST)_(?:MV|ST)_(?:MV|ST)_(?:MV|ST)$")


def index_masks(gt_dir: Path) -> dict:
    """Scan the mask folder once.
    Returns {isic_id: {"single": [paths], "MV": path or None, "ST": path or None}}."""
    index = defaultdict(lambda: {"single": [], "MV": None, "ST": None})
    with os.scandir(gt_dir) as it:
        for entry in it:
            p = Path(entry.path)
            if p.suffix.lower() not in MASK_EXTENSIONS:
                continue
            m = _CONSENSUS.match(p.stem)
            if m:
                index[m.group(1)][m.group(2)] = p
                continue
            m = _SINGLE.match(p.stem)
            if m:
                index[m.group(1)]["single"].append(p)
    return dict(index)


def pick_ground_truth(entry: dict, consensus_tag: str):
    """Choose one mask per image. Returns (path, source, n_annotators).
    - consensus mask if the image has one (it has several annotators)
    - otherwise the single annotator's mask"""
    n = len(entry["single"])
    if entry.get(consensus_tag) is not None:
        return entry[consensus_tag], f"consensus_{consensus_tag}", n
    if n == 1:
        return entry["single"][0], "single", n
    if n > 1:  # several annotators but no consensus mask: unexpected, flagged in the log
        return sorted(entry["single"])[0], "single_first_of_many", n
    return None, "missing", 0


def read_mask(path: Path):
    """Read a mask as a boolean array (True = lesion)."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return img > (0 if img.max() <= 1 else 127)


def load_labels(path: Path, column: str) -> dict:
    """Return {isic_id: 'benign' | 'malignant'}. Other values (e.g. indeterminate) are dropped."""
    assert path.exists(), f"Label file not found: {path}"
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    assert "isic_id" in df.columns, f"No 'isic_id' column. Columns: {list(df.columns)}"
    assert column in df.columns, \
        f"No '{column}' column. Set LABEL_COLUMN in config.py. Columns: {list(df.columns)}"
    lab = df[["isic_id", column]].drop_duplicates("isic_id").copy()
    lab[column] = lab[column].astype(str).str.strip().str.lower()
    print(f"Label file: {len(df)} rows | values in '{column}':")
    print(lab[column].value_counts().to_string())
    keep = lab[lab[column].isin(["benign", "malignant"])]
    return dict(zip(keep["isic_id"], keep[column]))