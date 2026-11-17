"""Remove every trace of the images listed in wrong.txt from the project's RESULTS.

For each ID in wrong.txt (e.g. 0000991 -> ISIC_0000991) it:
  1. deletes result files named after the image, in EVERY folder of the repo
     (preprocessed/, preprocessed_no_clahe/, segmentation_5000_images/*, candidates/, reports/ ...)
  2. deletes folders named after the image (e.g. candidates/ISIC_0000991/)
  3. removes the image's rows from every CSV (segmentation_metrics_*.csv, reports/image_tags.csv, ...)
     - original CSVs are first copied to delete_backup_csv/ (only the first time, so the original is never overwritten)

NOT touched (unless --include-inputs):  Images/ (raw photos), segs/ (ground-truth masks),
                                        ima_metadata*.csv, seg_metadata*.csv
NEVER touched:  .venv, .git, __pycache__, src/, models/, *.py files

Usage (from the repo root, venv active):
    python delete_wrong.py                    # DRY RUN: only shows what would happen
    python delete_wrong.py --delete           # really delete / edit
    python delete_wrong.py --delete --include-inputs   # also delete raw images, GT masks, metadata rows
"""
import argparse
import os
import re
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
WRONG_FILE = ROOT / "wrong.txt"
BACKUP_DIR = ROOT / "delete_backup_csv"

NEVER_SCAN = {".git", ".venv", "venv", "env", "__pycache__", "node_modules", "src", "models", BACKUP_DIR.name.lower()}
INPUT_DIRS = {"images", "segs"}                       # raw photos + ground-truth masks (top level only)
INPUT_CSV_PREFIXES = ("ima_metadata", "seg_metadata")  # dataset metadata (top level only)
OTHER_DATA_EXT = {".pkl", ".pickle", ".npz", ".npy", ".parquet", ".xlsx", ".joblib", ".json"}
ID_RE = re.compile(r"ISIC_(\d{7})(?!\d)")


def read_ids(path: Path) -> list[str]:
    """Unique 7-digit ISIC ids found in the text file (accepts `0000991`, 0000991 or ISIC_0000991)."""
    ids = re.findall(r"(?:ISIC_)?(\d{7})", path.read_text(encoding="utf-8"))
    return list(dict.fromkeys(ids))


def clean_csv(path: Path, wanted: set, apply: bool):
    """Drop rows whose image-name column holds a wanted id. Returns (rows_removed, rows_total, note)."""
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)   # dtype=str: values are written back unchanged
    except Exception as e:                                         # noqa: BLE001
        return 0, 0, f"could not read ({e})"
    drop = pd.Series(False, index=df.index)
    for col in df.columns:
        ext = df[col].astype(str).str.extract(ID_RE)[0]
        if len(ext) and ext.notna().mean() >= 0.5:                 # this column holds image names
            drop |= ext.isin(wanted)
    n = int(drop.sum())
    if n and apply:
        BACKUP_DIR.mkdir(exist_ok=True)
        backup = BACKUP_DIR / "__".join(path.relative_to(ROOT).parts)
        if not backup.exists():                                    # keep the very first original
            shutil.copy2(path, backup)
        df[~drop].to_csv(path, index=False)
    return n, len(df), ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delete", action="store_true", help="really delete / edit (default is a dry run)")
    ap.add_argument("--include-inputs", action="store_true",
                    help="also delete raw images (Images/), ground-truth masks (segs/) and metadata rows")
    args = ap.parse_args()
    apply = args.delete

    assert WRONG_FILE.exists(), f"Not found: {WRONG_FILE}"
    wanted = set(read_ids(WRONG_FILE))
    print(f"IDs in wrong.txt: {len(wanted)}")
    print(f"Mode: {'DELETE' if apply else 'DRY RUN (nothing will be changed)'}"
          f"{' + INPUTS (raw images, GT masks, metadata)' if args.include_inputs else ''}\n")

    def hit(name: str) -> bool:
        return any(i in wanted for i in ID_RE.findall(name))

    files, dirs, csvs, other_data = [], [], [], []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dp = Path(dirpath)
        at_root = dp == ROOT
        keep = []
        for d in dirnames:
            if d.lower() in NEVER_SCAN:
                continue
            if at_root and d.lower() in INPUT_DIRS and not args.include_inputs:
                continue
            if hit(d):                                   # a folder named after an image
                dirs.append(dp / d)
                continue
            keep.append(d)
        dirnames[:] = keep
        for f in filenames:
            p, low = dp / f, f.lower()
            if low.endswith(".py"):
                continue
            if low.endswith(".csv"):
                if at_root and low.startswith(INPUT_CSV_PREFIXES) and not args.include_inputs:
                    continue
                csvs.append(p)
            elif hit(f):
                files.append(p)
            elif p.suffix.lower() in OTHER_DATA_EXT and not ID_RE.search(f):
                other_data.append(p)      # combined file (no image id in its name) -> can't be edited automatically

    verb = "deleting" if apply else "would delete"
    for p in sorted(files):
        print(f"{verb} file:   {p.relative_to(ROOT)}")
        if apply:
            p.unlink(missing_ok=True)
    for p in sorted(dirs):
        print(f"{verb} folder: {p.relative_to(ROOT)}")
        if apply:
            shutil.rmtree(p, ignore_errors=True)

    rows_removed = 0
    print()
    for p in sorted(csvs):
        n, total, note = clean_csv(p, wanted, apply)
        if note:
            print(f"skipped CSV {p.relative_to(ROOT)}: {note}")
        elif n:
            print(f"{'removed' if apply else 'would remove'} {n} rows of {total}: {p.relative_to(ROOT)}")
            rows_removed += n

    print(f"\n{'Deleted' if apply else 'Would delete'}: {len(files)} files, {len(dirs)} folders; "
          f"{'removed' if apply else 'would remove'} {rows_removed} CSV rows")
    if other_data:
        print("\nHeads-up (NOT deleted, just FYI): these combined data files may contain rows for the deleted images:")
        for p in sorted(other_data):
            print(f"   {p.relative_to(ROOT)}")
        print("   (regenerate them, or check them, after deleting)")
    if apply and rows_removed:
        print(f"\nOriginal CSVs backed up in: {BACKUP_DIR.name}/")
    if not apply:
        print("\nLooks right? Run again with:  python delete_wrong.py --delete")


if __name__ == "__main__":
    main()