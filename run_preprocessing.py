"""Preprocess every image in Images/ and write PNGs to two folders:
  preprocessed/            hair removal + median + CLAHE  (for segmentation)
  preprocessed_no_clahe/   hair removal + median only     (for colour features)
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import cv2
import pandas as pd
from tqdm import tqdm

from src.config import (CFG, IMAGE_EXTENSIONS, OUT_DIR, OUT_NO_CLAHE_DIR,
                        RAW_DIR, REPORTS_DIR)
from src.preprocessing import run_pipeline

LOG_PATH = REPORTS_DIR / "preprocessing_log.csv"


def process_one(job):
    in_path, out_path, out_nc_path = job
    cv2.setNumThreads(1)  # avoid oversubscription across worker processes
    row = {"image": in_path.stem, "status": "ok",
           "height": None, "width": None,
           "hair_fraction": None, "hair_threshold": None}
    try:
        img = cv2.imread(str(in_path), cv2.IMREAD_COLOR)
        if img is None:
            row["status"] = "read_failed"
            return row
        result = run_pipeline(img, CFG)
        ok_final = cv2.imwrite(str(out_path), result.final)         # with CLAHE
        ok_nc = cv2.imwrite(str(out_nc_path), result.denoised)      # before CLAHE
        if not (ok_final and ok_nc):
            row["status"] = "write_failed"
            return row
        row.update(height=img.shape[0], width=img.shape[1],
                   hair_fraction=round(result.hair_fraction, 5),
                   hair_threshold=result.hair_threshold_used)
    except Exception as e:  # keep the batch running; record the failure
        row["status"] = f"error: {e}"
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="only process the first N images (for testing)")
    parser.add_argument("--overwrite", action="store_true",
                        help="reprocess images that already exist in both output folders")
    parser.add_argument("--workers", type=int,
                        default=max(1, (os.cpu_count() or 2) - 1))
    args = parser.parse_args()

    raw_files = sorted(p for p in RAW_DIR.iterdir()
                       if p.suffix.lower() in IMAGE_EXTENSIONS)
    assert raw_files, f"No images found in {RAW_DIR}"
    stems = [p.stem for p in raw_files]
    assert len(stems) == len(set(stems)), \
        "Duplicate file names (ignoring extension) found in Images/"
    print(f"Raw images found: {len(raw_files)}")

    if args.limit:
        raw_files = raw_files[:args.limit]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_NO_CLAHE_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # An image counts as done only if BOTH outputs exist.
    jobs, skipped = [], 0
    for p in raw_files:
        out = OUT_DIR / f"{p.stem}.png"
        out_nc = OUT_NO_CLAHE_DIR / f"{p.stem}.png"
        if out.exists() and out_nc.exists() and not args.overwrite:
            skipped += 1
        else:
            jobs.append((p, out, out_nc))
    print(f"Selected: {len(raw_files)} | to process: {len(jobs)} | already done (skipped): {skipped}")

    rows = []
    if jobs:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for row in tqdm(ex.map(process_one, jobs, chunksize=8), total=len(jobs)):
                rows.append(row)

    if rows:
        new = pd.DataFrame(rows)
        if LOG_PATH.exists():
            log = pd.concat([pd.read_csv(LOG_PATH), new]).drop_duplicates("image", keep="last")
        else:
            log = new
        log.sort_values("image").to_csv(LOG_PATH, index=False)

    failed = [r for r in rows if r["status"] != "ok"]
    print(f"Processed OK: {len(rows) - len(failed)} | failed: {len(failed)}")
    for r in failed[:10]:
        print("  FAILED:", r["image"], r["status"])

    if LOG_PATH.exists():
        log = pd.read_csv(LOG_PATH)
        ok = log[log["status"] == "ok"]
        print(f"Log rows: {len(log)} | status ok: {len(ok)}")
        print("hair_fraction summary:")
        print(ok["hair_fraction"].describe().round(4).to_string())

    missing = [p.stem for p in raw_files
               if not ((OUT_DIR / f"{p.stem}.png").exists()
                       and (OUT_NO_CLAHE_DIR / f"{p.stem}.png").exists())]
    assert not missing, f"{len(missing)} selected images lack an output, e.g. {missing[:5]}"
    n_main = len(list(OUT_DIR.glob("*.png")))
    n_nc = len(list(OUT_NO_CLAHE_DIR.glob("*.png")))
    print(f"Files in preprocessed/: {n_main} | in preprocessed_no_clahe/: {n_nc}")
    print("Check passed: every selected image has an output in both folders")


if __name__ == "__main__":  # required on Windows for multiprocessing
    main()