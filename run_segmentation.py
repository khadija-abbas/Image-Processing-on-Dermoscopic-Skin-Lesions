"""Segment every preprocessed image with Otsu, GrabCut and active contours,
save the predicted masks, and score them (Dice, IoU) against ground truth.

Outputs
  segmentation/<method>/<ISIC_id>.png   predicted masks (0/255, full resolution)
  reports/segmentation_metrics.csv      one row per image and method
"""
import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import (GT_DIR, LABEL_COLUMN, LABELS_FILE, OUT_DIR, OUT_NO_CLAHE_DIR,
                        REPORTS_DIR, SEG_CFG, SEG_OUT_DIR)
from src.ground_truth import index_masks, load_labels, pick_ground_truth, read_mask
from src.metrics import dice_iou
from src.segmentation import METHODS, run_methods

METRICS_PATH = REPORTS_DIR / "segmentation_metrics.csv"


def process_one(job):
    stem, label, gt_path, gt_source, n_ann = job
    cv2.setNumThreads(1)  # avoid oversubscription across worker processes
    base = {"image": stem, "label": label, "gt_source": gt_source, "n_annotators": n_ann}
    try:
        clahe = cv2.imread(str(OUT_DIR / f"{stem}.png"), cv2.IMREAD_COLOR)
        noclahe = cv2.imread(str(OUT_NO_CLAHE_DIR / f"{stem}.png"), cv2.IMREAD_COLOR)
        gt = read_mask(gt_path)
        if clahe is None or noclahe is None or gt is None:
            raise ValueError("could not read image or ground-truth mask")
        H, W = clahe.shape[:2]
        gt_resized = False
        if gt.shape != (H, W):
            if abs((gt.shape[1] / gt.shape[0]) / (W / H) - 1) > 0.01:
                raise ValueError(f"GT shape {gt.shape} vs image {(H, W)}: aspect ratio differs")
            gt = cv2.resize(gt.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
            gt_resized = True
        if not gt.any():
            raise ValueError("ground-truth mask is empty")

        t0 = time.time()
        results, vig_frac = run_methods(clahe, noclahe, SEG_CFG)
        seconds = round(time.time() - t0, 2)
        rows = []
        for method in METHODS:
            pred, status = results[method]
            out_path = SEG_OUT_DIR / method / f"{stem}.png"
            if not cv2.imwrite(str(out_path), pred.astype(np.uint8) * 255):
                status = "write_failed"
            dice, iou = dice_iou(pred, gt)
            rows.append({**base, "method": method, "dice": round(dice, 5), "iou": round(iou, 5),
                         "status": status, "pred_area_frac": round(float(pred.mean()), 5),
                         "gt_area_frac": round(float(gt.mean()), 5),
                         "vignette_frac": round(vig_frac, 5), "gt_resized": gt_resized,
                         "height": H, "width": W, "seconds_all_methods": seconds})
        return rows
    except Exception as e:
        return [{**base, "method": m, "dice": np.nan, "iou": np.nan, "status": f"error: {e}"}
                for m in METHODS]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="only the first N eligible images")
    parser.add_argument("--overwrite", action="store_true", help="redo images that already have results")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = parser.parse_args()

    # 1) Which images are available, and what do we know about each?
    stems = sorted(p.stem for p in OUT_DIR.glob("*.png") if (OUT_NO_CLAHE_DIR / p.name).exists())
    assert stems, f"No preprocessed images found in {OUT_DIR} / {OUT_NO_CLAHE_DIR}"
    print(f"Preprocessed images (in both folders): {len(stems)}")

    assert GT_DIR.exists(), f"Ground-truth folder not found: {GT_DIR} (edit GT_DIR in src/config.py)"
    mask_index = index_masks(GT_DIR)
    print(f"Ground-truth folder: masks found for {len(mask_index)} distinct images")
    labels = load_labels(LABELS_FILE, LABEL_COLUMN)

    jobs, skipped = [], {"no_label": 0, "no_ground_truth": 0}
    src_counts = {}
    for s in stems:
        if s not in labels:
            skipped["no_label"] += 1
            continue
        path, source, n_ann = pick_ground_truth(mask_index.get(s, {"single": [], "MV": None, "ST": None}),
                                                SEG_CFG.consensus_tag)
        if path is None:
            skipped["no_ground_truth"] += 1
            continue
        src_counts[source] = src_counts.get(source, 0) + 1
        jobs.append((s, labels[s], path, source, n_ann))
    print(f"Eligible: {len(jobs)} | skipped: {skipped}")
    print(f"Ground-truth source: {src_counts}")
    print(f"Labels among eligible: {pd.Series([j[1] for j in jobs]).value_counts().to_dict()}")
    assert jobs, "No eligible images (check LABELS_FILE / LABEL_COLUMN / GT_DIR)"
    if args.limit:
        jobs = jobs[:args.limit]

    # 2) Resume logic: skip images that already have all methods in the CSV and all mask files
    for m in METHODS:
        (SEG_OUT_DIR / m).mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    old = pd.read_csv(METRICS_PATH) if METRICS_PATH.exists() else None
    if old is not None and not args.overwrite:
        done = old[old["status"].astype(str).str.startswith(("ok", "empty"))]
        counts = done.groupby("image")["method"].nunique()
        complete = set(counts[counts == len(METHODS)].index)
        complete = {s for s in complete
                    if all((SEG_OUT_DIR / m / f"{s}.png").exists() for m in METHODS)}
        n_before = len(jobs)
        jobs = [j for j in jobs if j[0] not in complete]
        print(f"Already done (skipped): {n_before - len(jobs)} | to process: {len(jobs)}")

    # 3) Segment and score
    rows = []
    if jobs:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for r in tqdm(ex.map(process_one, jobs, chunksize=4), total=len(jobs)):
                rows.extend(r)
    if rows:
        new = pd.DataFrame(rows)
        log = pd.concat([old, new]) if old is not None else new
        log = log.drop_duplicates(["image", "method"], keep="last")
        log.sort_values(["image", "method"]).to_csv(METRICS_PATH, index=False)

    # 4) Report and sanity checks
    log = pd.read_csv(METRICS_PATH)
    print(f"\nMetrics file: {len(log)} rows, {log['image'].nunique()} images")
    print("Status counts:\n" + log["status"].astype(str).str.split(":").str[0].value_counts().to_string())
    good = log[~log["status"].astype(str).str.startswith("error")]
    print("\nMean / median Dice and IoU by method and label (empty predictions count as 0):")
    summary = good.groupby(["method", "label"])[["dice", "iou"]].agg(["mean", "median", "count"]).round(3)
    print(summary.to_string())

    assert good[["dice", "iou"]].notna().all().all(), "NaN scores among non-error rows"
    assert good["dice"].between(0, 1).all() and good["iou"].between(0, 1).all(), "score outside [0, 1]"
    per_image = log.groupby("image")["method"].nunique()
    assert (per_image == len(METHODS)).all(), "some images are missing a method row"
    print("\nChecks passed: every image has one row per method, scores are within [0, 1]")


if __name__ == "__main__":  # required on Windows for multiprocessing
    main()