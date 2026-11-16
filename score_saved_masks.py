"""Re-score segmentation masks that are ALREADY saved on disk. No segmentation is re-run.

Reads   segmentation_5000_images/<method>/<ISIC_id>.png   (predicted masks, 0/255)
        segs/                                              (ground-truth masks)
        ima_metadata_*.csv                                 (benign / malignant labels)
Prints  mean / median Dice and IoU by method and label (same table as run_segmentation.py)
Writes  reports/segmentation_metrics_recomputed.csv        (one row per image and method)

Put this file in the project root (next to run_segmentation.py) and run:

    python score_saved_masks.py                       # recompute from the saved masks
    python score_saved_masks.py --csv                 # instant: only summarise the existing CSV
    python score_saved_masks.py --exclude ISIC_0000991 ISIC_0000647    # drop images with bad ground truth
    python score_saved_masks.py --seg-dir segmentation_v4 --out reports/my_scores.csv
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.config import GT_DIR, LABEL_COLUMN, LABELS_FILE, REPORTS_DIR, SEG_CFG, SEG_OUT_DIR
from src.ground_truth import index_masks, load_labels, pick_ground_truth, read_mask
from src.metrics import dice_iou
from src.segmentation import METHODS

try:
    from tqdm import tqdm
except ImportError:  # tqdm is optional
    def tqdm(it, **_):
        return it

EXISTING_CSV = REPORTS_DIR / "segmentation_metrics_v6.csv"
DEFAULT_OUT = REPORTS_DIR / "segmentation_metrics_recomputed.csv"


def score_one(job):
    """Score every available method for one image against its ground truth."""
    stem, label, gt_path, gt_source, n_ann, seg_dir, methods = job
    cv2.setNumThreads(1)
    base = {"image": stem, "label": label, "gt_source": gt_source, "n_annotators": n_ann}
    try:
        gt_raw = read_mask(gt_path)
        if gt_raw is None:
            raise ValueError("could not read ground-truth mask")
        if not gt_raw.any():
            raise ValueError("ground-truth mask is empty")
    except Exception as e:
        return [{**base, "method": m, "dice": np.nan, "iou": np.nan, "status": f"error: {e}"}
                for m in methods]

    rows = []
    for m in methods:
        path = Path(seg_dir) / m / f"{stem}.png"
        try:
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError("predicted mask file missing or unreadable")
            pred = img > 127
            H, W = pred.shape
            gt, gt_resized = gt_raw, False
            if gt.shape != (H, W):  # same rule as run_segmentation.py
                if abs((gt.shape[1] / gt.shape[0]) / (W / H) - 1) > 0.01:
                    raise ValueError(f"GT shape {gt.shape} vs mask {(H, W)}: aspect ratio differs")
                gt = cv2.resize(gt.astype(np.uint8), (W, H),
                                interpolation=cv2.INTER_NEAREST).astype(bool)
                gt_resized = True
            dice, iou = dice_iou(pred, gt)
            rows.append({**base, "method": m, "dice": round(dice, 5), "iou": round(iou, 5),
                         "status": "ok" if pred.any() else "empty",
                         "pred_area_frac": round(float(pred.mean()), 5),
                         "gt_area_frac": round(float(gt.mean()), 5),
                         "gt_resized": gt_resized, "height": H, "width": W})
        except Exception as e:
            rows.append({**base, "method": m, "dice": np.nan, "iou": np.nan,
                         "status": f"error: {e}"})
    return rows


def summarise(log, exclude=()):
    """Print the same summary tables as run_segmentation.py (+ a few useful extras)."""
    if exclude:
        ex = {e.upper() for e in exclude}
        log = log[~log["image"].str.upper().isin(ex)]
        print(f"Excluded {len(ex)} requested image id(s); {log['image'].nunique()} images remain.")

    status = log["status"].astype(str)
    good = log[~status.str.startswith("error")]
    print(f"\n{len(log)} rows, {log['image'].nunique()} images. Status counts:")
    print(status.str.split(":").str[0].value_counts().to_string())

    print("\nMean / median Dice and IoU by method and label (empty predictions count as 0):")
    table = good.groupby(["method", "label"])[["dice", "iou"]].agg(["mean", "median", "count"]).round(3)
    print(table.to_string())

    print("\nOverall (all labels):")
    print(good.groupby("method")[["dice", "iou"]].agg(["mean", "median"]).round(3).to_string())

    # extras
    print("\nImages with Dice == 0, and Dice < 0.5, by method:")
    extra = good.groupby("method")["dice"].agg(
        n=("count"), zero=lambda s: int((s == 0).sum()), below_half=lambda s: int((s < 0.5).sum()))
    extra["pct_zero"] = (100 * extra["zero"] / extra["n"]).round(1)
    print(extra.to_string())

    num = good["image"].str.extract(r"(\d+)$")[0].astype(float)
    sticker = num >= 558  # wide-field images with calibration stickers
    if sticker.any() and (~sticker).any():
        print("\nMean Dice, images without stickers (ISIC < 558) vs with stickers (ISIC >= 558):")
        grp = good.assign(group=np.where(sticker, "sticker (>=558)", "no sticker (<558)"))
        print(grp.groupby(["method", "group"])["dice"].agg(["mean", "median", "count"]).round(3).to_string())

    if good[["dice", "iou"]].notna().all().all():
        ok = good["dice"].between(0, 1).all() and good["iou"].between(0, 1).all()
        print("\nCheck: all scores within [0, 1]" if ok else "\nWARNING: score outside [0, 1]")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seg-dir", default=str(SEG_OUT_DIR), help="folder with <method>/<id>.png masks")
    ap.add_argument("--gt-dir", default=str(GT_DIR))
    ap.add_argument("--labels", default=str(LABELS_FILE))
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="where to write the recomputed CSV")
    ap.add_argument("--csv", nargs="?", const=str(EXISTING_CSV), default=None, metavar="PATH",
                    help="skip recomputation, just summarise an existing metrics CSV "
                         f"(default {EXISTING_CSV.name})")
    ap.add_argument("--exclude", nargs="*", default=[], help="image ids to leave out of the summary")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()

    # Fast path: just summarise a CSV that already exists
    if args.csv:
        print(f"Summarising existing file: {args.csv}")
        summarise(pd.read_csv(args.csv), args.exclude)
        return

    seg_dir = Path(args.seg_dir)
    methods = [m for m in METHODS if (seg_dir / m).is_dir()]
    assert methods, f"No method folders ({', '.join(METHODS)}) found inside {seg_dir}"
    print(f"Mask folder: {seg_dir} | methods found: {methods}")

    mask_index = index_masks(Path(args.gt_dir))
    labels = load_labels(Path(args.labels), LABEL_COLUMN)

    # images that have a saved mask for at least one method
    stems = sorted({p.stem for m in methods for p in (seg_dir / m).glob("*.png")})
    jobs, skipped = [], {"no_label": 0, "no_ground_truth": 0}
    for s in stems:
        if s not in labels:
            skipped["no_label"] += 1
            continue
        path, source, n_ann = pick_ground_truth(
            mask_index.get(s, {"single": [], "MV": None, "ST": None}), SEG_CFG.consensus_tag)
        if path is None:
            skipped["no_ground_truth"] += 1
            continue
        jobs.append((s, labels[s], path, source, n_ann, str(seg_dir), methods))
    print(f"Images with saved masks: {len(stems)} | scoring: {len(jobs)} | skipped: {skipped}")
    assert jobs, "Nothing to score (check --seg-dir, --gt-dir, --labels)"

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for r in tqdm(ex.map(score_one, jobs, chunksize=8), total=len(jobs)):
            rows.extend(r)

    log = pd.DataFrame(rows).sort_values(["image", "method"])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    log.to_csv(out, index=False)
    print(f"Wrote {len(log)} rows to {out}")
    summarise(log, args.exclude)


if __name__ == "__main__":  # required on Windows for multiprocessing
    main()