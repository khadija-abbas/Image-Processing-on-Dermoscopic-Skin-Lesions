"""Find out why scores are low. Reads reports/segmentation_metrics.csv and the saved masks
(no re-segmentation). It
  1) summarises Dice by ID range, lesion size, vignette, ground-truth source and predicted size
  2) checks that the CSV agrees with the saved masks
  3) tests whether low-scoring images have a ground truth that is flipped or rotated
     relative to the image.
"""
import argparse

import cv2
import numpy as np
import pandas as pd

from src.config import GT_DIR, REPORTS_DIR, SEG_CFG, SEG_OUT_DIR
from src.ground_truth import index_masks, pick_ground_truth, read_mask
from src.metrics import dice_iou
from src.segmentation import METHODS

METRICS_PATH = REPORTS_DIR / "segmentation_metrics.csv"
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)


def load_pred(method, stem):
    img = cv2.imread(str(SEG_OUT_DIR / method / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
    return None if img is None else img > 127


def load_gt(stem, mask_index, shape):
    """Ground truth for an image, resized to `shape` if needed (same rule as run_segmentation)."""
    path, _, _ = pick_ground_truth(mask_index[stem], SEG_CFG.consensus_tag)
    gt = read_mask(path)
    if gt is None:
        return None
    if gt.shape != shape:
        h, w = shape
        if abs((gt.shape[1] / gt.shape[0]) / (w / h) - 1) > 0.01:
            return None
        gt = cv2.resize(gt.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    return gt


def table(df, by, title):
    med = df.pivot_table(index=by, columns="method", values="dice", aggfunc="median", observed=True).round(3)
    n = df[df["method"] == METHODS[0]].groupby(by, observed=True).size().rename("n_images")
    print(f"\n{title} (median Dice)")
    print(med.join(n).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--align-n", type=int, default=150, help="how many of the lowest-scoring images to test")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    log = pd.read_csv(METRICS_PATH)
    status = log["status"].astype(str)
    print(f"Rows: {len(log)} | images: {log['image'].nunique()}")
    bad = log[status.str.startswith(("error", "empty"))]
    print(f"\nRows with error / empty status: {len(bad)}")
    if len(bad):
        print(bad[["image", "method", "status"]].head(40).to_string(index=False))

    # ---------------------------------------------------------- 1) breakdowns
    d = log[~status.str.startswith("error")].copy()
    d["id_num"] = d["image"].str.extract(r"(\d+)", expand=False).astype(int)
    d["id_range"] = pd.cut(d["id_num"], [-1, 99, 499, 999, 1499, 1999, 10**9],
                           labels=["0-99", "100-499", "500-999", "1000-1499", "1500-1999", "2000+"])
    d["lesion_size"] = pd.cut(d["gt_area_frac"], [0, 0.01, 0.03, 0.06, 0.12, 0.25, 1.0],
                              labels=["<1%", "1-3%", "3-6%", "6-12%", "12-25%", ">25%"],
                              include_lowest=True)
    d["has_vignette"] = np.where(d["vignette_frac"] > 0.05, "vignette", "no vignette")
    d["pred_over_gt"] = d["pred_area_frac"] / d["gt_area_frac"].clip(lower=1e-6)

    table(d, "label", "By label")
    table(d, "id_range", "By image ID range")
    table(d, "lesion_size", "By lesion size (share of the image covered by the ground-truth mask)")
    one = d[d["method"] == METHODS[0]]
    print("\nNumber of images per lesion-size bucket and label:")
    print(pd.crosstab(one["lesion_size"], one["label"]).to_string())
    print("\nMedian lesion size (fraction of image) by label:")
    print(one.groupby("label")["gt_area_frac"].median().round(4).to_string())
    table(d, "has_vignette", "By vignette")
    table(d, "gt_source", "By ground-truth source")
    print("\nMedian predicted area / ground-truth area (1 = right size, >>1 = blob, <<1 = too small):")
    print(d.pivot_table(index="lesion_size", columns="method", values="pred_over_gt",
                        aggfunc="median", observed=True).round(2).to_string())
    print("\nShare of images with Dice < 0.05 (essentially no overlap):")
    print(d.assign(near_zero=d["dice"] < 0.05).groupby(["method", "label"])["near_zero"]
          .mean().round(3).unstack("label").to_string())

    # ---------------------------------------------------------- 2) CSV vs saved masks
    mask_index = index_masks(GT_DIR)
    sample = d[d["status"].astype(str).isin(["ok", "empty"])]
    sample = sample.sample(min(30, len(sample)), random_state=args.seed)
    diffs = []
    for _, row in sample.iterrows():
        pred = load_pred(row["method"], row["image"])
        gt = None if pred is None else load_gt(row["image"], mask_index, pred.shape)
        if gt is not None:
            diffs.append(abs(dice_iou(pred, gt)[0] - row["dice"]))
    print("\nCheck: Dice recomputed from the saved masks vs the CSV")
    print(f"  {len(diffs)} rows checked, largest difference = {max(diffs):.4f}" if diffs
          else "  nothing could be checked")

    # ---------------------------------------------------------- 3) misalignment test
    wide = d.pivot(index="image", columns="method", values="dice").dropna()
    worst = wide.mean(axis=1).sort_values().head(args.align_n).index
    transforms = {"identity": lambda g: g,
                  "rotate 180": lambda g: g[::-1, ::-1],
                  "flip horizontal": lambda g: g[:, ::-1],
                  "flip vertical": lambda g: g[::-1, :]}
    rows = []
    for stem in worst:
        preds = {m: load_pred(m, stem) for m in METHODS}
        if any(p is None for p in preds.values()):
            continue
        gt = load_gt(stem, mask_index, preds[METHODS[0]].shape)
        if gt is None:
            continue
        scores = {t: float(np.mean([dice_iou(p, f(gt))[0] for p in preds.values()]))
                  for t, f in transforms.items()}
        rows.append({"image": stem, **scores})
    res = pd.DataFrame(rows)
    print(f"\nMisalignment test on the {len(res)} lowest-scoring images "
          "(mean Dice over the 3 methods, with the ground truth flipped / rotated):")
    if len(res):
        names = list(transforms)
        best = res[names].idxmax(axis=1)
        gain = res[names].max(axis=1) - res["identity"]
        flagged = res[(best != "identity") & (gain > 0.3)]
        print(res[names].mean().round(3).to_string())
        print(f"Images where a flipped/rotated ground truth fits clearly better (gain > 0.3): "
              f"{len(flagged)} of {len(res)}")
        if len(flagged):
            print(flagged.head(15).round(3).to_string(index=False))


if __name__ == "__main__":
    main()