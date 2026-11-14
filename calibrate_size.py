"""Does growing / shrinking the predicted masks raise Dice?  (no re-segmentation needed)

    python calibrate_size.py                    # uses segmentation_v4/ensemble masks
    python calibrate_size.py --method otsu

Each saved prediction is dilated (f > 0) or eroded (f < 0) by  f * (equivalent radius of the mask),
so the change is relative to lesion size.  Scores are computed per lesion-size bucket.
To avoid tuning on the same images you report, the best f is chosen on the EVEN-numbered images and
checked on the ODD-numbered ones (and vice versa).
"""
import argparse

import cv2
import numpy as np
import pandas as pd

FS = (-0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.20, 0.30)
BUCKETS = [0, 0.05, 0.15, 0.35, 1.0]
LABELS = ["<5%", "5-15%", "15-35%", ">35%"]


def dice(a, b):
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else 2.0 * np.logical_and(a, b).sum() / s


def adjust(mask, f):
    """Grow (f>0) or shrink (f<0) a boolean mask by |f| * its equivalent radius."""
    area = mask.sum()
    if area == 0 or f == 0:
        return mask
    r = int(round(abs(f) * np.sqrt(area / np.pi)))
    if r < 1:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    m = mask.astype(np.uint8)
    return (cv2.dilate(m, k) if f > 0 else cv2.erode(m, k)) > 0


def score_pair(pred, gt, fs=FS):
    return [dice(adjust(pred, f), gt) for f in fs]


def summarize(rows, fs=FS):
    df = pd.DataFrame(rows, columns=["image", "gt_frac", *[f"f={f:+.2f}" for f in fs]])
    df["size"] = pd.cut(df["gt_frac"], BUCKETS, labels=LABELS, include_lowest=True)
    df["num"] = df["image"].str.extract(r"(\d+)")[0].astype(int)
    cols = [c for c in df.columns if c.startswith("f=")]
    print("Mean Dice by lesion-size bucket and growth factor f:")
    print(df.groupby("size", observed=True)[cols].mean().round(3).to_string())
    print("\nAll images:", df[cols].mean().round(4).to_dict())
    print("\nHold-out check (best f chosen on one half, scored on the other half):")
    for name, a, b in (("even->odd", df[df.num % 2 == 0], df[df.num % 2 == 1]),
                       ("odd->even", df[df.num % 2 == 1], df[df.num % 2 == 0])):
        best = a[cols].mean().idxmax()
        print(f"  {name}: best on training half = {best} | held-out Dice {b[best].mean():.4f} "
              f"vs f=+0.00 {b['f=+0.00'].mean():.4f}")
    print("\nBest f per size bucket (all images):")
    print(df.groupby("size", observed=True)[cols].mean().idxmax(axis=1).to_string())
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="ensemble")
    ap.add_argument("--max-side", type=int, default=512)
    args = ap.parse_args()
    from tqdm import tqdm
    from src.config import GT_DIR, REPORTS_DIR, SEG_CFG, SEG_OUT_DIR
    from src.ground_truth import index_masks, pick_ground_truth, read_mask

    idx = index_masks(GT_DIR)
    rows = []
    for p in tqdm(sorted((SEG_OUT_DIR / args.method).glob("*.png"))):
        entry = idx.get(p.stem.upper())
        if entry is None:
            continue
        gp = pick_ground_truth(entry, SEG_CFG.consensus_tag)[0]
        if gp is None:
            continue
        pred = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) > 127
        gt = read_mask(gp)
        s = args.max_side / max(pred.shape)
        size = (round(pred.shape[1] * s), round(pred.shape[0] * s))
        pred = cv2.resize(pred.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
        gt = cv2.resize(gt.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
        rows.append([p.stem, float(gt.mean()), *score_pair(pred, gt)])
    out = summarize(rows)
    out.to_csv(REPORTS_DIR / f"size_calibration_{args.method}.csv", index=False)


if __name__ == "__main__":
    main()