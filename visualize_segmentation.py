"""Qualitative check: ground-truth contour (green) vs predicted contour (red) per method.
Each label is printed INSIDE the image it describes."""
import argparse
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import GT_DIR, IMAGE_EXTENSIONS, RAW_DIR, REPORTS_DIR, SEG_CFG, SEG_OUT_DIR
from src.ground_truth import index_masks, pick_ground_truth, read_mask
from src.segmentation import METHODS

METRICS_PATH = REPORTS_DIR / "segmentation_metrics.csv"
TITLES = {"otsu": "Otsu", "grabcut": "GrabCut", "active_contour": "Active contour"}


def overlay(bgr, gt, pred=None, max_side=512):
    """Draw GT (green) and optionally the prediction (red) contours on a downscaled image."""
    h, w = bgr.shape[:2]
    s = min(1.0, max_side / max(h, w))
    size = (max(1, round(w * s)), max(1, round(h * s)))
    img = cv2.resize(bgr, size, interpolation=cv2.INTER_AREA)
    for mask, color in ((gt, (0, 255, 0)), (pred, (0, 0, 255))):
        if mask is None:
            continue
        m = cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, contours, -1, color, 2)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def tag(ax, text):
    """Label drawn inside the axes, so it can never be mistaken for the neighbouring image's label."""
    ax.text(0.01, 0.99, text, transform=ax.transAxes, ha="left", va="top", fontsize=8,
            bbox=dict(boxstyle="square,pad=0.25", facecolor="white", alpha=0.85, edgecolor="none"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--label", choices=["any", "benign", "malignant"], default="any")
    parser.add_argument("--mode", choices=["random", "worst"], default="random",
                        help="'worst' shows the images with the lowest mean Dice across methods")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    log = pd.read_csv(METRICS_PATH)
    log = log[log["status"].astype(str).str.startswith(("ok", "empty"))]
    if args.label != "any":
        log = log[log["label"] == args.label]
    wide = log.pivot(index="image", columns="method", values="dice").dropna()
    assert len(wide), "No matching images in the metrics file"
    if args.mode == "worst":
        stems = wide.mean(axis=1).sort_values().head(args.n).index.tolist()
    else:
        stems = random.Random(args.seed).sample(sorted(wide.index), min(args.n, len(wide)))

    raw = {p.stem: p for p in RAW_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS}
    mask_index = index_masks(GT_DIR)

    fig, axes = plt.subplots(len(stems), 4, figsize=(15, 3.4 * len(stems)), squeeze=False)
    for r, stem in enumerate(stems):
        img = cv2.imread(str(raw[stem]), cv2.IMREAD_COLOR)
        gt_path, source, _ = pick_ground_truth(mask_index[stem], SEG_CFG.consensus_tag)
        gt = read_mask(gt_path)
        if gt.shape != img.shape[:2]:
            gt = cv2.resize(gt.astype(np.uint8), (img.shape[1], img.shape[0]),
                            interpolation=cv2.INTER_NEAREST).astype(bool)
        label = log[log["image"] == stem]["label"].iloc[0]
        gt_pct = 100 * gt.mean()
        axes[r, 0].imshow(overlay(img, gt))
        tag(axes[r, 0], f"{stem} | {label} | GT: {source} | lesion {gt_pct:.1f}% of image")
        for c, method in enumerate(METHODS, start=1):
            pred = cv2.imread(str(SEG_OUT_DIR / method / f"{stem}.png"), cv2.IMREAD_GRAYSCALE) > 127
            axes[r, c].imshow(overlay(img, gt, pred))
            tag(axes[r, c], f"{TITLES[method]}: Dice {wide.loc[stem, method]:.2f}")
        for c in range(4):
            axes[r, c].axis("off")
    plt.tight_layout()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / f"segmentation_preview_{args.label}_{args.mode}.png"
    fig.savefig(out, dpi=110)
    print(f"Saved preview to {out}")
    plt.show()


if __name__ == "__main__":
    main()