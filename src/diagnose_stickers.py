"""Why do the worst images fail?  Put this next to run_segmentation.py and run:

    python diagnose_stickers.py                       # sticker / no-sticker Dice split
    python diagnose_stickers.py --ids ISIC_0001474 ISIC_0001454 ISIC_0001464 ...

Part 1 splits the metrics CSV by ISIC number (your note: >557 have stickers).
Part 2 re-runs detect_stickers() on the given images with its built-in `debug` hook
and prints, for every coloured blob, WHY it was accepted or rejected
("small", "solidity", "not circular", "radius small", "disc not filled", ...).
It saves overlays to reports/sticker_debug/ using colours that are visible on red stickers:
    yellow  = what detect_stickers() masked      cyan    = final valid-area boundary
    green   = ground truth                       magenta = ensemble prediction (if the file exists)
"""
import argparse

import cv2
import numpy as np
import pandas as pd

from src.config import (GT_DIR, OUT_NO_CLAHE_DIR, REPORTS_DIR, SEG_CFG, SEG_OUT_DIR)
from src.ground_truth import index_masks, pick_ground_truth, read_mask
from src.segmentation import _shrink, detect_stickers, detect_vignette

STICKER_FROM = 558  # first ISIC number you believe has stickers


def split_report(csv_path):
    df = pd.read_csv(csv_path)
    df = df[~df["status"].astype(str).str.startswith("error")].copy()
    df["num"] = df["image"].str.extract(r"(\d+)").astype(int)
    df["group"] = np.where(df["num"] >= STICKER_FROM, "ISIC>=558 (stickers)", "ISIC<558")
    g = df.groupby(["method", "group"])["dice"].agg(["mean", "median", "count"]).round(3)
    g["frac<0.5"] = df.groupby(["method", "group"])["dice"].apply(lambda s: (s < 0.5).mean()).round(3)
    print(g.to_string())


def contour_overlay(img, mask, color):
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, cnts, -1, color, 1)


def debug_one(stem, gt_index):
    ref = cv2.imread(str(OUT_NO_CLAHE_DIR / f"{stem}.png"), cv2.IMREAD_COLOR)
    if ref is None:
        print(stem, ": preprocessed image not found")
        return
    ref = _shrink(ref, SEG_CFG.work_max_side)
    valid, vfrac, fov, art = detect_vignette(ref, SEG_CFG, return_masks=True)
    dbg = {}
    detect_stickers(ref, fov, SEG_CFG, debug=dbg)   # same call detect_vignette makes
    print(f"\n=== {stem}  vignette_frac={vfrac:.3f}  sticker_px={art.mean():.3f}  "
          f"skin a0,b0=({dbg.get('a0', 0):.1f},{dbg.get('b0', 0):.1f})  skin chroma={dbg.get('c_skin', 0):.1f}")
    comps = dbg.get("comps", [])
    if not comps:
        print("  no coloured candidate blobs at all -> colour thresholds (a_max, b_max, vivid_*, min_rel) too strict")
    for c in sorted(comps, key=lambda c: -c["area"]):
        print("  " + ", ".join(f"{k}={v}" for k, v in c.items()))

    vis = ref.copy()
    path = pick_ground_truth(gt_index.get(stem, {"single": [], "MV": None, "ST": None}),
                             SEG_CFG.consensus_tag)[0]
    if path is not None:
        gt = read_mask(path)
        gt = cv2.resize(gt.astype(np.uint8), (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_NEAREST)
        contour_overlay(vis, gt, (0, 255, 0))
    pred = cv2.imread(str(SEG_OUT_DIR / "ensemble" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
    if pred is not None:
        pred = cv2.resize(pred, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_NEAREST) > 127
        contour_overlay(vis, pred, (255, 0, 255))
    contour_overlay(vis, art, (0, 255, 255))
    contour_overlay(vis, ~valid, (255, 255, 0))
    out_dir = REPORTS_DIR / "sticker_debug"
    out_dir.mkdir(parents=True, exist_ok=True)
    side = np.hstack([ref, vis])
    cv2.imwrite(str(out_dir / f"{stem}.png"), cv2.resize(side, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(REPORTS_DIR / "segmentation_metrics_v6.csv"))
    ap.add_argument("--ids", nargs="*", default=[])
    args = ap.parse_args()
    split_report(args.csv)
    if args.ids:
        gt_index = index_masks(GT_DIR)
        for s in args.ids:
            debug_one(s.upper().replace("ISIC_", "ISIC_"), gt_index)
        print(f"\nOverlays saved in {REPORTS_DIR / 'sticker_debug'}")


if __name__ == "__main__":
    main()