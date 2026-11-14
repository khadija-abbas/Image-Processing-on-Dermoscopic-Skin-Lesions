"""Where does segmentation fail? Reads reports/segmentation_metrics_v3.csv (no images needed).

For every method and label it prints how many images fall into each failure type, using the
predicted / ground-truth area ratio that run_segmentation.py already stores:
  empty       prediction is empty
  over-seg    prediction >= 1.5x the GT area  (flooded, leaked onto skin / vignette / ruler)
  under-seg   prediction <= 0.6x the GT area  (only the dark core, or one lobe of the lesion)
  displaced   area is similar but Dice < 0.6  (wrong object: ink, hairs, a second lesion)
Then it lists the worst images for the ensemble and the effect of removing the 'big' lesions.

Usage:  python analyze_failures.py [--csv PATH] [--worst 15] [--method ensemble]"""
import argparse

import numpy as np
import pandas as pd

from src.config import REPORTS_DIR


def categorize(r):
    if str(r["status"]).startswith("empty") or r["pred_area_frac"] == 0:
        return "empty"
    ratio = r["pred_area_frac"] / max(r["gt_area_frac"], 1e-9)
    if r["dice"] >= 0.8:
        return "good"
    if ratio >= 1.5:
        return "over-seg"
    if ratio <= 0.6:
        return "under-seg"
    return "displaced" if r["dice"] < 0.6 else "boundary-off"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(REPORTS_DIR / "segmentation_metrics_v6.csv"))
    ap.add_argument("--worst", type=int, default=15)
    ap.add_argument("--method", default="ensemble")
    args = ap.parse_args()

    log = pd.read_csv(args.csv)
    log = log[~log["status"].astype(str).str.startswith("error")].copy()
    log["failure"] = log.apply(categorize, axis=1)

    print("Failure types per method and label (counts):")
    print(pd.crosstab([log["method"], log["label"]], log["failure"]).to_string())

    print("\nShare of images with Dice < 0.5 (the tail that drags the mean down):")
    tail = log.assign(bad=log["dice"] < 0.5).groupby(["method", "label"])["bad"].agg(["mean", "sum"])
    print(tail.round(3).to_string())

    print("\nMean Dice by GT lesion size (fraction of the image):")
    log["gt_size"] = pd.cut(log["gt_area_frac"], [0, 0.05, 0.15, 0.35, 1.0],
                            labels=["<5%", "5-15%", "15-35%", ">35%"])
    print(log.pivot_table(index="gt_size", columns=["method"], values="dice", aggfunc="mean",
                          observed=True).round(3).to_string())
    print("\nImages per size bucket and label:")
    print(pd.crosstab(log[log["method"] == args.method]["gt_size"],
                      log[log["method"] == args.method]["label"]).to_string())

    if "vignette_frac" in log:
        log["has_vignette"] = log["vignette_frac"] > 0.02
        print("\nMean Dice with / without a black vignette:")
        print(log.pivot_table(index="has_vignette", columns="method", values="dice",
                              aggfunc="mean").round(3).to_string())

    worst = log[log["method"] == args.method].sort_values("dice").head(args.worst)
    cols = ["image", "label", "dice", "pred_area_frac", "gt_area_frac", "vignette_frac", "failure"]
    print(f"\nWorst {args.worst} images for '{args.method}':")
    print(worst[[c for c in cols if c in worst]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()