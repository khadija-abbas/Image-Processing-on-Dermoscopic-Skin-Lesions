"""Compare two segmentation metrics CSVs (old vs new) and explain regressions.

    python compare_runs.py reports/segmentation_metrics_v6_before.csv reports/segmentation_metrics_v6.csv
"""
import sys

import numpy as np
import pandas as pd

old_path, new_path = sys.argv[1], sys.argv[2]
keep = ["image", "method", "label", "dice", "pred_area_frac", "gt_area_frac", "vignette_frac"]
old = pd.read_csv(old_path)[keep].drop(columns="label")
old = old.rename(columns=lambda c: c if c in ("image", "method") else c + "_old")
new = pd.read_csv(new_path)[keep].rename(columns=lambda c: c if c in ("image", "method", "label") else c + "_new")
d = old.merge(new, on=["image", "method"])
d["num"] = d["image"].str.extract(r"(\d+)")[0].astype(int)
d["group"] = np.where(d["num"] >= 558, "ISIC>=558 (stickers)", "ISIC<558")
d["delta"] = d["dice_new"] - d["dice_old"]
d["dvig"] = d["vignette_frac_new"] - d["vignette_frac_old"]
print(f"{d['image'].nunique()} images in both files\n")

print("Mean Dice / share below 0.5, old -> new")
grp = d.groupby(["method", "group"])
g = grp.agg(old=("dice_old", "mean"), new=("dice_new", "mean"), n=("image", "count"))
g["below0.5_old"] = grp["dice_old"].apply(lambda s: (s < 0.5).mean())
g["below0.5_new"] = grp["dice_new"].apply(lambda s: (s < 0.5).mean())
print(g.round(3).to_string())

print("\nMean Dice by diagnosis label, old -> new")
gl = d.groupby(["method", "label"]).agg(old=("dice_old", "mean"), new=("dice_new", "mean"), n=("image", "count"))
print(gl.round(3).to_string())

e = d[d["method"] == "ensemble"].copy()
e["cause"] = np.where(e["dvig"].abs() > 0.03, "sticker/vignette mask changed", "same mask -> component selection or growth")
print(f"\nEnsemble: improved by >0.2: {(e.delta > 0.2).sum()} | worsened by >0.2: {(e.delta < -0.2).sum()}")
worse = e[e.delta < -0.2]
print("Likely cause of the worsened images:\n" + worse["cause"].value_counts().to_string())
print("...by label:\n" + worse["label"].value_counts().to_string())

show = ["image", "label", "dice_old", "dice_new", "pred_area_frac_old", "pred_area_frac_new",
        "gt_area_frac_new", "vignette_frac_old", "vignette_frac_new"]
print("\n15 biggest REGRESSIONS (ensemble):")
print(e.sort_values("delta").head(15)[show].round(3).to_string(index=False))
mal = e[e.label == "malignant"].sort_values("delta").head(10)
print("\n10 biggest regressions among MALIGNANT:")
print(mal[show].round(3).to_string(index=False))
print("\n15 worst images now:")
print(e.sort_values("dice_new").head(15)[show].round(3).to_string(index=False))