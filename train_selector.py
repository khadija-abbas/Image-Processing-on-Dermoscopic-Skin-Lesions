"""Step 2: train the candidate selector with 5-fold CV (grouped by image), pick the best candidate per image using
OUT-OF-FOLD predictions (honest scores), write masks + metrics, and save a final model for new images.
    python train_selector.py
Outputs: segmentation_v4/selected/<id>.png, reports/segmentation_metrics_selected.csv, models/selector.joblib"""
import pickle
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from src.candidates import FN, SRC, build_X
from src.config import GT_DIR, REPORTS_DIR, SEG_CFG, SEG_OUT_DIR
from src.ground_truth import index_masks, pick_ground_truth, read_mask
from src.metrics import dice_iou
import argparse

TOPK, TEMP = 5, 0.05

def model():
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, max_leaf_nodes=24, min_samples_leaf=20,
                                         l2_regularization=1.0, random_state=0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()

    files = sorted(Path("candidates").glob("*.pkl"))
    files = files[:a.limit] if a.limit else files

    D = [pickle.load(open(f, "rb")) for f in files]
    D = [d for d in D if len(d["F"])]; print(len(D), "images")
    Xs, ys, gs = [], [], []
    for i, d in enumerate(D):
        h, w = d["shape"]; M = [np.unpackbits(m)[:h * w].reshape(h, w).astype(bool) for m in d["masks"]]
        d["M"] = M; Xs.append(build_X(d["F"], M)); ys.append(d["y"]); gs += [i] * len(d["y"])
    X, y, g = np.vstack(Xs), np.concatenate(ys), np.array(gs)
    pred = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, g):
        pred[te] = model().fit(X[tr], y[tr]).predict(X[te])
    (SEG_OUT_DIR / "selected").mkdir(parents=True, exist_ok=True); REPORTS_DIR.mkdir(exist_ok=True)
    idx = index_masks(GT_DIR); rows, wk = [], []
    for i, d in enumerate(D):
        # s = np.where(g == i)[0]; k = int(np.argmax(pred[s]))
        # H, W = d["full_shape"]; m = cv2.resize(d["M"][k].astype(np.uint8) * 255, (W, H), interpolation=cv2.INTER_LINEAR) > 127

        s = np.where(g == i)[0]; o = np.argsort(-pred[s])[:TOPK]; k = int(o[0])
        wt = np.exp((pred[s][o] - pred[s][o].max()) / TEMP); wt /= wt.sum()          # soft vote of the top-K candidates
        fused = sum(w_ * d["M"][j].astype(np.float32) for w_, j in zip(wt, o)) >= 0.5
        if not fused.any(): fused = d["M"][k]
        H, W = d["full_shape"]; m = cv2.resize(fused.astype(np.uint8) * 255, (W, H), interpolation=cv2.INTER_LINEAR) > 127


        
        cv2.imwrite(str(SEG_OUT_DIR / "selected" / f"{d['stem']}.png"), m.astype(np.uint8) * 255)
        gt = read_mask(pick_ground_truth(idx[d["stem"]], SEG_CFG.consensus_tag)[0])
        if gt.shape != (H, W): gt = cv2.resize(gt.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
        dc, io = dice_iou(m, gt)
        rows.append(dict(image=d["stem"], label=d["label"], method="selected", dice=dc, iou=io, picked=d["names"][k]))
        ens = [j for j, n in enumerate(d["names"]) if n == "ensemble"]
        wk.append(dict(label=d["label"], ensemble_workres=d["y"][ens[0]] if ens else 0.0, selected_workres=d["y"][k],
                       oracle_workres=d["y"].max()))
    r = pd.DataFrame(rows); r.to_csv(REPORTS_DIR / "segmentation_metrics_selected.csv", index=False)
    print("\nFULL-RESOLUTION, out-of-fold:"); print(r.groupby("label")[["dice", "iou"]].agg(["mean", "median", "count"]).round(3).to_string())
    print("overall mean dice %.4f  iou %.4f" % (r.dice.mean(), r.iou.mean()))
    print("\nworking-res comparison (same images):"); print(pd.DataFrame(wk).drop(columns="label").mean().round(4).to_string())
    Path("models").mkdir(exist_ok=True); joblib.dump(model().fit(X, y), "models/selector.joblib"); print("saved models/selector.joblib")

if __name__ == "__main__":
    main()