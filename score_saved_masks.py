# """Re-score segmentation results that are ALREADY on disk. No segmentation or training is re-run.

# Works on three kinds of input
#   A) saved masks   segmentation_5000_images/<method>/<ISIC_id>.png   (otsu, grabcut, active_contour,
#                    ensemble, and `selected` from train_selector.py) scored against the ground truth in segs/
#   B) a metrics CSV reports/segmentation_metrics_v6.csv        (run_segmentation.py, many methods)
#                    reports/segmentation_metrics_selected.csv  (train_selector.py: image,label,method,dice,iou,picked)
#   C) candidates/*.pkl  (from build_candidates.py) for the working-resolution ensemble / oracle numbers

# Usage (run from the project root, next to run_segmentation.py)

#     python score_saved_masks.py --selected        # summarise reports/segmentation_metrics_selected.csv
#     python score_saved_masks.py --selected --candidates     # ... plus the working-res comparison block
#     python score_saved_masks.py                   # recompute everything from the saved masks (incl. selected/)
#     python score_saved_masks.py --csv             # summarise reports/segmentation_metrics_v6.csv
#     python score_saved_masks.py --csv some/other.csv
#     python score_saved_masks.py --selected --exclude ISIC_0000991 ISIC_0000647   # drop bad-GT images

# Output for the `selected` method matches train_selector.py:
#     FULL-RESOLUTION, out-of-fold:  mean / median / count of Dice and IoU by label, plus overall means.
# The `selected_workres` line of train_selector.py needs the model's out-of-fold predictions, so it cannot be
# rebuilt from files; ensemble_workres and oracle_workres can (--candidates).
# """
# import argparse
# import os
# import pickle
# from concurrent.futures import ProcessPoolExecutor
# from pathlib import Path

# import cv2
# import numpy as np
# import pandas as pd

# from src.config import GT_DIR, LABEL_COLUMN, LABELS_FILE, REPORTS_DIR, SEG_CFG, SEG_OUT_DIR
# from src.ground_truth import index_masks, load_labels, pick_ground_truth, read_mask
# from src.metrics import dice_iou
# from src.segmentation import METHODS

# try:
#     from tqdm import tqdm
# except ImportError:  # tqdm is optional
#     def tqdm(it, **_):
#         return it

# EXISTING_CSV = REPORTS_DIR / "segmentation_metrics_v6.csv"
# SELECTED_CSV = REPORTS_DIR / "segmentation_metrics_selected.csv"
# DEFAULT_OUT = REPORTS_DIR / "segmentation_metrics_recomputed.csv"
# SELECTED = "selected"


# # ------------------------------------------------------------------ scoring saved masks
# def score_one(job):
#     """Score every available method for one image against its ground truth."""
#     stem, label, gt_path, gt_source, n_ann, seg_dir, methods = job
#     cv2.setNumThreads(1)
#     base = {"image": stem, "label": label, "gt_source": gt_source, "n_annotators": n_ann}
#     try:
#         gt_raw = read_mask(gt_path)
#         if gt_raw is None:
#             raise ValueError("could not read ground-truth mask")
#         if not gt_raw.any():
#             raise ValueError("ground-truth mask is empty")
#     except Exception as e:
#         return [{**base, "method": m, "dice": np.nan, "iou": np.nan, "status": f"error: {e}"}
#                 for m in methods]

#     rows = []
#     for m in methods:
#         path = Path(seg_dir) / m / f"{stem}.png"
#         try:
#             img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
#             if img is None:
#                 raise ValueError("predicted mask file missing or unreadable")
#             pred = img > 127
#             H, W = pred.shape
#             gt, gt_resized = gt_raw, False
#             if gt.shape != (H, W):  # same rule as run_segmentation.py
#                 if abs((gt.shape[1] / gt.shape[0]) / (W / H) - 1) > 0.01:
#                     raise ValueError(f"GT shape {gt.shape} vs mask {(H, W)}: aspect ratio differs")
#                 gt = cv2.resize(gt.astype(np.uint8), (W, H),
#                                 interpolation=cv2.INTER_NEAREST).astype(bool)
#                 gt_resized = True
#             dice, iou = dice_iou(pred, gt)
#             rows.append({**base, "method": m, "dice": round(dice, 5), "iou": round(iou, 5),
#                          "status": "ok" if pred.any() else "empty",
#                          "pred_area_frac": round(float(pred.mean()), 5),
#                          "gt_area_frac": round(float(gt.mean()), 5),
#                          "gt_resized": gt_resized, "height": H, "width": W})
#         except Exception as e:
#             rows.append({**base, "method": m, "dice": np.nan, "iou": np.nan,
#                          "status": f"error: {e}"})
#     return rows


# # ------------------------------------------------------------------ summaries
# def selector_block(sel):
#     """Same layout as train_selector.py's 'FULL-RESOLUTION, out-of-fold' printout."""
#     print("\nFULL-RESOLUTION, out-of-fold:")
#     print(sel.groupby("label")[["dice", "iou"]].agg(["mean", "median", "count"]).round(3).to_string())
#     print("overall mean dice %.4f  iou %.4f" % (sel["dice"].mean(), sel["iou"].mean()))
#     if "picked" in sel.columns and sel["picked"].notna().any():
#         print("\nWhich candidate the selector ranked first (top-1) and how its final mask scored:")
#         pk = sel.groupby("picked")["dice"].agg(count="count", mean_dice="mean", median_dice="median")
#         pk["share_%"] = (100 * pk["count"] / pk["count"].sum()).round(1)
#         print(pk.sort_values("count", ascending=False).round(3).to_string())


# def working_res(cand_dir, images=None):
#     """ensemble_workres / oracle_workres from candidates/*.pkl (no model needed)."""
#     files = sorted(Path(cand_dir).glob("*.pkl"))
#     if not files:
#         print(f"\nNo candidate files found in {cand_dir}/ (run build_candidates.py first).")
#         return
#     rows = []
#     for f in files:
#         if images is not None and f.stem not in images:
#             continue
#         with open(f, "rb") as fh:
#             d = pickle.load(fh)
#         if not len(d["F"]):
#             continue
#         ens = [j for j, n in enumerate(d["names"]) if n == "ensemble"]
#         rows.append({"ensemble_workres": float(d["y"][ens[0]]) if ens else 0.0,
#                      "oracle_workres": float(np.max(d["y"]))})
#     if not rows:
#         print("\nNo matching candidate files for the scored images.")
#         return
#     print(f"\nworking-res comparison ({len(rows)} images; from {cand_dir}/):")
#     print(pd.DataFrame(rows).mean().round(4).to_string())
#     print("(selected_workres is not shown: it needs the model's out-of-fold predictions, "
#           "so it is only printed by train_selector.py)")


# def summarise(log, exclude=(), candidates=None):
#     log = log.copy()
#     if exclude:
#         ex = {e.upper() for e in exclude}
#         log = log[~log["image"].str.upper().isin(ex)]
#         print(f"Excluded {len(ex)} requested image id(s); {log['image'].nunique()} images remain.")

#     if "status" not in log.columns:
#         log["status"] = "ok"
#     status = log["status"].astype(str)
#     good = log[~status.str.startswith("error")]
#     print(f"\n{len(log)} rows, {log['image'].nunique()} images. Status counts:")
#     print(status.str.split(":").str[0].value_counts().to_string())

#     sel = good[good["method"] == SELECTED]
#     others = good[good["method"] != SELECTED]

#     if len(sel):
#         selector_block(sel)
#     if len(others):
#         print("\nMean / median Dice and IoU by method and label (empty predictions count as 0):")
#         print(others.groupby(["method", "label"])[["dice", "iou"]]
#               .agg(["mean", "median", "count"]).round(3).to_string())

#     print("\nOverall (all labels):")
#     print(good.groupby("method")[["dice", "iou"]].agg(["mean", "median"]).round(3).to_string())

#     print("\nImages with Dice == 0, and Dice < 0.5, by method:")
#     extra = good.groupby("method")["dice"].agg(
#         n="count", zero=lambda s: int((s == 0).sum()), below_half=lambda s: int((s < 0.5).sum()))
#     extra["pct_zero"] = (100 * extra["zero"] / extra["n"]).round(1)
#     print(extra.to_string())

#     num = good["image"].str.extract(r"(\d+)$")[0].astype(float)
#     sticker = num >= 558  # wide-field images with calibration stickers
#     if sticker.any() and (~sticker).any():
#         print("\nMean Dice, images without stickers (ISIC < 558) vs with stickers (ISIC >= 558):")
#         grp = good.assign(group=np.where(sticker, "sticker (>=558)", "no sticker (<558)"))
#         print(grp.groupby(["method", "group"])["dice"].agg(["mean", "median", "count"]).round(3).to_string())

#     if good[["dice", "iou"]].notna().all().all():
#         ok = good["dice"].between(0, 1).all() and good["iou"].between(0, 1).all()
#         print("\nCheck: all scores within [0, 1]" if ok else "\nWARNING: score outside [0, 1]")

#     if candidates:
#         working_res(candidates, set(sel["image"]) if len(sel) else None)


# # ------------------------------------------------------------------ main
# def main():
#     ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
#     ap.add_argument("--seg-dir", default=str(SEG_OUT_DIR), help="folder with <method>/<id>.png masks")
#     ap.add_argument("--gt-dir", default=str(GT_DIR))
#     ap.add_argument("--labels", default=str(LABELS_FILE))
#     ap.add_argument("--out", default=str(DEFAULT_OUT), help="where to write the recomputed CSV")
#     ap.add_argument("--csv", nargs="?", const=str(EXISTING_CSV), default=None, metavar="PATH",
#                     help=f"summarise an existing metrics CSV instead of recomputing (default {EXISTING_CSV.name})")
#     ap.add_argument("--selected", action="store_true",
#                     help=f"shortcut for --csv {SELECTED_CSV.as_posix()} (the train_selector.py results)")
#     ap.add_argument("--candidates", nargs="?", const="candidates", default=None, metavar="DIR",
#                     help="also print ensemble/oracle working-resolution means from candidates/*.pkl")
#     ap.add_argument("--exclude", nargs="*", default=[], help="image ids to leave out of the summary")
#     ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
#     args = ap.parse_args()

#     # Fast path: summarise a CSV that already exists
#     csv_path = str(SELECTED_CSV) if args.selected else args.csv
#     if csv_path:
#         print(f"Summarising existing file: {csv_path}")
#         summarise(pd.read_csv(csv_path), args.exclude, args.candidates)
#         return

#     # Recompute from saved masks
#     seg_dir = Path(args.seg_dir)
#     methods = [m for m in (*METHODS, SELECTED) if (seg_dir / m).is_dir()]
#     assert methods, f"No method folders ({', '.join(METHODS)}, {SELECTED}) found inside {seg_dir}"
#     print(f"Mask folder: {seg_dir} | methods found: {methods}")

#     mask_index = index_masks(Path(args.gt_dir))
#     labels = load_labels(Path(args.labels), LABEL_COLUMN)

#     stems = sorted({p.stem for m in methods for p in (seg_dir / m).glob("*.png")})
#     jobs, skipped = [], {"no_label": 0, "no_ground_truth": 0}
#     for s in stems:
#         if s not in labels:
#             skipped["no_label"] += 1
#             continue
#         path, source, n_ann = pick_ground_truth(
#             mask_index.get(s, {"single": [], "MV": None, "ST": None}), SEG_CFG.consensus_tag)
#         if path is None:
#             skipped["no_ground_truth"] += 1
#             continue
#         jobs.append((s, labels[s], path, source, n_ann, str(seg_dir), methods))
#     print(f"Images with saved masks: {len(stems)} | scoring: {len(jobs)} | skipped: {skipped}")
#     assert jobs, "Nothing to score (check --seg-dir, --gt-dir, --labels)"

#     rows = []
#     with ProcessPoolExecutor(max_workers=args.workers) as ex:
#         for r in tqdm(ex.map(score_one, jobs, chunksize=8), total=len(jobs)):
#             rows.extend(r)
#     log = pd.DataFrame(rows).sort_values(["image", "method"])

#     # attach the 'picked' candidate name for the selector rows, if train_selector.py's CSV exists
#     if SELECTED_CSV.exists():
#         pk = pd.read_csv(SELECTED_CSV)[["image", "picked"]].drop_duplicates("image")
#         log = log.merge(pk.assign(method=SELECTED), on=["image", "method"], how="left")

#     out = Path(args.out)
#     out.parent.mkdir(parents=True, exist_ok=True)
#     log.to_csv(out, index=False)
#     print(f"Wrote {len(log)} rows to {out}")
#     summarise(log, args.exclude, args.candidates)


# if __name__ == "__main__":  # required on Windows for multiprocessing
#     main()

"""Re-score segmentation results that are ALREADY on disk. No segmentation or training is re-run.

Works on three kinds of input
  A) saved masks   segmentation_5000_images/<method>/<ISIC_id>.png   (otsu, grabcut, active_contour,
                   ensemble, and `selected` from train_selector.py) scored against the ground truth in segs/
  B) a metrics CSV reports/segmentation_metrics_v6.csv        (run_segmentation.py, many methods)
                   reports/segmentation_metrics_selected.csv  (train_selector.py: image,label,method,dice,iou,picked)
  C) candidates/*.pkl  (from build_candidates.py) to rebuild the 'working-res comparison' block exactly

Usage (run from the project root, next to run_segmentation.py)

    python score_saved_masks.py --selected        # summarise reports/segmentation_metrics_selected.csv
    python score_saved_masks.py --selected --workres        # the full train_selector.py printout
    python score_saved_masks.py                   # recompute everything from the saved masks (incl. selected/)
    python score_saved_masks.py --csv             # summarise reports/segmentation_metrics_v6.csv
    python score_saved_masks.py --csv some/other.csv
    python score_saved_masks.py --selected --exclude ISIC_0000991 ISIC_0000647   # drop bad-GT images

`--selected --workres` reproduces the whole train_selector.py printout:
    FULL-RESOLUTION, out-of-fold  (from the CSV: mean / median / count of Dice and IoU by label, overall means)
    working-res comparison        (ensemble_workres / selected_workres / oracle_workres, from candidates/*.pkl).
The working-res block cannot come from the CSV alone: selected_workres is the working-resolution Dice of the
top-ranked candidate, and the CSV's `picked` column only has its type ("sweep" appears many times per image).
So --workres re-runs the same 5-fold CV on candidates/*.pkl (nothing is written; src/candidates.py and
train_selector.py must be in the project, as they are).
"""
import argparse
import os
import pickle
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
SELECTED_CSV = REPORTS_DIR / "segmentation_metrics_selected.csv"
DEFAULT_OUT = REPORTS_DIR / "segmentation_metrics_recomputed.csv"
SELECTED = "selected"


# ------------------------------------------------------------------ scoring saved masks
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


# ------------------------------------------------------------------ summaries
def selector_block(sel):
    """Same layout as train_selector.py's 'FULL-RESOLUTION, out-of-fold' printout."""
    print(f"\n{sel['image'].nunique()} images")
    print("\nFULL-RESOLUTION, out-of-fold:")
    print(sel.groupby("label")[["dice", "iou"]].agg(["mean", "median", "count"]).round(3).to_string())
    print("overall mean dice %.4f  iou %.4f" % (sel["dice"].mean(), sel["iou"].mean()))


def picked_table(sel):
    if "picked" in sel.columns and sel["picked"].notna().any():
        print("\nWhich candidate the selector ranked first (top-1) and how its final mask scored:")
        pk = sel.groupby("picked")["dice"].agg(count="count", mean_dice="mean", median_dice="median")
        pk["share_%"] = (100 * pk["count"] / pk["count"].sum()).round(1)
        print(pk.sort_values("count", ascending=False).round(3).to_string())


def working_res(cand_dir):
    """Exact 'working-res comparison' of train_selector.py, rebuilt from candidates/*.pkl.

    ensemble_workres and oracle_workres only need the saved candidate scores. selected_workres needs the
    selector's out-of-fold predictions, so this re-runs the SAME 5-fold grouped CV as train_selector.py
    (same features, same model, same seed). It reads only candidates/*.pkl, writes nothing, and takes about
    as long as the CV step of train_selector.py."""
    from sklearn.model_selection import GroupKFold

    from src.candidates import build_X
    from train_selector import model  # same hyper-parameters as the selector you trained

    files = sorted(Path(cand_dir).glob("*.pkl"))
    if not files:
        print(f"\nNo candidate files found in {cand_dir}/ (run build_candidates.py first).")
        return
    D = []
    for f in files:
        with open(f, "rb") as fh:
            d = pickle.load(fh)
        if len(d["F"]):
            D.append(d)
    print(f"\n{len(D)} images with candidates; re-running the 5-fold CV (this can take a minute)...")
    Xs, ys, gs = [], [], []
    for i, d in enumerate(D):
        h, w = d["shape"]
        M = [np.unpackbits(m)[:h * w].reshape(h, w).astype(bool) for m in d["masks"]]
        Xs.append(build_X(d["F"], M))
        ys.append(d["y"])
        gs += [i] * len(d["y"])
    X, y, g = np.vstack(Xs), np.concatenate(ys), np.array(gs)
    pred = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, g):
        pred[te] = model().fit(X[tr], y[tr]).predict(X[te])
    rows = []
    for i, d in enumerate(D):
        s = np.where(g == i)[0]
        k = int(np.argmax(pred[s]))
        ens = [j for j, n in enumerate(d["names"]) if n == "ensemble"]
        rows.append({"ensemble_workres": d["y"][ens[0]] if ens else 0.0,
                     "selected_workres": d["y"][k], "oracle_workres": d["y"].max()})
    print("\nworking-res comparison (same images):")
    print(pd.DataFrame(rows).mean().round(4).to_string())


def summarise(log, exclude=(), workres=None):
    log = log.copy()
    if exclude:
        ex = {e.upper() for e in exclude}
        log = log[~log["image"].str.upper().isin(ex)]
        print(f"Excluded {len(ex)} requested image id(s); {log['image'].nunique()} images remain.")

    if "status" not in log.columns:
        log["status"] = "ok"
    status = log["status"].astype(str)
    good = log[~status.str.startswith("error")]
    print(f"\n{len(log)} rows, {log['image'].nunique()} images. Status counts:")
    print(status.str.split(":").str[0].value_counts().to_string())

    sel = good[good["method"] == SELECTED]
    others = good[good["method"] != SELECTED]

    if len(sel):
        selector_block(sel)          # exactly the first block of train_selector.py's printout
    if workres:
        working_res(workres)         # ... and the second block, right below it
    if len(sel):
        picked_table(sel)
    if len(others):
        print("\nMean / median Dice and IoU by method and label (empty predictions count as 0):")
        print(others.groupby(["method", "label"])[["dice", "iou"]]
              .agg(["mean", "median", "count"]).round(3).to_string())

    print("\nOverall (all labels):")
    print(good.groupby("method")[["dice", "iou"]].agg(["mean", "median"]).round(3).to_string())

    print("\nImages with Dice == 0, and Dice < 0.5, by method:")
    extra = good.groupby("method")["dice"].agg(
        n="count", zero=lambda s: int((s == 0).sum()), below_half=lambda s: int((s < 0.5).sum()))
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


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seg-dir", default=str(SEG_OUT_DIR), help="folder with <method>/<id>.png masks")
    ap.add_argument("--gt-dir", default=str(GT_DIR))
    ap.add_argument("--labels", default=str(LABELS_FILE))
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="where to write the recomputed CSV")
    ap.add_argument("--csv", nargs="?", const=str(EXISTING_CSV), default=None, metavar="PATH",
                    help=f"summarise an existing metrics CSV instead of recomputing (default {EXISTING_CSV.name})")
    ap.add_argument("--selected", action="store_true",
                    help=f"shortcut for --csv {SELECTED_CSV.as_posix()} (the train_selector.py results)")
    ap.add_argument("--workres", nargs="?", const="candidates", default=None, metavar="DIR",
                    help="also print the 'working-res comparison' block (ensemble / selected / oracle). "
                         "Rebuilt exactly from candidates/*.pkl by re-running the selector's 5-fold CV")
    ap.add_argument("--exclude", nargs="*", default=[], help="image ids to leave out of the summary")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()

    # Fast path: summarise a CSV that already exists
    csv_path = str(SELECTED_CSV) if args.selected else args.csv
    if csv_path:
        print(f"Summarising existing file: {csv_path}")
        summarise(pd.read_csv(csv_path), args.exclude, args.workres)
        return

    # Recompute from saved masks
    seg_dir = Path(args.seg_dir)
    methods = [m for m in (*METHODS, SELECTED) if (seg_dir / m).is_dir()]
    assert methods, f"No method folders ({', '.join(METHODS)}, {SELECTED}) found inside {seg_dir}"
    print(f"Mask folder: {seg_dir} | methods found: {methods}")

    mask_index = index_masks(Path(args.gt_dir))
    labels = load_labels(Path(args.labels), LABEL_COLUMN)

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

    # attach the 'picked' candidate name for the selector rows, if train_selector.py's CSV exists
    if SELECTED_CSV.exists():
        pk = pd.read_csv(SELECTED_CSV)[["image", "picked"]].drop_duplicates("image")
        log = log.merge(pk.assign(method=SELECTED), on=["image", "method"], how="left")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    log.to_csv(out, index=False)
    print(f"Wrote {len(log)} rows to {out}")
    summarise(log, args.exclude, args.workres)


if __name__ == "__main__":  # required on Windows for multiprocessing
    main()