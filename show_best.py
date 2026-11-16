# # show_best.py - save best / worst / random segmentations as overlay images (green = GT, red = prediction)
# """Save the best / worst / random segmentations as individual overlay images (green = GT, red = prediction).
#     python show_best.py                                   best 10 overall
#     python show_best.py --tag hair                        best 10 HAIRY images
#     python show_best.py --tag sticker                     best 10 images with a calibration STICKER
#     python show_best.py --tag hair --mode worst           honest failures on hairy images
# Tags are computed lazily (only for images that are visited) and cached in reports/image_tags.csv."""
# import argparse
# import cv2
# import numpy as np
# import pandas as pd

# from src.config import GT_DIR, RAW_DIR, REPORTS_DIR, SEG_CFG, SEG_OUT_DIR
# from src.ground_truth import index_masks, pick_ground_truth, read_mask
# from src.preprocessing import PrepCfg, preprocess_image
# from src.segmentation import _shrink, detect_stickers

# ap = argparse.ArgumentParser()
# ap.add_argument("--mode", default="best", choices=["best", "worst", "random", "median"])
# ap.add_argument("--n", type=int, default=10)
# ap.add_argument("--method", default="selected")                                   # folder in segmentation_v4/
# ap.add_argument("--csv", default="segmentation_metrics_selected.csv")             # file in reports/
# ap.add_argument("--tag", default="any", choices=["any", "hair", "sticker"])
# ap.add_argument("--hair-thr", type=float, default=0.03)       # >3% of pixels detected as hair
# ap.add_argument("--sticker-thr", type=float, default=0.01)    # >1% of pixels detected as sticker
# ap.add_argument("--id-min", type=int, default=None)           # keep only ISIC_<number> >= this
# ap.add_argument("--id-max", type=int, default=None)           # keep only ISIC_<number> <= this
# a = ap.parse_args()

# PREP_CFG = PrepCfg()

# df = pd.read_csv(REPORTS_DIR / a.csv).sort_values("dice", ascending=False)
# num = df.image.str.extract(r"(\d+)$")[0].astype(int)
# df = df[(num >= (a.id_min if a.id_min is not None else -1)) & (num <= (a.id_max if a.id_max is not None else 10**9))]
# if a.mode == "worst": df = df.iloc[::-1]
# elif a.mode == "random": df = df.sample(frac=1, random_state=0)
# elif a.mode == "median":
#     if a.tag != "any": raise SystemExit("--mode median is only supported with --tag any")
#     mid = len(df) // 2; df = df.iloc[mid - a.n // 2: mid + (a.n + 1) // 2]

# TAGS = REPORTS_DIR / "image_tags.csv"
# cache = pd.read_csv(TAGS).set_index("image").to_dict("index") if TAGS.exists() else {}
# def tags(stem):
#     if stem not in cache:
#         im = cv2.imread(str(RAW_DIR / f"{stem}.jpg"))
#         sm = np.ascontiguousarray(_shrink(im, SEG_CFG.work_max_side))
#         st = float(detect_stickers(sm, np.ones(sm.shape[:2], bool), SEG_CFG).mean())
#         cache[stem] = dict(hair=float(preprocess_image(im, PREP_CFG)["hair_frac"]), sticker=st)
#         pd.DataFrame.from_dict(cache, orient="index").rename_axis("image").reset_index().to_csv(TAGS, index=False)
#     return cache[stem]
# def keep(stem):
#     if a.tag == "any": return True
#     t = tags(stem)
#     return t["hair"] > a.hair_thr if a.tag == "hair" else t["sticker"] > a.sticker_thr

# rows = []
# for _, r in df.iterrows():
#     if keep(r.image): rows.append(r)
#     if len(rows) == a.n: break
# if len(rows) < a.n: print(f"only found {len(rows)} images with tag '{a.tag}'")

# rng = f"_ids{a.id_min}-{a.id_max}" if (a.id_min is not None or a.id_max is not None) else ""
# out_dir = REPORTS_DIR / f"{a.mode}_{a.n}_{a.tag}{rng}_images"; out_dir.mkdir(parents=True, exist_ok=True)
# idx = index_masks(GT_DIR); TW = 420
# for i, r in enumerate(rows):
#     img = cv2.imread(str(RAW_DIR / f"{r.image}.jpg")); H, W = img.shape[:2]
#     pred = cv2.imread(str(SEG_OUT_DIR / a.method / f"{r.image}.png"), 0) > 0
#     gt = read_mask(pick_ground_truth(idx[r.image], SEG_CFG.consensus_tag)[0])
#     if gt.shape != (H, W): gt = cv2.resize(gt.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
#     if pred.shape != (H, W): pred = cv2.resize(pred.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
#     v = img.copy(); t = max(2, W // 250)
#     for m, c in ((gt, (0, 255, 0)), (pred, (0, 0, 255))):
#         cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         cv2.drawContours(v, cs, -1, c, t)
#     v = cv2.resize(v, (TW, int(TW * H / W)))
#     v = cv2.copyMakeBorder(cv2.resize(v, (TW, TW * 3 // 4)), 0, 34, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
#     cap = f"{r.image[-7:]} {r.label[:3]} Dice {r.dice:.3f}"
#     if a.tag == "hair": cap += f" hair {cache[r.image]['hair']:.0%}"
#     if a.tag == "sticker": cap += f" sticker {cache[r.image]['sticker']:.0%}"
#     cv2.putText(v, cap, (6, TW * 3 // 4 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
#     p = out_dir / f"{i + 1:02d}_{r.image}_dice_{r.dice:.3f}.png"; cv2.imwrite(str(p), v); print("saved", p)
# print(f"Saved {len(rows)} images to: {out_dir}")

# show_best.py - save best / worst / random segmentations as overlay images (green = GT, red = prediction)
"""Save the best / worst / random segmentations as individual overlay images (green = GT, red = prediction).
    python show_best.py                                   best 10 overall
    python show_best.py --tag hair                        best 10 HAIRY images
    python show_best.py --tag sticker                     best 10 images with a calibration STICKER
    python show_best.py --tag hair --mode worst           honest failures on hairy images
Tags are computed lazily (only for images that are visited) and cached in reports/image_tags.csv.

Images whose prediction mask, raw image or ground truth no longer exists (e.g. deleted via wrong.txt)
are skipped automatically, and the script keeps going until it has --n valid images."""
import argparse
import cv2
import numpy as np
import pandas as pd

from src.config import GT_DIR, RAW_DIR, REPORTS_DIR, SEG_CFG, SEG_OUT_DIR
from src.ground_truth import index_masks, pick_ground_truth, read_mask
from src.preprocessing import PrepCfg, preprocess_image
from src.segmentation import _shrink, detect_stickers

ap = argparse.ArgumentParser()
ap.add_argument("--mode", default="best", choices=["best", "worst", "random", "median"])
ap.add_argument("--n", type=int, default=10)
ap.add_argument("--method", default="selected")                                   # folder in segmentation_v4/
ap.add_argument("--csv", default="segmentation_metrics_selected.csv")             # file in reports/
ap.add_argument("--tag", default="any", choices=["any", "hair", "sticker"])
ap.add_argument("--hair-thr", type=float, default=0.03)       # >3% of pixels detected as hair
ap.add_argument("--sticker-thr", type=float, default=0.01)    # >1% of pixels detected as sticker
ap.add_argument("--id-min", type=int, default=None)           # keep only ISIC_<number> >= this
ap.add_argument("--id-max", type=int, default=None)           # keep only ISIC_<number> <= this
a = ap.parse_args()

PREP_CFG = PrepCfg()
idx = index_masks(GT_DIR)

df = pd.read_csv(REPORTS_DIR / a.csv).sort_values("dice", ascending=False)
num = df.image.str.extract(r"(\d+)$")[0].astype(int)
df = df[(num >= (a.id_min if a.id_min is not None else -1)) & (num <= (a.id_max if a.id_max is not None else 10**9))]
if a.mode == "worst": df = df.iloc[::-1]
elif a.mode == "random": df = df.sample(frac=1, random_state=0)
elif a.mode == "median":
    if a.tag != "any": raise SystemExit("--mode median is only supported with --tag any")
    mid = len(df) // 2; df = df.iloc[mid - a.n // 2: mid + (a.n + 1) // 2]

TAGS = REPORTS_DIR / "image_tags.csv"
cache = pd.read_csv(TAGS).set_index("image").to_dict("index") if TAGS.exists() else {}
def tags(stem):
    if stem not in cache:
        im = cv2.imread(str(RAW_DIR / f"{stem}.jpg"))
        sm = np.ascontiguousarray(_shrink(im, SEG_CFG.work_max_side))
        st = float(detect_stickers(sm, np.ones(sm.shape[:2], bool), SEG_CFG).mean())
        cache[stem] = dict(hair=float(preprocess_image(im, PREP_CFG)["hair_frac"]), sticker=st)
        pd.DataFrame.from_dict(cache, orient="index").rename_axis("image").reset_index().to_csv(TAGS, index=False)
    return cache[stem]
def keep(stem):
    if a.tag == "any": return True
    t = tags(stem)
    return t["hair"] > a.hair_thr if a.tag == "hair" else t["sticker"] > a.sticker_thr

def available(stem):
    """True only if the prediction mask, the raw image and a ground-truth mask all still exist."""
    return ((SEG_OUT_DIR / a.method / f"{stem}.png").exists()
            and (RAW_DIR / f"{stem}.jpg").exists()
            and stem in idx)

rows, skipped = [], []
for _, r in df.iterrows():
    if not available(r.image):          # deleted / missing -> skip it and move on to the next image
        skipped.append(r.image)
        continue
    if keep(r.image): rows.append(r)
    if len(rows) == a.n: break
if skipped: print(f"Skipped {len(skipped)} images with missing files (e.g. deleted): "
                  f"{', '.join(s[-7:] for s in skipped[:10])}{' ...' if len(skipped) > 10 else ''}")
if len(rows) < a.n: print(f"only found {len(rows)} of {a.n} requested images with tag '{a.tag}'")

rng = f"_ids{a.id_min}-{a.id_max}" if (a.id_min is not None or a.id_max is not None) else ""
out_dir = REPORTS_DIR / f"{a.mode}_{a.n}_{a.tag}{rng}_images"; out_dir.mkdir(parents=True, exist_ok=True)
TW = 420
for i, r in enumerate(rows):
    img = cv2.imread(str(RAW_DIR / f"{r.image}.jpg")); H, W = img.shape[:2]
    pred = cv2.imread(str(SEG_OUT_DIR / a.method / f"{r.image}.png"), 0) > 0
    gt = read_mask(pick_ground_truth(idx[r.image], SEG_CFG.consensus_tag)[0])
    if gt.shape != (H, W): gt = cv2.resize(gt.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
    if pred.shape != (H, W): pred = cv2.resize(pred.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
    v = img.copy(); t = max(2, W // 250)
    for m, c in ((gt, (0, 255, 0)), (pred, (0, 0, 255))):
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(v, cs, -1, c, t)
    v = cv2.resize(v, (TW, int(TW * H / W)))
    v = cv2.copyMakeBorder(cv2.resize(v, (TW, TW * 3 // 4)), 0, 34, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    cap = f"{r.image[-7:]} {r.label[:3]} Dice {r.dice:.3f}"
    if a.tag == "hair": cap += f" hair {cache[r.image]['hair']:.0%}"
    if a.tag == "sticker": cap += f" sticker {cache[r.image]['sticker']:.0%}"
    cv2.putText(v, cap, (6, TW * 3 // 4 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    p = out_dir / f"{i + 1:02d}_{r.image}_dice_{r.dice:.3f}.png"; cv2.imwrite(str(p), v); print("saved", p)
print(f"Saved {len(rows)} images to: {out_dir}")