"""Step 1: generate candidates + features + true Dice for every image -> candidates/<id>.pkl  (resumable)
    python build_candidates.py [--limit N] [--workers W]
Optional: if dl_probs/<id>.png exists (from dl_unet.py) it is used as extra features + one extra candidate."""
import argparse, os, pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from src.candidates import dice, generate
from src.config import GT_DIR, LABEL_COLUMN, LABELS_FILE, OUT_NO_CLAHE_DIR, SEG_CFG
from src.ground_truth import index_masks, load_labels, pick_ground_truth, read_mask

CAND_DIR = Path("candidates"); DL_DIR = Path("dl_probs")


def one(job):
    stem, label, gt_path = job
    cv2.setNumThreads(1)
    out = CAND_DIR / f"{stem}.pkl"
    if out.exists(): return None
    img = cv2.imread(str(OUT_NO_CLAHE_DIR / f"{stem}.png"), cv2.IMREAD_COLOR)
    gt = read_mask(gt_path)
    if img is None or gt is None or not gt.any(): return stem
    dlp = DL_DIR / f"{stem}.png"
    dl = cv2.imread(str(dlp), 0).astype(np.float32) / 255 if dlp.exists() else None
    d = generate(img, SEG_CFG, dl); h, w = d["shape"]
    gtw = cv2.resize(gt.astype(np.uint8) * 255, (w, h), interpolation=cv2.INTER_AREA) > 127
    pickle.dump(dict(stem=stem, label=label, shape=d["shape"], full_shape=img.shape[:2], names=d["names"], F=d["F"],
                     y=np.array([dice(m, gtw) for m in d["masks"]], np.float32),
                     masks=[np.packbits(m) for m in d["masks"]]), open(out, "wb"))
    return None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int); ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args(); CAND_DIR.mkdir(exist_ok=True)
    idx, labels = index_masks(GT_DIR), load_labels(LABELS_FILE, LABEL_COLUMN); jobs = []
    for s in sorted(p.stem for p in OUT_NO_CLAHE_DIR.glob("*.png")):
        if s in labels and s in idx:
            gp, _, _ = pick_ground_truth(idx[s], SEG_CFG.consensus_tag)
            if gp is not None: jobs.append((s, labels[s], gp))
    jobs = jobs[:a.limit] if a.limit else jobs
    with ProcessPoolExecutor(a.workers) as ex:
        bad = [r for r in tqdm(ex.map(one, jobs, chunksize=2), total=len(jobs)) if r]
    print("skipped (unreadable image/GT):", bad)


if __name__ == "__main__":
    main()