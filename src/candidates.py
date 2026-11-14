"""Candidate generation + features for the learned candidate selector.

For every image we build ~20 candidate lesion masks (your 4 methods + a threshold sweep over the
saliency map + optionally the U-Net mask) and describe each one with features. A gradient-boosting
model (train_selector.py) then picks the best candidate per image.
"""
import cv2
import numpy as np
from scipy import ndimage as ndi

from .config import SEG_CFG
from .segmentation import (_disk, _shrink, clean_mask, prepare, regularize_shape, segment_active_contour,
                           segment_grabcut, segment_otsu)

SRC = {"otsu": 0, "grabcut": 1, "active_contour": 2, "ensemble": 3, "sweep": 4, "dl": 5}
FN = ["af", "sqaf", "solid", "circ", "aspect", "bboxfrac", "dc", "cx", "cy", "fborder", "si", "sr", "sdiff",
      "sratio", "dL", "da", "db", "chrom", "dchrom", "gb", "gin", "ssd", "mass", "ringvalid", "vig", "validfrac",
      "thr", "fallback", "inverted", "Lin", "Lring", "dl_mean", "dl_dice", "src"]


def dice(a, b):
    s = a.sum() + b.sum()
    return 2.0 * (a & b).sum() / s if s else 1.0


def _feats(m, ctx, lab, grad, src, dl):
    h, w = m.shape
    valid, sal, mn = ctx["valid"], ctx["sal"], min(h, w)
    area = m.sum(); af = area / (h * w); mu = m.astype(np.uint8)
    cnts, _ = cv2.findContours(mu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea)
    ca = max(cv2.contourArea(c), 1); per = max(cv2.arcLength(c, True), 1)
    hull = max(cv2.contourArea(cv2.convexHull(c)), 1)
    x, y, bw, bh = cv2.boundingRect(c)
    ys, xs = np.nonzero(m); cx, cy = xs.mean() / w, ys.mean() / h
    bnd = (mu > 0) & (cv2.erode(mu, np.ones((3, 3), np.uint8)) == 0)
    edge = np.pad(np.zeros((h - 2, w - 2)), 1, constant_values=1) > 0
    inval = cv2.dilate((~valid).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    f_border = (bnd & (edge | inval)).sum() / max(bnd.sum(), 1)
    dil = cv2.dilate(mu, _disk(max(3, int(0.05 * mn)))) > 0
    ring = dil & ~m & valid
    inner = cv2.erode(mu, _disk(max(1, int(0.03 * mn)))) > 0
    if not inner.any(): inner = m
    ms = lambda ch, mk: ch[mk].mean() if mk.any() else 0.0
    si = sal[inner].mean(); sr = sal[ring].mean() if ring.any() else 0.0
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    ci, cr = np.hypot(ms(a, inner), ms(b, inner)), np.hypot(ms(a, ring), ms(b, ring))
    gall = grad[valid].mean() + 1e-6
    dl_mean = dl_d = 0.0
    if dl is not None:
        dl_mean = float(dl[m].mean()); dl_d = dice(m, dl > 0.5)
    return [af, np.sqrt(af), ca / hull, 4 * np.pi * ca / per ** 2, bw / max(bh, 1), bw * bh / (h * w),
            np.hypot(cx - .5, cy - .5), cx, cy, f_border, si, sr, si - sr, si / (sr + 1e-3),
            ms(L, ring) - ms(L, inner), ms(a, inner) - ms(a, ring), ms(b, inner) - ms(b, ring), ci, ci - cr,
            grad[bnd].mean() / gall, grad[inner].mean() / gall, sal[m].std(),
            sal[m].sum() / (sal[valid].sum() + 1e-6), ring.sum() / max(dil.sum() - area, 1),
            ctx["vig_frac"], valid.mean(), ctx["thr"], float(ctx.get("fallback", False)),
            float(ctx.get("inverted", False)), ms(L, inner), ms(L, ring), dl_mean, dl_d, src]


def generate(img_bgr, cfg=SEG_CFG, dl_prob=None):
    """img_bgr: the preprocessed (no-CLAHE) full-res image. dl_prob: optional float map in [0,1] (any size).
    Returns dict(shape, names, masks (working-res bool), F (n x len(FN)))."""
    img = np.ascontiguousarray(_shrink(img_bgr, cfg.work_max_side)); h, w = img.shape[:2]
    ctx = prepare(img, img, cfg); valid = ctx["valid"]
    dl = None if dl_prob is None else cv2.resize(dl_prob.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    lab = cv2.cvtColor(cv2.GaussianBlur(img, (0, 0), 1.5).astype(np.float32) / 255, cv2.COLOR_BGR2LAB)
    gray = cv2.GaussianBlur(lab[..., 0], (0, 0), 1.5)
    grad = np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    use_hull = not ctx.get("fallback"); cands, cl = [], {}
    fns = {"otsu": segment_otsu, "grabcut": segment_grabcut, "active_contour": segment_active_contour}
    for nm, fn in fns.items():
        try:
            m = clean_mask(fn(ctx, cfg), valid, cfg, sal=ctx["sal"])
            if use_hull: m = regularize_shape(m, valid)
        except Exception:
            m = np.zeros((h, w), bool)
        cl[nm] = m; cands.append((nm, m))
    votes = sum(int(wt) * cl[nm].astype(np.uint8) for wt, nm in zip(cfg.ensemble_weights, fns))
    ens = clean_mask(votes >= cfg.ensemble_min_votes, valid, cfg, sal=ctx["sal"])
    if use_hull: ens = regularize_shape(ens, valid)
    cands.append(("ensemble", ens))
    # threshold sweep over the saliency map: the "wrong region" failures are fixed by having alternatives
    sal, thr = ctx["sal"], ctx["thr"]; k = max(3, int(0.01 * min(h, w))) | 1
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    for mult in (0.35, 0.55, 0.8, 1.2, 1.7):
        b = cv2.morphologyEx(((sal > thr * mult) & valid).astype(np.uint8), cv2.MORPH_OPEN, ker)
        nl, lb, st, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
        sc = [(st[i, 4] * sal[lb == i].mean(), i) for i in range(1, nl) if st[i, 4] >= 0.0008 * h * w]
        for _, i in sorted(sc, reverse=True)[:4]:
            m = ndi.binary_fill_holes(cv2.morphologyEx((lb == i).astype(np.uint8), cv2.MORPH_CLOSE, ker) > 0) & valid
            if m.any(): cands.append(("sweep", m))
    if dl is not None:
        m = clean_mask((dl > 0.5) & valid, valid, cfg, sal=sal)
        if m.any(): cands.append(("dl", m))
    keep = []
    for nm, m in cands:
        if m.any() and not any(nm == n2 and dice(m, m2) > 0.95 for n2, m2 in keep): keep.append((nm, m))
    rows, masks, names = [], [], []
    for nm, m in keep:
        try: rows.append(_feats(m, ctx, lab, grad, SRC[nm], dl))
        except Exception: continue
        masks.append(m); names.append(nm)
    return dict(shape=(h, w), names=names, masks=masks, F=np.array(rows, np.float32))


def build_X(F, masks):
    """Feature matrix for ONE image: raw features + within-image relative features + agreement with other candidates."""
    rel = []
    for j in [FN.index(c) for c in ("si", "sdiff", "af", "dc", "fborder", "solid", "gb")]:
        v = F[:, j]; rel += [v - v.max(), v - np.median(v), v.argsort().argsort() / len(v)]
    agree = np.array([np.mean([2 * (a & b).sum() / max(a.sum() + b.sum(), 1) for b in masks]) for a in masks])
    return np.c_[F, np.array(rel).T, agree, np.full(len(F), len(F))]