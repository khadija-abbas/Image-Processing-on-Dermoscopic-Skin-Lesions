"""Preprocessing v2 for dermoscopic images (drop-in replacement for run_preprocessing.py).

Pipeline (every stage can be switched off in PrepCfg):
  1. Hair / ruler removal   oriented thin-line detector + hysteresis, lesion-aware, masked-fill
  2. Shading correction     robust quadratic surface fitted on skin only (fixes uneven light / vignette)
  3. Optional colour constancy (Shades-of-Gray)
  4. Light median denoise                                   -> saved to OUT_NO_CLAHE_DIR
  5. Brightness-preserving CLAHE (local contrast only)      -> saved to OUT_DIR

The two outputs keep the same file names as before (<stem>.png), so run_segmentation.py works unchanged.

Usage (from the repo root):
  python preprocessing_v2.py --limit 20 --overwrite --preview 8     # test run + preview sheet
  python preprocessing_v2.py --overwrite                            # full run (resumes if interrupted)
  python preprocessing_v2.py --input images --out-clahe out/clahe --out-noclahe out/noclahe   # explicit paths
"""
from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


# ----------------------------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------------------------
@dataclass
class PrepCfg:
    # working resolution for hair detection (the mask is upsampled back to full resolution)
    work_max_side: int = 1024

    # --- hair detection -------------------------------------------------------------------
    # response units are gray levels of Lab-L (0-255): "how much darker is the line than BOTH sides"
    hair_sigma_w: tuple = (0.9, 1.6, 2.5)   # line half-widths searched (px at working resolution)
    hair_short_len: float = 5.0             # along-line smoothing for the precise (short) detector
    hair_long_len: float = 14.0             # along-line smoothing for the continuity (long) detector
    hair_seed: float = 2.3                  # long-detector response needed to START a hair
    hair_grow_long: float = 1.25            # ...and to CONTINUE along it (faint stretches)
    hair_grow_short: float = 1.3            #    (must also hold for the short detector = exact location)
    hair_min_len: int = 35                  # shortest accepted hair (px, working resolution)
    hair_long_keep: int = 100               # components at least this long are kept regardless of shape
    hair_min_aspect: float = 3.0            # shorter ones must be elongated (length / width) - kills dots/stars
    hair_border: int = 8                    # ignore detections this close to the image border (px)
    hair_border_frac: float = 0.035         # wider guard (fraction of min(h, w)) used ONLY when a dark vignette frame is found
    hair_dilate: int = 2                    # grow the mask by this many px (working res) to cover soft hair edges
    protect_lesion: bool = True             # do not "remove" dark streaks inside the lesion itself

    # --- shading correction ---------------------------------------------------------------
    shading: bool = True
    shading_max: float = 25.0               # clip the correction to +-this many L levels

    # --- colour constancy (OFF by default; changes colour features, enable deliberately) ---
    color_constancy: bool = False
    cc_minkowski_p: float = 6.0

    # --- denoise + contrast ---------------------------------------------------------------
    median_min: int = 3
    median_max: int = 7                     # kernel ~ min(h, w) / 256, clipped to [min, max], odd
    clahe_median: int = 5                   # the CLAHE branch is smoothed at least this much first: CLAHE exaggerates
                                            # whatever hairs/texture are left, a 5x5 median wipes <=2 px-wide residual hairs
    clahe_clip: float = 1.5
    clahe_grid: int = 8
    clahe_keep_lowfreq: bool = True         # re-impose original low-frequency brightness after CLAHE
    clahe_lf_sigma_frac: float = 0.08       # low-frequency scale = this * min(h, w)


# ----------------------------------------------------------------------------------------------
# Hair detection
# ----------------------------------------------------------------------------------------------
def _odd(n: float) -> int:
    n = int(round(n))
    return n if n % 2 else n + 1


def line_response(L: np.ndarray, sigmas_w, sigma_l: float, n_ang: int) -> np.ndarray:
    """Oriented dark-thin-line detector (float32 in, float32 out, same units as L).

    For every orientation the image is rotated so the line is horizontal, smoothed with an elongated
    Gaussian (sigma_l along the line, sigma_w across), and the response is
        min(flank_above - centre, flank_below - centre).
    Requiring the centre to be darker than BOTH sides means a one-sided step (lesion border) scores ~0,
    while a thin dark line scores its contrast. Long sigma_l averages noise and suppresses dots.
    """
    h, w = L.shape
    margin = int(np.ceil(3 * sigma_l)) + 12
    D = int(np.ceil(np.hypot(h, w))) + 2 * margin
    best = np.zeros((h, w), np.float32)
    for i in range(n_ang):
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), 180.0 * i / n_ang, 1.0)
        M[0, 2] += D / 2.0 - w / 2.0
        M[1, 2] += D / 2.0 - h / 2.0
        rot = cv2.warpAffine(L, M, (D, D), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        acc = None
        for sw in sigmas_w:
            B = cv2.GaussianBlur(rot, (0, 0), sigmaX=sigma_l, sigmaY=sw, borderType=cv2.BORDER_REFLECT)
            d = max(1, int(round(2.2 * sw + 1.5)))
            up = np.empty_like(B)
            dn = np.empty_like(B)
            up[d:] = B[:-d]
            up[:d] = B[:1]
            dn[:-d] = B[d:]
            dn[-d:] = B[-1:]
            r = np.minimum(up - B, dn - B)
            acc = r if acc is None else np.maximum(acc, r)
        back = cv2.warpAffine(acc, M, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                              borderMode=cv2.BORDER_REPLICATE)
        np.maximum(best, back, out=best)
    return np.maximum(best, 0)


def _long_response(L: np.ndarray, cfg: PrepCfg) -> np.ndarray:
    """Continuity detector. Run at half resolution (6x faster, ~same result), widths matched."""
    h, w = L.shape
    if min(h, w) < 300:
        return line_response(L, (1.0, 1.8), cfg.hair_long_len, 18)
    half = cv2.resize(L, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    r = line_response(half, (0.5, 0.9), cfg.hair_long_len / 2.0, 18)
    return cv2.resize(r, (w, h), interpolation=cv2.INTER_LINEAR)


def lesion_guard(lab: np.ndarray, max_hair_w: int = 7):
    """Rough dark-lesion region (None if the image has no clearly dark lesion). Hairs are closed away first."""
    L8 = lab[..., 0]
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * max_hair_w + 1, 2 * max_hair_w + 1))
    Lc = cv2.GaussianBlur(cv2.morphologyEx(L8, cv2.MORPH_CLOSE, se), (0, 0), 3)
    _, m = cv2.threshold(Lc, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    m = m.astype(bool)
    frac = m.mean()
    if frac < 0.005 or frac > 0.6 or (Lc[~m].mean() - Lc[m].mean()) < 25:
        return None
    n, lab_cc, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = np.zeros(n, bool)
    keep[1:] = areas >= 0.25 * areas.max()
    return cv2.dilate(keep[lab_cc].astype(np.uint8),
                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))).astype(bool)


def frame_mask(lab: np.ndarray) -> np.ndarray:
    """Black / grey vignette frame and corners: very dark, colourless, connected to the image border."""
    L8 = cv2.GaussianBlur(lab[..., 0], (0, 0), 2)
    chroma = np.hypot(lab[..., 1].astype(np.float32) - 128, lab[..., 2].astype(np.float32) - 128)
    ref = float(np.percentile(L8, 60))
    dark = ((L8 < 0.60 * ref) | (L8 < 40)) & (chroma < 14)
    n, cc = cv2.connectedComponents(dark.astype(np.uint8), connectivity=8)
    if n <= 1:
        return np.zeros(dark.shape, bool)
    border_ids = np.unique(np.concatenate([cc[0], cc[-1], cc[:, 0], cc[:, -1]]))
    keep = np.zeros(n, bool)
    keep[border_ids[border_ids > 0]] = True
    m = keep[cc]
    return cv2.dilate(m.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))).astype(bool)


def _hysteresis(strong: np.ndarray, weak: np.ndarray) -> np.ndarray:
    n, lab = cv2.connectedComponents(weak.astype(np.uint8), connectivity=8)
    ids = np.unique(lab[strong])
    keep = np.zeros(n, bool)
    keep[ids[ids > 0]] = True
    return keep[lab]


def _shape_filter(mask: np.ndarray, cfg: PrepCfg) -> np.ndarray:
    m = mask.astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(m)
    for c in cnts:
        (_, _), (rw, rh), _ = cv2.minAreaRect(c)
        long_, short_ = max(rw, rh), max(min(rw, rh), 1.0)
        if long_ >= cfg.hair_long_keep or (long_ >= cfg.hair_min_len and long_ / short_ >= cfg.hair_min_aspect):
            cv2.drawContours(out, [c], -1, 1, thickness=cv2.FILLED)
    return out.astype(bool) & mask


def detect_hair(img: np.ndarray, cfg: PrepCfg, lab=None, guard=None, frame=None) -> np.ndarray:
    """Boolean hair/ruler mask at the resolution of `img`."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB) if lab is None else lab
    L = lab[..., 0].astype(np.float32)
    rs = line_response(L, cfg.hair_sigma_w, cfg.hair_short_len, 12)
    rl = _long_response(L, cfg)
    # skin texture level: the MEDIAN of the continuity response (hairs are sparse, so it is not driven by them).
    # Clean skin ~0.27, fine-textured ~0.5, mottled / JPEG-noisy ~0.7+ -> thresholds scale with texture^0.75
    noise = float(np.clip(np.median(rl) / 0.27, 1.0, 3.0) ** 0.75)
    for blocked in (guard if cfg.protect_lesion else None, frame):
        if blocked is not None:
            rs[blocked] = 0
            rl[blocked] = 0
    has_frame = frame is not None and bool(frame.any())
    b = max(cfg.hair_border, int(round(cfg.hair_border_frac * min(L.shape)))) if has_frame else cfg.hair_border
    for r in (rs, rl):
        r[:b] = 0
        r[-b:] = 0
        r[:, :b] = 0
        r[:, -b:] = 0
    mask = _hysteresis(rl > cfg.hair_seed * noise, (rl > cfg.hair_grow_long * noise) & (rs > cfg.hair_grow_short * noise))
    return _shape_filter(mask, cfg)


# ----------------------------------------------------------------------------------------------
# Filling, shading, colour, contrast
# ----------------------------------------------------------------------------------------------
def fill_masked(img: np.ndarray, mask: np.ndarray, base_sigmas=(2, 4, 8, 16)) -> np.ndarray:
    """Replace masked pixels by a Gaussian-weighted average of the surrounding valid pixels
    (normalised convolution, small scale first). Smoother and streak-free compared with Telea."""
    if not mask.any():
        return img.copy()
    s = max(1.0, min(img.shape[:2]) / 768.0)
    f = img.astype(np.float32)
    valid = (~mask).astype(np.float32)
    out = f.copy()
    todo = mask.copy()
    for sg in base_sigmas:
        sg = sg * s
        num = cv2.GaussianBlur(f * valid[..., None], (0, 0), sg)
        den = cv2.GaussianBlur(valid, (0, 0), sg)
        ok = todo & (den > 0.35)
        out[ok] = num[ok] / den[ok][:, None]
        todo &= ~ok
        if not todo.any():
            break
    if todo.any():                                   # pathological: fall back to Telea
        t = cv2.inpaint(img, todo.astype(np.uint8) * 255, 5, cv2.INPAINT_TELEA).astype(np.float32)
        out[todo] = t[todo]
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def shading_correction(img: np.ndarray, exclude: np.ndarray | None, cfg: PrepCfg) -> np.ndarray:
    """Subtract a smooth quadratic brightness surface fitted (robustly) on skin pixels only."""
    h, w = img.shape[:2]
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L = lab[..., 0].astype(np.float32)
    ws_ = 96
    hs_ = max(8, int(round(ws_ * h / w)))
    Ls = cv2.resize(L, (ws_, hs_), interpolation=cv2.INTER_AREA)
    ex = np.zeros((hs_, ws_), bool) if exclude is None else cv2.resize(exclude.astype(np.uint8), (ws_, hs_),
                                                                         interpolation=cv2.INTER_AREA) > 0
    yy, xx = np.mgrid[0:hs_, 0:ws_].astype(np.float64)
    x, y = xx / (ws_ - 1) * 2 - 1, yy / (hs_ - 1) * 2 - 1
    A = np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], -1).reshape(-1, 6)
    z = Ls.astype(np.float64).ravel()
    use = ~ex.ravel()
    if use.sum() < 200:
        return img
    for _ in range(5):
        Aw = A[use]
        coef = np.linalg.solve(Aw.T @ Aw + 1e-6 * np.eye(6), Aw.T @ z[use])
        r = z - A @ coef
        sd = 1.4826 * np.median(np.abs(r[use] - np.median(r[use]))) + 1e-6
        new = ~ex.ravel() & (r > -1.5 * sd) & (r < 3.0 * sd)       # dark outliers = lesion / dirt
        if new.sum() < 200:
            break
        use = new
    surf = (A @ coef).reshape(hs_, ws_)
    corr = np.clip(surf - np.median(surf[use.reshape(hs_, ws_)]), -cfg.shading_max, cfg.shading_max)
    corr = cv2.resize(corr.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    lab[..., 0] = np.clip(L - corr + 0.5, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def color_constancy(img: np.ndarray, ignore: np.ndarray | None, p: float) -> np.ndarray:
    """Shades-of-Gray (Minkowski-p mean) illuminant estimate, per-channel von Kries scaling."""
    f = img.astype(np.float32) / 255.0
    sel = np.ones(img.shape[:2], bool) if ignore is None else ~ignore
    if sel.sum() < 1000:
        sel[:] = True
    e = np.array([np.mean(f[..., c][sel] ** p) ** (1.0 / p) for c in range(3)]) + 1e-6
    gain = e.mean() / e
    return np.clip(f * gain[None, None, :] * 255.0 + 0.5, 0, 255).astype(np.uint8)


def gentle_clahe(img: np.ndarray, cfg: PrepCfg) -> np.ndarray:
    """CLAHE on Lab-L that only adds LOCAL contrast: the low-frequency brightness of the input is kept,
    so plain skin is not pushed toward mid-grey tile by tile (which is what makes plain CLAHE blotchy)."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L = lab[..., 0]
    Lc = cv2.createCLAHE(clipLimit=cfg.clahe_clip, tileGridSize=(cfg.clahe_grid, cfg.clahe_grid)).apply(L)
    if cfg.clahe_keep_lowfreq:
        sg = max(4.0, cfg.clahe_lf_sigma_frac * min(L.shape))
        Lf, Cf = L.astype(np.float32), Lc.astype(np.float32)
        Lc = np.clip(cv2.GaussianBlur(Lf, (0, 0), sg) + Cf - cv2.GaussianBlur(Cf, (0, 0), sg) + 0.5,
                     0, 255).astype(np.uint8)
    lab[..., 0] = Lc
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ----------------------------------------------------------------------------------------------
# Full pipeline for one image
# ----------------------------------------------------------------------------------------------
def preprocess_image(img: np.ndarray, cfg: PrepCfg | None = None) -> dict:
    """img: uint8 BGR. Returns dict(noclahe, clahe, hair_removed, hair_mask, hair_frac)."""
    cfg = cfg or PrepCfg()
    h, w = img.shape[:2]
    s = min(1.0, cfg.work_max_side / max(h, w))
    small = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA) if s < 1 else img
    lab_s = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    guard_s = lesion_guard(lab_s)
    frame_s = frame_mask(lab_s)

    # 1) hair / ruler removal
    hair_s = detect_hair(small, cfg, lab_s, guard_s, frame_s)
    if s < 1:
        hair = cv2.resize(hair_s.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR) > 0.3
    else:
        hair = hair_s
    k = 2 * int(round(cfg.hair_dilate / s)) + 1 if cfg.hair_dilate > 0 else 1
    frame_full = frame_s if s == 1 else cv2.resize(frame_s.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    if k > 1:
        hair = cv2.dilate(hair.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))).astype(bool)
    hair &= ~frame_full
    removed = fill_masked(img, hair)

    # 2) shading, 3) colour
    skin_excl = None
    if guard_s is not None or frame_s.any():
        e = np.zeros(small.shape[:2], bool)
        if guard_s is not None:
            e |= guard_s
        e |= frame_s
        skin_excl = e if s == 1 else cv2.resize(e.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    out = shading_correction(removed, skin_excl, cfg) if cfg.shading else removed
    if frame_full.any():
        out[frame_full] = removed[frame_full]      # keep the black frame black
    if cfg.color_constancy:
        out = color_constancy(out, skin_excl, cfg.cc_minkowski_p)

    # 4) denoise  -> "no CLAHE" image;  5) local contrast -> "CLAHE" image
    kmed = int(np.clip(_odd(min(h, w) / 256.0), cfg.median_min, cfg.median_max))
    noclahe = cv2.medianBlur(out, kmed)
    clahe_in = noclahe if kmed >= cfg.clahe_median else cv2.medianBlur(out, cfg.clahe_median)
    clahe = gentle_clahe(clahe_in, cfg)
    return {"noclahe": noclahe, "clahe": clahe, "hair_removed": removed, "hair_mask": hair,
            "hair_frac": float(hair.mean())}


# ----------------------------------------------------------------------------------------------
# Batch runner
# ----------------------------------------------------------------------------------------------
def _resolve_paths(args):
    try:
        from src import config as C
    except Exception:                                   # noqa: BLE001
        C = None

    def pick(*names):
        for n in names:
            if C is not None and hasattr(C, n):
                return Path(getattr(C, n))
        return None

    inp = Path(args.input) if args.input else pick("IMG_DIR", "IMAGES_DIR", "IMAGE_DIR", "RAW_DIR", "INPUT_DIR", "DATA_DIR")
    if inp is None and Path("images").exists():
        inp = Path("images")
    out_c = Path(args.out_clahe) if args.out_clahe else pick("OUT_DIR")
    out_n = Path(args.out_noclahe) if args.out_noclahe else pick("OUT_NO_CLAHE_DIR")
    rep = pick("REPORTS_DIR") or Path("reports")
    if inp is None or out_c is None or out_n is None:
        raise SystemExit("Could not infer paths from src/config.py - pass --input, --out-clahe and --out-noclahe.")
    return inp, out_c, out_n, rep


def _process_one(job):
    path, out_c, out_n, mask_dir, cfg = job
    cv2.setNumThreads(1)
    t0 = time.time()
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return path.stem, np.nan, "error: unreadable image"
    try:
        r = preprocess_image(img, cfg)
        ok = cv2.imwrite(str(out_c / f"{path.stem}.png"), r["clahe"]) and \
            cv2.imwrite(str(out_n / f"{path.stem}.png"), r["noclahe"])
        if mask_dir is not None:
            cv2.imwrite(str(mask_dir / f"{path.stem}.png"), r["hair_mask"].astype(np.uint8) * 255)
        return path.stem, r["hair_frac"], ("ok" if ok else "write_failed") + f" {time.time() - t0:.1f}s"
    except Exception as e:                              # noqa: BLE001
        return path.stem, np.nan, f"error: {e}"


def make_preview(paths, cfg, out_file: Path, n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    paths = list(paths)
    pick = [paths[i] for i in rng.permutation(len(paths))[:n]]
    rows = []
    for p in pick:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        r = preprocess_image(img, cfg)
        ov = img.copy()
        ov[r["hair_mask"]] = (0, 0, 255)
        tiles = [ov, r["hair_removed"], r["noclahe"], r["clahe"]]
        names = [f"{p.stem}: detected hair (red)", "hair removed", "final: no CLAHE", "final: CLAHE"]
        scale = 360.0 / img.shape[0]
        row = []
        for t, nm in zip(tiles, names):
            t = cv2.resize(t, (int(img.shape[1] * scale), 360), interpolation=cv2.INTER_AREA)
            cv2.putText(t, nm, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(t, nm, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            row.append(t)
        rows.append(np.hstack(row))
    width = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, 0, 0, width - r.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for r in rows]
    out_file.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_file), np.vstack(rows))
    print(f"Preview sheet written to {out_file}")


def main():
    ap = argparse.ArgumentParser(description="Preprocessing v2 (hair removal, shading, gentle CLAHE)")
    ap.add_argument("--input", help="folder with the original images (default: from src/config.py or ./images)")
    ap.add_argument("--out-clahe", help="output folder for the CLAHE images (default: config OUT_DIR)")
    ap.add_argument("--out-noclahe", help="output folder for the no-CLAHE images (default: config OUT_NO_CLAHE_DIR)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--preview", type=int, default=0, help="also write a preview sheet with N random images")
    ap.add_argument("--save-masks", action="store_true", help="save hair masks next to the outputs (hair_masks_v2/)")
    ap.add_argument("--color-constancy", action="store_true", help="enable Shades-of-Gray colour normalisation")
    ap.add_argument("--no-shading", action="store_true", help="disable shading correction")
    ap.add_argument("--plain-clahe", action="store_true", help="classic CLAHE (do not keep low-frequency brightness)")
    args = ap.parse_args()

    cfg = PrepCfg(color_constancy=args.color_constancy, shading=not args.no_shading,
                  clahe_keep_lowfreq=not args.plain_clahe)
    inp, out_c, out_n, rep = _resolve_paths(args)
    paths = sorted(p for p in inp.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    assert paths, f"no images found in {inp}"
    if args.limit:
        paths = paths[:args.limit]
    out_c.mkdir(parents=True, exist_ok=True)
    out_n.mkdir(parents=True, exist_ok=True)
    mask_dir = None
    if args.save_masks:
        mask_dir = out_c.parent / "hair_masks_v2"
        mask_dir.mkdir(parents=True, exist_ok=True)
    if args.preview:
        make_preview(paths, cfg, rep / "preprocess_v2_preview.png", args.preview)
    todo = paths if args.overwrite else [p for p in paths
                                         if not ((out_c / f"{p.stem}.png").exists() and (out_n / f"{p.stem}.png").exists())]
    print(f"{len(paths)} images found, {len(todo)} to process, {args.workers} workers")
    if not todo:
        return
    from tqdm import tqdm
    jobs = [(p, out_c, out_n, mask_dir, cfg) for p in todo]
    res = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for r in tqdm(ex.map(_process_one, jobs, chunksize=2), total=len(jobs)):
            res.append(r)
    errs = [r for r in res if str(r[2]).startswith(("error", "write_failed"))]
    fr = np.array([r[1] for r in res if not np.isnan(r[1])])
    print(f"Done. errors: {len(errs)} | hair fraction: median {np.median(fr):.3%}, p95 {np.percentile(fr, 95):.3%}, "
          f"max {fr.max():.3%}")
    for r in errs[:10]:
        print("  ", r)


if __name__ == "__main__":      # required on Windows for multiprocessing
    main()