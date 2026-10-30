"""Classical lesion segmentation: Otsu, GrabCut and active contours (Chan-Vese).

Every method takes a BGR image at working resolution plus a boolean `valid` mask
(False on the black dermoscope vignette) and returns a boolean lesion mask.
All methods share the same cleanup and the same upsampling back to full size."""
import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import disk_level_set, morphological_chan_vese

from .config import SegmentationConfig

METHODS = ("otsu", "grabcut", "active_contour")


# ----------------------------------------------------------------- helpers
def _odd(n: int) -> int:
    n = int(n)
    return n if n % 2 == 1 else n + 1


def _shrink(img: np.ndarray, max_side: int) -> np.ndarray:
    """Downscale so the longest side is max_side (never upscales)."""
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s >= 1.0:
        return img
    return cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))),
                      interpolation=cv2.INTER_AREA)


def to_full_res(mask: np.ndarray, shape_hw) -> np.ndarray:
    """Upsample a boolean mask to (H, W) with linear interpolation, then re-binarise."""
    h, w = shape_hw
    if mask.shape == (h, w):
        return mask
    up = cv2.resize(mask.astype(np.uint8) * 255, (w, h), interpolation=cv2.INTER_LINEAR)
    return up > 127


def detect_vignette(bgr: np.ndarray, cfg: SegmentationConfig) -> np.ndarray:
    """Return `valid`: True everywhere except the black dermoscope vignette."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    dark = (gray < cfg.vignette_dark_thresh).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=4)
    vig = np.zeros((h, w), np.uint8)
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        touches_border = x == 0 or y == 0 or x + bw == w or y + bh == h
        if touches_border and area >= cfg.vignette_min_area_frac * h * w:
            vig[labels == i] = 1
    if vig.any():
        vig = cv2.dilate(vig, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    if vig.mean() > cfg.vignette_max_frac:  # degenerate (very dark image): ignore
        vig[:] = 0
    return vig == 0


def _ring(shape, frac: float) -> np.ndarray:
    h, w = shape
    t = max(2, int(frac * min(h, w)))
    ring = np.ones(shape, bool)
    ring[t:-t, t:-t] = False
    return ring


def _pick_side(region: np.ndarray, valid: np.ndarray, cfg: SegmentationConfig) -> np.ndarray:
    """A two-class split has no built-in polarity. The lesion is the side that covers
    less of the image border ring (lesions sit away from the edges, skin does not)."""
    region = region & valid
    ring = _ring(region.shape, cfg.border_ring_frac) & valid
    if not ring.any():
        return region
    return region if region[ring].mean() <= 0.5 else (~region & valid)


def clean_mask(mask: np.ndarray, valid: np.ndarray, cfg: SegmentationConfig) -> np.ndarray:
    """Shared post-processing: opening, largest connected component, hole filling."""
    m = (mask & valid).astype(np.uint8)
    k = _odd(max(3, round(cfg.clean_open_frac * min(m.shape))))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return np.zeros(m.shape, bool)
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return ndi.binary_fill_holes(labels == biggest)


# ----------------------------------------------------------------- methods
def segment_otsu(bgr: np.ndarray, valid: np.ndarray, cfg: SegmentationConfig) -> np.ndarray:
    """Global Otsu threshold on the Lab lightness channel, computed over valid pixels only."""
    L = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[:, :, 0]
    L = cv2.GaussianBlur(L, (_odd(cfg.otsu_blur_ksize),) * 2, 0)
    vals = L[valid].reshape(-1, 1)
    if vals.size == 0:
        return np.zeros(valid.shape, bool)
    thr, _ = cv2.threshold(vals, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return _pick_side(L <= thr, valid, cfg)


def _fill_vignette(bgr: np.ndarray, valid: np.ndarray, cfg: SegmentationConfig) -> np.ndarray:
    """Paint the vignette with colours resampled from the skin near the edge of the valid area,
    so that black pixels do not swamp GrabCut's background model."""
    if valid.all() or not valid.any():
        return bgr
    t = max(3, int(cfg.border_ring_frac * min(valid.shape)))
    inner = cv2.erode(valid.astype(np.uint8), np.ones((2 * t + 1, 2 * t + 1), np.uint8)) > 0
    band = (valid & ~inner) | (_ring(valid.shape, cfg.border_ring_frac) & valid)
    samples = bgr[band]
    rng = np.random.default_rng(0)  # fixed seed: reproducible results
    out = bgr.copy()
    out[~valid] = samples[rng.integers(0, len(samples), int((~valid).sum()))]
    return out


def segment_grabcut(bgr: np.ndarray, valid: np.ndarray, cfg: SegmentationConfig) -> np.ndarray:
    """GrabCut initialised from a centred rectangle (independent of the other methods).
    Pixels outside the rectangle and on the vignette are fixed as background."""
    h, w = valid.shape
    x0, y0 = int(cfg.grabcut_margin_frac * w), int(cfg.grabcut_margin_frac * h)
    mask = np.full((h, w), cv2.GC_BGD, np.uint8)
    mask[y0:h - y0, x0:w - x0] = cv2.GC_PR_FGD
    mask[~valid] = cv2.GC_BGD
    if not (mask == cv2.GC_PR_FGD).any():
        return np.zeros((h, w), bool)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(np.ascontiguousarray(_fill_vignette(bgr, valid, cfg)), mask, None, bgd, fgd,
                cfg.grabcut_iters, cv2.GC_INIT_WITH_MASK)
    return (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)


def segment_active_contour(bgr: np.ndarray, valid: np.ndarray, cfg: SegmentationConfig) -> np.ndarray:
    """Morphological Chan-Vese (a level-set method) started from a centred disk."""
    L = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
    if cfg.ac_blur_sigma > 0:
        L = cv2.GaussianBlur(L, (0, 0), cfg.ac_blur_sigma)
    if valid.any() and not valid.all():
        L[~valid] = np.median(L[valid])      # the vignette must not form its own class
    L = (L - L.min()) / (float(np.ptp(L)) + 1e-6)
    h, w = L.shape
    init = disk_level_set((h, w), radius=cfg.ac_init_radius_frac * min(h, w))
    region = morphological_chan_vese(L, num_iter=cfg.ac_iters, init_level_set=init,
                                     smoothing=cfg.ac_smoothing).astype(bool)
    return _pick_side(region, valid, cfg)


# ----------------------------------------------------------------- driver
def run_methods(clahe_bgr: np.ndarray, noclahe_bgr: np.ndarray, cfg: SegmentationConfig):
    """Run all three methods on one image.
    Returns ({method: (full_resolution_bool_mask, status)}, vignette_fraction)."""
    H, W = clahe_bgr.shape[:2]
    cache = {}

    def prep(side):
        if side not in cache:
            img = np.ascontiguousarray(_shrink(clahe_bgr, side))
            valid = detect_vignette(_shrink(noclahe_bgr, side), cfg)  # vignette from the un-CLAHE'd image
            cache[side] = (img, valid)
        return cache[side]

    plan = (("otsu", segment_otsu, cfg.work_max_side),
            ("grabcut", segment_grabcut, cfg.work_max_side),
            ("active_contour", segment_active_contour, cfg.ac_max_side))
    out = {}
    for name, fn, side in plan:
        img, valid = prep(side)
        try:
            mask = clean_mask(fn(img, valid, cfg), valid, cfg)
            status = "ok" if mask.any() else "empty"
            out[name] = (to_full_res(mask, (H, W)), status)
        except Exception as e:  # record and keep going
            out[name] = (np.zeros((H, W), bool), f"error: {e}")
    vignette_frac = 1.0 - float(prep(cfg.work_max_side)[1].mean())
    return out, vignette_frac