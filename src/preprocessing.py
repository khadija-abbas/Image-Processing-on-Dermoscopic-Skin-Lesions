"""Pure preprocessing functions. Inputs are never modified in place."""
from dataclasses import dataclass

import cv2
import numpy as np

from .config import PreprocessConfig


@dataclass
class PreprocessResult:
    hair_mask: np.ndarray        # uint8, 255 where hair was detected
    hair_removed: np.ndarray     # BGR after hair removal
    denoised: np.ndarray         # BGR after median filtering (same as hair_removed if disabled)
    final: np.ndarray            # BGR after CLAHE: this is what gets saved
    hair_fraction: float         # fraction of pixels in the final hair mask (for QA)
    hair_threshold_used: int     # black-hat threshold after adaptive raising (for QA)


def _odd(n: int) -> int:
    n = int(n)
    return n if n % 2 == 1 else n + 1


def _ellipse(d: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))

def detect_hair(bgr: np.ndarray, cfg: PreprocessConfig):
    """Return (mask, threshold_used). Mask is 255 on thin, elongated dark structures."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if cfg.hair_blur_sigma > 0:
        gray = cv2.GaussianBlur(gray, (0, 0), cfg.hair_blur_sigma)
    m = min(gray.shape)

    k = _odd(max(cfg.hair_kernel_min, round(cfg.hair_kernel_frac * m)))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, _ellipse(k))

    # 1) Adaptive threshold: raise it until the raw mask is a plausible size.
    thr = cfg.hair_threshold
    while True:
        raw = (blackhat > thr).astype(np.uint8) * 255
        raw_frac = np.count_nonzero(raw) / raw.size
        if raw_frac <= cfg.hair_max_raw_fraction or thr >= cfg.hair_threshold_max:
            break
        thr += cfg.hair_threshold_step
    mask = raw

    # 2) Remove thick structures (dark dots, lesion edges): anything that survives
    #    an opening wider than a hair is not hair.
    max_w = max(cfg.hair_max_width_min, round(cfg.hair_max_width_frac * m))
    thick = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _ellipse(_odd(max_w + 2)))
    if thick.any():
        mask[cv2.dilate(thick, np.ones((3, 3), np.uint8)) > 0] = 0

    # 3) Ignore a thin strip at the image border (frame lines are not hair).
    b = cfg.hair_border_px
    if b > 0:
        mask[:b, :] = 0
        mask[-b:, :] = 0
        mask[:, :b] = 0
        mask[:, -b:] = 0

    # 4) Keep only components that are both long and elongated. Short dashes
    #    (e.g. lesion globules) and compact blobs are dropped.
    min_len = max(cfg.hair_min_length_min, round(cfg.hair_min_length_frac * m))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros(n, dtype=bool)
    if n > 1:
        length = np.maximum(stats[1:, cv2.CC_STAT_WIDTH], stats[1:, cv2.CC_STAT_HEIGHT])
        area = np.maximum(stats[1:, cv2.CC_STAT_AREA], 1)
        elongation = length / np.sqrt(area)
        keep[1:] = (length >= min_len) & (elongation >= cfg.hair_min_elongation)
    mask = keep[labels].astype(np.uint8) * 255

    # 5) Grow slightly so inpainting covers the soft edges of each hair.
    if cfg.hair_dilate_iter > 0 and mask.any():
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=cfg.hair_dilate_iter)
    return mask, int(thr)


def remove_hair(bgr: np.ndarray, cfg: PreprocessConfig):
    """Detect hair, then fill it in with Telea inpainting."""
    mask, thr = detect_hair(bgr, cfg)
    if not mask.any():
        return bgr.copy(), mask, thr
    cleaned = cv2.inpaint(bgr, mask, cfg.inpaint_radius, cv2.INPAINT_TELEA)
    return cleaned, mask, thr


def suppress_noise(bgr: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """Median filter to suppress small speckle artifacts."""
    if not cfg.apply_median:
        return bgr.copy()
    return cv2.medianBlur(bgr, _odd(cfg.median_ksize))


def apply_clahe_lab(bgr: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """CLAHE on the Lab lightness channel only, so colors are not distorted."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=cfg.clahe_clip_limit,
                            tileGridSize=cfg.clahe_tile_grid)
    return cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)


def convert_color_space(bgr: np.ndarray, space: str) -> np.ndarray:
    """Helper for the segmentation stage: 'hsv' or 'lab' from a BGR image."""
    codes = {"hsv": cv2.COLOR_BGR2HSV, "lab": cv2.COLOR_BGR2LAB}
    return cv2.cvtColor(bgr, codes[space.lower()])


def run_pipeline(bgr: np.ndarray, cfg: PreprocessConfig) -> PreprocessResult:
    assert bgr.ndim == 3 and bgr.shape[2] == 3 and bgr.dtype == np.uint8, \
        "Expected an 8-bit 3-channel BGR image"
    hair_removed, mask, thr = remove_hair(bgr, cfg)
    denoised = suppress_noise(hair_removed, cfg)
    final = apply_clahe_lab(denoised, cfg)
    assert final.shape == bgr.shape, "Preprocessing must not change image size"
    return PreprocessResult(
        hair_mask=mask,
        hair_removed=hair_removed,
        denoised=denoised,
        final=final,
        hair_fraction=float(np.count_nonzero(mask)) / mask.size,
        hair_threshold_used=thr,
    )