"""Paths and tunable parameters for the preprocessing and segmentation stages."""
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "Images"
OUT_DIR = PROJECT_ROOT / "preprocessed"
OUT_NO_CLAHE_DIR = PROJECT_ROOT / "preprocessed_no_clahe"
REPORTS_DIR = PROJECT_ROOT / "reports"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# --- Segmentation stage: paths ---
DATA_DIR = PROJECT_ROOT / "data"
# SEG_OUT_DIR = PROJECT_ROOT / "segmentation_v4"                # v4 = field-of-view + polarity + fallback (new folder keeps v1 results)
SEG_OUT_DIR = PROJECT_ROOT / "segmentation_5000_images"                # v4 = field-of-view + polarity + fallback (new folder keeps v1 results)

GT_DIR = PROJECT_ROOT / "segs"
LABELS_FILE = PROJECT_ROOT / "ima_metadata_2026-10-22.csv"
LABEL_COLUMN = "diagnosis_1"                                # EDIT if the benign/malignant column is named differently
MASK_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class PreprocessConfig:
    # --- Hair detection (black-hat) ---
    hair_blur_sigma: float = 0.5
    hair_kernel_frac: float = 0.025
    hair_kernel_min: int = 13
    hair_threshold: int = 10
    hair_threshold_step: int = 3
    hair_threshold_max: int = 80
    hair_max_raw_fraction: float = 0.15

    # --- Shape filters ---
    hair_max_width_frac: float = 0.012
    hair_max_width_min: int = 5
    hair_min_length_frac: float = 0.04
    hair_min_length_min: int = 15
    hair_min_elongation: float = 3.0
    hair_border_px: int = 3

    # --- Inpainting ---
    hair_dilate_iter: int = 2
    inpaint_radius: int = 3

    # --- Noise / artifact suppression ---
    apply_median: bool = True
    median_ksize: int = 3

    # --- CLAHE (applied to the L channel of Lab only) ---
    clahe_clip_limit: float = 1.5
    clahe_tile_grid: tuple = (8, 8)


@dataclass(frozen=True)
class SegmentationConfig:
    # --- Working resolution (masks are upsampled back to full size for scoring) ---
    work_max_side: int = 384

    # --- Dermoscope field of view / vignette (everything outside is excluded from every method) ---
    fov_rel_thr: float = 0.72             # field of view = convex hull of pixels with gray >= 0.72 * (75th percentile)
    fov_min_frac: float = 0.35            # ignore the FOV result if it would cover less than this fraction of the image
    vignette_dark_thresh: int = 35        # additionally: pure-black pixels connected to the border
    vignette_min_area_frac: float = 0.01
    vignette_max_frac: float = 0.6        # if "vignette" covers more than this, ignore it
    vignette_margin_frac: float = 0.02    # extra erosion of the valid area (soft rim)

    # --- Shared post-processing (identical for all methods) ---
    clean_open_frac: float = 0.01
    clean_close_frac: float = 0.02
    merge_min_area_frac: float = 0.15
    merge_dist_frac: float = 0.20
    use_clahe: bool = False

    # --- Skin-colour model + saliency (shared by all methods) ---
    sal_blur_sigma: float = 1.5
    skin_band_inner_frac: float = 0.01
    skin_band_outer_frac: float = 0.12
    skin_keep_frac: float = 0.7
    skin_trim_iters: int = 3
    skin_cov_reg: float = 1.0
    sal_floor: float = 3.0
    thin_open_frac: float = 0.008
    otsu_relax: float = 0.8               # v5: <1 lowers Otsu's threshold (applied on the compressed scale below)
    sal_gamma: float = 0.5                # v5: Otsu runs on saliency ** sal_gamma (0.5 = sqrt; 1.0 = old raw scale)
    max_seed_frac: float = 0.80
    min_seed_frac: float = 0.003

    # --- Pass 2: re-fit skin on everything outside the lesion ---
    refit_skin: bool = True
    refit_dilate_frac: float = 0.05
    refit_min_frac: float = 0.05
    use_poly_bg: bool = True              # v4: RADIAL illumination model [1, x, y, r^2, r^4] (v3: quadratic, hurt)
    poly_keep_frac: float = 0.85
    fix_polarity: bool = True             # seed brighter+less coloured than the rest => it is skin, lesion fills the field

    # --- Whole-field fallback (lesion fills the field, no coherent lesion/skin contrast) ---
    fallback_whole_field: bool = True
    fallback_max_hi: float = 5.5          # only if overall colour contrast (99.5th pct Mahalanobis) is below this ...
    fallback_max_cc: float = 0.6          # ... and the above-threshold pixels are this fragmented (largest component share)

    # --- Hysteresis growth of the seed (diffuse lesions) ---
    grow_low_frac: float = 0.0            # v5: OFF. With the compressed-scale threshold growth only leaked onto skin (tested 0.6-0.9)
    grow_max_dilate_frac: float = 0.05    # ... only within this distance of the seed

    # --- GrabCut (seeded from the saliency mask) ---
    grabcut_iters: int = 5
    gc_sure_fg_frac: float = 0.03
    gc_prob_fg_frac: float = 0.02
    gc_prob_bg_frac: float = 0.10
    gc_sure_bg_sal: float = 0.4

    # --- Active contours (morphological Chan-Vese on sqrt(saliency)) ---
    ac_iters: int = 100
    ac_smoothing: int = 3
    ac_init_erode_frac: float = 0.03
    ac_gamma: float = 0.5

    # --- Ensemble ---
    ensemble_weights: tuple = (2, 1, 1)   # v5: Otsu is the strongest member, so it counts double ((1, 1, 1) = plain majority)
    ensemble_min_votes: int = 2

    # --- Ground truth ---
    consensus_tag: str = "ST"             # "ST" or "MV"

    # --- Calibration stickers / coloured foreign objects (excluded like the vignette) ---
    # Validated on all 2000 images against GT lesion masks (see detect_stickers docstring).
    sticker_detect: bool = True
    # colour candidates (OpenCV Lab, a/b centred on 0)
    sticker_a_max: float = -8.0           # green / teal / cyan: a below this (no GT lesion pixel reaches it)
    sticker_b_max: float = -15.0          # blue / violet: b below this AND below skin b - sticker_min_rel
    sticker_vivid_chroma: float = 45.0    # saturated red / orange / yellow: chroma above this ...
    sticker_vivid_sat: float = 120.0      # ... and HSV saturation above this (lesions can reach it: see guard)
    sticker_min_rel: float = 20.0         # every candidate this far from the image's median skin colour
    sticker_min_area_frac: float = 0.008
    # shape
    sticker_min_solidity: float = 0.85
    sticker_min_circle_inliers: float = 0.6   # share of the free outline within 4% of r from the fitted circle
    sticker_min_radius_frac: float = 0.14     # fitted radius / min(h, w); real stickers >= 0.19, lesion blobs <= 0.08
    sticker_min_disc_cover: float = 0.6       # the fitted disc must be filled by the component
    # concentric rings
    sticker_ring_skin_chroma: float = 8.0     # grey rings only searched when skin chroma >= this
    sticker_ring_grey_frac: float = 0.45      # grey = chroma below this fraction of skin chroma
    sticker_ring_min_frac: float = 0.6        # shell kept while this share of it is ring material
    sticker_ring_gap: float = 0.08            # tolerated gap (fraction of r) between centre and ring
    sticker_ring_max: float = 2.3             # rings searched out to this multiple of r
    # vascular-lesion guard: vivid disc fully inside the field needs one of these
    sticker_min_clipped: float = 0.03         # share of hull outline on the image / FOV edge
    sticker_ring_min_ratio: float = 1.25      # ring radius / centre radius
    sticker_min_unnatural: float = 0.15       # share of green / blue pixels
    sticker_max_frac: float = 0.45            # per sticker; larger components are rejected individually
    sticker_margin_frac: float = 0.015        # safety margin (v1 used 0.045, which spread onto skin)


CFG = PreprocessConfig()
SEG_CFG = SegmentationConfig()