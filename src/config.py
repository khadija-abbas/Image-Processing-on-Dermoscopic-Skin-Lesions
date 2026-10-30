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
SEG_OUT_DIR = PROJECT_ROOT / "segmentation"                 # one sub-folder per method
GT_DIR = Path(r"D:\segs")                                   # EDIT: folder with the IMA++ masks
LABELS_FILE = DATA_DIR / "ima_metadata_2026-10-29.csv"      # EDIT: metadata file (.csv or .xlsx)
LABEL_COLUMN = "diagnosis_1"                                # EDIT if the benign/malignant column is named differently
MASK_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class PreprocessConfig:
    # --- Hair detection (black-hat) ---
    hair_blur_sigma: float = 1.0          # light blur before black-hat so noise isn't flagged
    hair_kernel_frac: float = 0.025       # black-hat kernel as a fraction of min(height, width)
    hair_kernel_min: int = 13             # lower bound for the kernel size (pixels)
    hair_threshold: int = 20              # starting threshold on the black-hat response
    hair_threshold_step: int = 5          # raise by this much while the raw mask is too large
    hair_threshold_max: int = 80
    hair_max_raw_fraction: float = 0.15   # raw mask above this fraction => threshold is too low

    # --- Shape filters: keep only thin, long, elongated structures ---
    hair_max_width_frac: float = 0.012    # wider structures (dark dots, lesion edges) are not hair
    hair_max_width_min: int = 5
    hair_min_length_frac: float = 0.04    # shorter fragments are treated as speckle / lesion globules
    hair_min_length_min: int = 15
    hair_min_elongation: float = 3.0      # bbox length / sqrt(area); blobs and short dashes score low
    hair_border_px: int = 3               # ignore detections within this many pixels of the border

    # --- Inpainting ---
    hair_dilate_iter: int = 1             # grow the mask slightly to cover hair edges
    inpaint_radius: int = 3

    # --- Noise / artifact suppression ---
    apply_median: bool = True
    median_ksize: int = 3                 # must be odd

    # --- CLAHE (applied to the L channel of Lab only) ---
    clahe_clip_limit: float = 1.5
    clahe_tile_grid: tuple = (8, 8)


@dataclass(frozen=True)
class SegmentationConfig:
    # --- Working resolution (masks are upsampled back to full size for scoring) ---
    work_max_side: int = 512              # Otsu and GrabCut
    ac_max_side: int = 256                # active contours (slowest, so smaller)

    # --- Black dermoscope vignette (excluded from every method) ---
    vignette_dark_thresh: int = 20        # gray level below which a pixel counts as "black"
    vignette_min_area_frac: float = 0.01  # dark region must touch the border and be this big
    vignette_max_frac: float = 0.6        # if "vignette" covers more than this, ignore it

    # --- Shared post-processing (identical for all methods) ---
    clean_open_frac: float = 0.01         # opening kernel as a fraction of min(height, width)
    border_ring_frac: float = 0.05        # border ring width used to decide which side is the lesion

    # --- Otsu ---
    otsu_blur_ksize: int = 5

    # --- GrabCut ---
    grabcut_margin_frac: float = 0.10     # rectangle = image minus this margin on every side
    grabcut_iters: int = 5

    # --- Active contours (morphological Chan-Vese) ---
    ac_iters: int = 150
    ac_smoothing: int = 2                 # higher = smoother contour, lower = more irregular
    ac_init_radius_frac: float = 0.30     # initial disk radius as a fraction of min(height, width)
    ac_blur_sigma: float = 2.0

    # --- Ground truth ---
    consensus_tag: str = "ST"             # "ST" or "MV": consensus mask used when an image has several annotators


CFG = PreprocessConfig()
SEG_CFG = SegmentationConfig()