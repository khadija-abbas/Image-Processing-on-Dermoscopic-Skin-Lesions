# Image Processing on Dermoscopic Skin Lesions

Classical (non-deep-learning) pipeline for **preprocessing** and **segmenting** skin lesions in dermoscopic images from the ISIC / IMA++ collection, scored against expert ground-truth masks with **Dice** and **IoU**. An optional learned **candidate selector** sits on top of the classical methods.

```
raw images ─► preprocessing ─► skin-colour saliency ─► seed ─► Otsu / GrabCut / Active contour ─► ensemble ─► scores
                                                                          └─► (optional) candidate selector
```

---

## 1. Setup

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

`requirements.txt`: opencv-python, numpy, pandas, matplotlib, tqdm, requests, scikit-image, scipy, scikit-learn, joblib.

### Data

If the `Images/` folder is empty, download the images (ISIC collection 482, up to 2000 images):

```bash
python download_isic.py
```

Expected layout (paths are set in `src/config.py`):

| Path | Contents |
|---|---|
| `Images/` | original images |
| `segs/` | ground-truth masks (single-annotator and `MV` / `ST` consensus masks) |
| `ima_metadata_2026-10-22.csv` | benign / malignant labels (`LABEL_COLUMN = "diagnosis_1"`) |
| `preprocessed/`, `preprocessed_no_clahe/` | preprocessing output (generated) |
| `segmentation_5000_images/<method>/` | predicted masks (generated) |
| `reports/` | metrics CSVs and overlay images (generated) |

Ground truth: the **ST (STAPLE) consensus** mask is used when present (`consensus_tag` in `SegmentationConfig`), otherwise the single annotator's mask.

---

## 2. Workflow

### Step 1: Preprocessing

```bash
python run_preprocessing.py --limit 20             # small test run
python visualize_preprocessing.py                  # look at random samples (--n, --seed)
python run_preprocessing.py --overwrite --limit 20 # rerun after tuning config.py
python run_preprocessing.py                        # full run (resumes if interrupted)
```

What it does (`src/preprocessing.py`): hair and ruler-mark removal, shading (uneven illumination) correction, optional Shades-of-Gray colour constancy, light denoising, and a brightness-preserving CLAHE. It writes **two** versions of each image: with CLAHE and without. Segmentation uses the **no-CLAHE** version by default (`use_clahe = False`).

Useful flags: `--preview N`, `--save-masks`, `--color-constancy`, `--no-shading`, `--plain-clahe`, `--workers`.

### Step 2: Segmentation

```bash
python run_segmentation.py --limit 20       # small test; prints GT sources, label counts, timings
python run_segmentation.py                  # full run (resumes; --overwrite to redo)
python run_segmentation.py --ids ISIC_0000004 ISIC_0000008
python run_segmentation.py --range 1 200    # by sorted position, inclusive
```

Outputs: predicted masks in `segmentation_5000_images/<method>/<ISIC_id>.png` and `reports/segmentation_metrics_v6.csv` (one row per image and method). It prints mean / median Dice and IoU by method and label.

### Step 3: Look at results

```bash
python visualize_segmentation.py --n 5 --label malignant
python visualize_segmentation.py --mode worst --n 6
python show_best.py --mode best  --n 10 --csv segmentation_metrics_v6.csv --method ensemble
python show_best.py --mode worst --n 20 --tag sticker --csv segmentation_metrics_v6.csv --method ensemble
```

Overlays: **green = ground truth, red = prediction**. `show_best.py` can filter by `--tag hair|sticker`, `--id-min` / `--id-max`, and `--mode best|worst|random|median`. Its default `--csv` name is out of date, so pass `--csv segmentation_metrics_v6.csv` as above.

### Step 4: Re-score saved masks (no re-segmentation)

```bash
python score_saved_masks.py                  # recompute Dice/IoU from the saved masks
python score_saved_masks.py --csv            # just summarise reports/segmentation_metrics_v6.csv
python score_saved_masks.py --exclude ISIC_0000991 ISIC_0000647   # leave out bad-ground-truth images
```

Prints the mean/median table by method and label, plus extras: Dice = 0 counts and sticker vs no-sticker split. Writes `reports/segmentation_metrics_recomputed.csv`.

### Step 5: Understand failures

```bash
python analyze_failures.py                   # failure categories + worst images (default method: ensemble)
python diagnose_segmentation.py              # Dice by ID range / lesion size / vignette; flipped-GT test
python -m src.diagnose_stickers --ids ISIC_0001886 ISIC_0001065   # why a sticker blob was accepted/rejected
python compare_runs.py old.csv new.csv       # regressions between two runs (splits ISIC >= 558 = stickers)
python calibrate_size.py --method ensemble   # does growing/shrinking masks help? (even/odd split)
python visualise_valid_region.py             # check the dermoscope field-of-view detection
```

Failure categories in `analyze_failures.py`: **good** (Dice ≥ 0.8), **over-segmented** (prediction ≥ 1.5× ground-truth area), **under-segmented** (≤ 0.6×), **displaced** (Dice < 0.6 with similar area), **boundary-off** (everything else).

### Step 6 (optional): Learned candidate selector

```bash
python build_candidates.py        # ~20 candidate masks + features + true Dice per image -> candidates/<id>.pkl
python train_selector.py          # 5-fold CV (grouped by image) -> selected masks, metrics, model
```

Outputs: `segmentation_5000_images/selected/<id>.png`, `reports/segmentation_metrics_selected.csv`, `models/selector.joblib`. A gradient-boosting model predicts each candidate's Dice from shape, contrast, border and agreement features, and the final mask is a soft vote of the top 5 candidates. Reported scores are **out-of-fold**. If `dl_probs/<id>.png` exists (from `dl_unet.py`, which is **not** in this repo) it is used as extra features and one extra candidate.

---

## 3. How segmentation works

All methods share one signal (the saliency map) and one starting mask (the seed), both built in `prepare()` in `src/segmentation.py`.

1. **Valid region.** Find the dermoscope's round field of view (convex hull of the bright region), remove black border vignettes, and remove coloured **calibration stickers** (`detect_stickers`: colour candidates, circle fit, concentric rings, vascular-lesion guard). Shrink the valid area slightly to drop the blurry rim.
2. **Skin model.** Fit the patient's healthy-skin colour in Lab space from a ring of valid pixels near the edge (trimmed Gaussian, repeatedly dropping the worst-fitting pixels).
3. **Saliency map.** For every pixel, the Mahalanobis distance from that skin model: near 0 for skin, high for anything different. A morphological opening removes thin structures (hairs, ruler ticks). Polarity is built in: "different from skin" always means lesion.
4. **Seed.** Otsu threshold on the saliency map (on a compressed `sal ** 0.5` scale, relaxed to 0.8× so the mask reaches the generous annotator outlines), with sanity checks on seed size.
5. **Refinement of the seed.** Refit the skin model on everything *outside* the seed with a radial illumination model; a polarity check (if the "seed" is actually skin, the lesion fills the field); a whole-field fallback when there is almost no colour contrast; hysteresis growth toward weaker saliency.
6. **Cleanup (`clean_mask`).** Opening, then keep the component that scores best on size, contrast, solidity, centrality and distance from the border / stickers; merge large nearby fragments; close gaps; fill holes. `regularize_shape` replaces a very ragged mask with its convex hull when that changes the area by at most 1.4×.

### The three methods and the ensemble

| Method | Idea | Uses saliency |
|---|---|---|
| **Otsu** | Threshold the saliency map; the result of step 5 is the mask. | Fully |
| **GrabCut** | The seed builds a trimap (sure lesion = eroded seed, probable lesion, probable skin, sure skin). GrabCut then re-decides the uncertain pixels from the real image colours with a graph cut. | Seed and sure-background only |
| **Active contour** | Morphological Chan-Vese: start from a shrunk seed, evolve a smooth curve on `sqrt(saliency)` for 100 iterations until inside and outside are as different as possible. | Fully |
| **Ensemble** | Weighted pixel vote of the three, weights `(2, 1, 1)` (Otsu counts double), at least 2 votes, then cleaned again. | Indirectly |

The three methods are **not independent**: they all start from the same seed, so a wrong seed makes all of them wrong and the ensemble mostly smooths boundaries.

Masks are computed at a working resolution (longest side 384 px) and upsampled to full resolution; scoring is done at full resolution.

---

## 4. Example results

Output of `run_segmentation.py` from the full 2000-image run (mean / median **Dice**; 1805 benign, 195 malignant). Re-run to refresh these numbers after any change.

| Method | Benign | Malignant |
|---|---|---|
| Otsu | 0.820 / 0.885 | 0.821 / 0.908 |
| GrabCut | 0.772 / 0.848 | 0.821 / 0.912 |
| Active contour | 0.801 / 0.892 | 0.814 / 0.900 |
| **Ensemble** | **0.821 / 0.885** | **0.829 / 0.912** |

---

## 5. Known limitations

- **Ground-truth quality.** Many of the lowest-scoring images (Dice = 0) have ground truth that sits on a sticker or blank skin while the prediction lands on a clearly visible mole. Check these before treating them as segmentation failures, and consider `--exclude` when reporting.
- **Hard cases for the method itself:** lesions touching the image border, very pale or tiny lesions with almost no colour contrast, stickers covering a large part of the image, and multiple lesions.
- **Tuning on the evaluation set.** Several thresholds (`otsu_relax`, the sticker limits, and others) were tuned on the same ~2000 images used for reporting, so classical-method scores are somewhat optimistic. Only the selector uses grouped cross-validation.
- **Inter-method dependence.** The ensemble cannot fix a seed that is wrong.
- `dl_unet.py` is referenced by the candidate builder but is not included.
- Hidden config defaults: `halo_dist_frac` and `select_by_saliency` are read with `getattr` and are not defined in `config.py`.

---

## 6. Repository map

```
download_isic.py              download images from the ISIC API
run_preprocessing.py          preprocessing entry point
run_segmentation.py           segmentation + scoring entry point
score_saved_masks.py          re-score saved masks without re-segmenting
visualize_preprocessing.py    preprocessing samples
visualize_segmentation.py     segmentation overlays (random / worst)
show_best.py                  best / worst / median overlays with hair and sticker filters
analyze_failures.py           failure categories and worst images
diagnose_segmentation.py      Dice by range / size / vignette; flipped-GT test
compare_runs.py               compare two metrics CSVs
calibrate_size.py             test growing/shrinking masks
visualise_valid_region.py     field-of-view check
build_candidates.py           candidate masks + features
train_selector.py             candidate selector (CV) + final model
src/config.py                 all paths and tunable parameters
src/preprocessing.py          hair removal, shading, CLAHE
src/segmentation.py           valid region, stickers, saliency, seed, methods, ensemble
src/candidates.py             candidate generation and features
src/ground_truth.py           ground-truth and label loading
src/metrics.py                Dice and IoU
src/diagnose_stickers.py      sticker-detection debugging
independent_experiments/      stand-alone field-of-view and mask diagnostics
models/selector.joblib        trained selector
```

Reference: the ensemble idea follows Tamoor et al. (2023).