"""Segmentation scores."""
import numpy as np


def dice_iou(pred: np.ndarray, gt: np.ndarray):
    """Dice coefficient and IoU between two boolean masks of the same shape."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    assert pred.shape == gt.shape, f"shape mismatch {pred.shape} vs {gt.shape}"
    inter = int(np.logical_and(pred, gt).sum())
    p, g = int(pred.sum()), int(gt.sum())
    union = p + g - inter
    dice = 2.0 * inter / (p + g) if (p + g) > 0 else 1.0
    iou = inter / union if union > 0 else 1.0
    return float(dice), float(iou)