"""Save a before/after preview grid so you can check the preprocessing visually."""
import argparse
import random

import cv2
import matplotlib.pyplot as plt
import pandas as pd

from src.config import CFG, IMAGE_EXTENSIONS, RAW_DIR, REPORTS_DIR
from src.preprocessing import run_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--mode", choices=["random", "hairiest"], default="random",
                        help="'hairiest' uses the log from run_preprocessing.py")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    paths = {p.stem: p for p in RAW_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS}
    assert paths, f"No images found in {RAW_DIR}"

    if args.mode == "hairiest":
        log = pd.read_csv(REPORTS_DIR / "preprocessing_log.csv")
        stems = log.dropna(subset=["hair_fraction"]) \
                   .sort_values("hair_fraction", ascending=False).head(args.n)["image"].tolist()
    else:
        stems = random.Random(args.seed).sample(sorted(paths), min(args.n, len(paths)))

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(stems), 4, figsize=(16, 4 * len(stems)), squeeze=False)
    for row, stem in enumerate(stems):
        img = cv2.imread(str(paths[stem]), cv2.IMREAD_COLOR)
        res = run_pipeline(img, CFG)
        panels = [
            (cv2.cvtColor(img, cv2.COLOR_BGR2RGB), f"{stem}: original"),
            (res.hair_mask, f"hair mask ({res.hair_fraction:.2%})"),
            (cv2.cvtColor(res.hair_removed, cv2.COLOR_BGR2RGB), "hair removed"),
            (cv2.cvtColor(res.final, cv2.COLOR_BGR2RGB), "final (median + CLAHE)"),
        ]
        for col, (im, title) in enumerate(panels):
            axes[row, col].imshow(im, cmap="gray" if im.ndim == 2 else None)
            axes[row, col].set_title(title, fontsize=9)
            axes[row, col].axis("off")
    plt.tight_layout()
    out = REPORTS_DIR / "preprocessing_preview.png"
    fig.savefig(out, dpi=120)
    print(f"Saved preview to {out}")
    plt.show()


if __name__ == "__main__":
    main()