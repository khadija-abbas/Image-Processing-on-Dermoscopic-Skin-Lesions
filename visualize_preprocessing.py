"""Visualize original images against the saved Preprocessing V2 outputs."""

import argparse
import random

import cv2
import matplotlib.pyplot as plt

from src.config import IMAGE_EXTENSIONS, RAW_DIR, REPORTS_DIR, OUT_DIR, OUT_NO_CLAHE_DIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Find original images
    paths = {
        p.stem: p
        for p in RAW_DIR.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    }

    assert paths, f"No images found in {RAW_DIR}"

    # Select images that have already been processed by V2
    available = [
        stem for stem in sorted(paths)
        if (OUT_DIR / f"{stem}.png").exists()
        and (OUT_NO_CLAHE_DIR / f"{stem}.png").exists()
    ]

    assert available, "No Preprocessing V2 outputs found."

    selected = random.Random(args.seed).sample(
        available, min(args.n, len(available))
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        len(selected), 3,
        figsize=(15, 4 * len(selected)),
        squeeze=False
    )

    for row, stem in enumerate(selected):

        original = cv2.imread(str(paths[stem]))
        clahe = cv2.imread(str(OUT_DIR / f"{stem}.png"))
        noclahe = cv2.imread(str(OUT_NO_CLAHE_DIR / f"{stem}.png"))

        panels = [
            (original, f"{stem}: original"),
            (noclahe, "V2: no CLAHE"),
            (clahe, "V2: CLAHE"),
        ]

        for col, (im, title) in enumerate(panels):
            if im is None:
                raise ValueError(f"Could not read image: {stem}")

            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

            axes[row, col].imshow(im)
            axes[row, col].set_title(title, fontsize=9)
            axes[row, col].axis("off")

    plt.tight_layout()

    out = REPORTS_DIR / "preprocessing_v2_visualization.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")

    print(f"Saved V2 visualization to {out}")
    print(f"Images displayed: {len(selected)}")

    plt.show()


if __name__ == "__main__":
    main()
