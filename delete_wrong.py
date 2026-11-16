"""Delete the images listed in wrong.txt from segmentation_5000_images/ (every sub-folder).

wrong.txt can look like:   * `0000991`     or     0000991     or     ISIC_0000991
Each ID is matched to files named ISIC_<id>.png (.jpg/.jpeg also accepted).

Usage (from the repo root, venv active):
    python delete_wrong.py              # DRY RUN: only shows what would be deleted
    python delete_wrong.py --delete     # actually deletes the files
"""
import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WRONG_FILE = ROOT / "wrong.txt"
TARGET_DIR = ROOT / "segmentation_5000_images"   # contains active_contour, ensemble, grabcut, otsu, selected
EXTENSIONS = {".png", ".jpg", ".jpeg"}


def read_ids(path: Path) -> list[str]:
    """Return the unique 7-digit ISIC ids found in the text file, in order."""
    text = path.read_text(encoding="utf-8")
    ids = re.findall(r"(?:ISIC_)?(\d{7})", text)
    return list(dict.fromkeys(ids))          # remove duplicates, keep order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delete", action="store_true",
                    help="really delete the files (default is a dry run)")
    args = ap.parse_args()

    assert WRONG_FILE.exists(), f"Not found: {WRONG_FILE}"
    assert TARGET_DIR.exists(), f"Not found: {TARGET_DIR}"

    ids = read_ids(WRONG_FILE)
    print(f"IDs in wrong.txt: {len(ids)}")
    print(f"Mode: {'DELETE' if args.delete else 'DRY RUN (nothing will be deleted)'}\n")

    deleted, not_found = 0, []
    for i in ids:
        matches = [p for p in TARGET_DIR.rglob(f"ISIC_{i}.*")
                   if p.is_file() and p.suffix.lower() in EXTENSIONS]
        if not matches:
            not_found.append(i)
            continue
        for p in matches:
            print(f"{'deleting' if args.delete else 'would delete'}: {p.relative_to(ROOT)}")
            if args.delete:
                p.unlink(missing_ok=True)
            deleted += 1

    print(f"\n{'Deleted' if args.delete else 'Would delete'}: {deleted} files")
    if not_found:
        print(f"Skipped, already deleted or not found ({len(not_found)}): {', '.join(not_found)}")
    if not args.delete:
        print("\nLooks right? Run again with:  python delete_wrong.py --delete")


if __name__ == "__main__":
    main()
