"""Second patch, to run AFTER apply_patches.py (from the project root):

    python apply_fix2.py

1. segmentation.py: grey/low-chroma stickers are now searched in a SEPARATE pass that can only ADD to
   the mask. In apply_patches.py they were mixed into the colour candidates, which on pale-skin images
   (e.g. ISIC_0000785) merged the sticker with its surroundings and made detection fail.
2. config.py: looser colour thresholds so dull / dark / pale-blue stickers are found
   (sticker_a_max -8 -> -5, sticker_b_max -15 -> -9, sticker_min_rel 20 -> 14).
Backups: *.bak2.  Nothing is written unless every patch point is found.
"""
import re
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
SEG, CFG = ROOT / "src" / "segmentation.py", ROOT / "src" / "config.py"

OLD_GREY = """    cand_col = (rel > cfg.sticker_min_rel) & (green | blue | vivid)
    c_all = float(np.median(C[valid]))
    grey = (C < cfg.sticker_ring_grey_frac*c_all) if c_all >= cfg.sticker_ring_skin_chroma else np.zeros_like(cand_col)
    cand = valid & (cand_col | grey)      # grey stickers have no colour to trigger the tests above
"""
NEW_GREY = """    cand = valid & (rel > cfg.sticker_min_rel) & (green | blue | vivid)
    if use_grey:                          # second pass only: grey / low-chroma stickers
        c_all = float(np.median(C[valid]))
        if c_all >= cfg.sticker_ring_skin_chroma:
            cand = cand | (valid & (C < cfg.sticker_ring_grey_frac*c_all))
"""
WRAPPER = """def detect_stickers(bgr, valid, cfg, debug=None):
    out = _detect_stickers(bgr, valid, cfg, debug, use_grey=False)       # colour pass (as before)
    return out | _detect_stickers(bgr, valid, cfg, None, use_grey=True)  # grey pass can only add


def detect_vignette(bgr, cfg, return_masks=False):"""


def read(p):
    raw = p.read_bytes().decode("utf-8")
    return raw.replace("\r\n", "\n"), "\r\n" in raw


def write(p, text, crlf):
    p.with_suffix(p.suffix + ".bak2").write_bytes(p.read_bytes())
    p.write_bytes((text.replace("\n", "\r\n") if crlf else text).encode("utf-8"))


def sub_last(s, old, new):
    """Replace the LAST occurrence (the live code comes after the commented-out old versions)."""
    n = s.count(old)
    assert n >= 1, f"patch point not found: {old[:70]!r}"
    k = s.rindex(old)
    return s[:k] + new + s[k + len(old):]


def patch_seg(s):
    assert s.count(OLD_GREY) == 1, "run apply_patches.py first (or this fix was already applied)"
    s = s.replace(OLD_GREY, NEW_GREY)
    s = sub_last(s, "def detect_stickers(bgr, valid, cfg, debug=None):",
                 "def _detect_stickers(bgr, valid, cfg, debug=None, use_grey=False):")
    return sub_last(s, "def detect_vignette(bgr, cfg, return_masks=False):", WRAPPER)


def patch_cfg(t):
    for key, old, new in (("sticker_a_max", "-8.0", "-5.0"), ("sticker_b_max", "-15.0", "-9.0"),
                          ("sticker_min_rel", "20.0", "14.0")):
        pat = re.compile(rf"(\n    {key}: float = ){re.escape(old)}(?![\d.])")
        assert len(pat.findall(t)) == 1, f"{key} = {old} not found in config.py (already changed?)"
        t = pat.sub(lambda m: m.group(1) + new, t)
    return t


if __name__ == "__main__":
    seg, c1 = read(SEG)
    cfg, c2 = read(CFG)
    seg2, cfg2 = patch_seg(seg), patch_cfg(cfg)
    write(SEG, seg2, c1)
    write(CFG, cfg2, c2)
    print("Patched src/segmentation.py and src/config.py (backups: *.bak2)")