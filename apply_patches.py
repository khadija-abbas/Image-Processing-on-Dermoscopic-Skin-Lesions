"""Apply the tested segmentation fixes to your project.

    python apply_patches.py            # run from the project root (the folder that contains src/)

What it changes (backups are written as *.bak; it refuses to run if a patch point is not found):
  src/segmentation.py
    1. clean_mask(): picks the lesion-like component (saliency, solidity, distance from sticker)
       instead of the biggest one, and the three call sites now pass sal=...
    2. detect_stickers(): a clipped sticker may be less solid (0.65 instead of 0.85)
    3. detect_stickers(): grey / low-chroma stickers are also sticker candidates
  src/config.py
    4. sticker_ring_skin_chroma 8.0 -> 4.0   (grey rings are searched on pale-skin images too)
    5. vignette_min_area_frac  0.01 -> 0.002 (small black corner triangles are masked too)
"""
import re
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
SEG, CFG = ROOT / "src" / "segmentation.py", ROOT / "src" / "config.py"

CLEAN_MASK_V2 = r"""
def clean_mask(mask, valid, cfg, sal=None):
    m = (mask & valid).astype(np.uint8)
    k = _odd(max(3, round(cfg.clean_open_frac * min(m.shape))))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, labels, stats, cents = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return np.zeros(m.shape, bool)
    h, w = m.shape
    near_vig = cv2.dilate((~valid).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0

    use_sal = sal is not None and getattr(cfg, "select_by_saliency", True)
    if use_sal and (~valid).any():
        # distance of every pixel to the nearest sticker / vignette pixel
        dist_inv = cv2.distanceTransform(valid.astype(np.uint8), cv2.DIST_L2, 3)
        halo = dist_inv < getattr(cfg, "halo_dist_frac", 0.08) * min(h, w)
    else:
        halo = np.zeros((h, w), bool)

    touching = {}
    best, best_score = 1, -1.0
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        comp = labels == i
        touching[i] = x == 0 or y == 0 or x + bw == w or y + bh == h or near_vig[comp].any()
        cx, cy = cents[i]
        d = np.hypot((cx - w / 2) / w, (cy - h / 2) / h)
        central = max(0.05, 1 - 2 * d)
        pen = 0.3 if touching[i] else 1.0
        if use_sal:
            cnts, _ = cv2.findContours(comp.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            hull_area = max(cv2.contourArea(cv2.convexHull(max(cnts, key=cv2.contourArea))), 1.0)
            solidity = min(1.0, area / hull_area)
            contrast = max(float(sal[comp].mean()), 0.05)
            halo_frac = float(halo[comp].mean())
            score = (np.sqrt(area) * contrast * solidity ** 2 * central * pen
                     * (1.0 - 0.85 * halo_frac))
        else:
            score = area * central * pen
        if score > best_score:
            best, best_score = i, score

    keep = labels == best
    if n > 2 and cfg.merge_min_area_frac > 0:
        dist = ndi.distance_transform_edt(~keep)
        for i in range(1, n):
            if i == best or touching[i] or stats[i, 4] < cfg.merge_min_area_frac * stats[best, 4]:
                continue
            comp = labels == i
            if dist[comp].min() <= cfg.merge_dist_frac * min(h, w):
                keep |= comp
    out = keep.astype(np.uint8)
    kc = _odd(max(3, round(cfg.clean_close_frac * min(h, w))))
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kc, kc)))
    return ndi.binary_fill_holes((out > 0) & valid)

"""


def read(p):
    raw = p.read_bytes().decode("utf-8")
    return raw.replace("\r\n", "\n"), "\r\n" in raw


def write(p, text, crlf):
    p.with_suffix(p.suffix + ".bak").write_bytes(p.read_bytes())
    p.write_bytes((text.replace("\n", "\r\n") if crlf else text).encode("utf-8"))


def sub1(s, old, new):
    n = s.count(old)
    assert n == 1, f"patch point found {n}x (expected 1): {old[:70]!r}"
    return s.replace(old, new)


def patch_segmentation(text):
    # the file keeps old versions as comments; only touch the live code
    marker = "\nimport cv2\nimport numpy as np\nfrom scipy import ndimage as ndi\n"
    assert text.count(marker) == 1, "could not find the start of the live code"
    k = text.index(marker)
    head, s = text[:k], text[k:]
    a = s.index("\ndef clean_mask(mask, valid, cfg, sal=None):")
    b = s.index("\ndef _skin_band")
    s = s[:a] + "\n" + CLEAN_MASK_V2.strip("\n") + "\n" + s[b:]
    s = sub1(s, "return clean_mask((u8 > t) & valid, valid, cfg), t",
             "return clean_mask((u8 > t) & valid, valid, cfg, sal=sal), t")
    s = sub1(s, "    return clean_mask(grown, valid, cfg)\n\ndef prepare",
             "    return clean_mask(grown, valid, cfg, sal=sal)\n\ndef prepare")
    s = sub1(s, "cleaned[name] = clean_mask(fn(ctx, cfg), valid, cfg)",
             'cleaned[name] = clean_mask(fn(ctx, cfg), valid, cfg, sal=ctx["sal"])')
    s = sub1(s, "ens = clean_mask(votes >= cfg.ensemble_min_votes, valid, cfg)",
             'ens = clean_mask(votes >= cfg.ensemble_min_votes, valid, cfg, sal=ctx["sal"])')
    s = sub1(s, """        if solidity < cfg.sticker_min_solidity:
            rec["why"] = "solidity"; continue""",
             """        min_sol = cfg.sticker_min_solidity if clipped < 0.2 else 0.65   # clipped stickers: notch where the inner disc meets the frame
        if solidity < min_sol:
            rec["why"] = "solidity"; continue""")
    s = sub1(s, "    cand = valid & (rel > cfg.sticker_min_rel) & (green | blue | vivid)\n",
             """    cand_col = (rel > cfg.sticker_min_rel) & (green | blue | vivid)
    c_all = float(np.median(C[valid]))
    grey = (C < cfg.sticker_ring_grey_frac*c_all) if c_all >= cfg.sticker_ring_skin_chroma else np.zeros_like(cand_col)
    cand = valid & (cand_col | grey)      # grey stickers have no colour to trigger the tests above
""")
    return head + s


def patch_config(text):
    for key, old, new in (("sticker_ring_skin_chroma", "8.0", "4.0"),
                          ("vignette_min_area_frac", "0.01", "0.002")):
        pat = re.compile(rf"(\n    {key}: float = ){re.escape(old)}\b")
        assert len(pat.findall(text)) == 1, f"{key} = {old} not found in config.py (already changed?)"
        text = pat.sub(lambda m: m.group(1) + new, text)
    return text


if __name__ == "__main__":
    seg, crlf1 = read(SEG)
    cfg, crlf2 = read(CFG)
    seg2, cfg2 = patch_segmentation(seg), patch_config(cfg)     # both succeed before anything is written
    write(SEG, seg2, crlf1)
    write(CFG, cfg2, crlf2)
    print("Patched src/segmentation.py and src/config.py (backups: *.bak)")