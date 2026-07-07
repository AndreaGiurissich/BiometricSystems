"""Build raw-vs-preprocessed montages for the contrast-mask pipeline.

This is the preprocessing GATE: it surfaces, as side-by-side panels, what the
operator actually does to SOCOFing images BEFORE the full evaluation matrix runs.
Each row shows one image and every intermediate stage:

    raw | denoised | laplacian | mask | soft-mask | sharpened | final

Samples are seeded (configs/default.yaml -> preprocessing.montage): n_real reals
plus n_per_level probes for each level. Montages are written to
<results_dir>/figures/preprocessing/ (gitignored) for human review.

Usage:
    python scripts/make_montages.py [--config configs/default.yaml]
                                    [--levels Easy,Medium,Hard]
                                    [--input-root /kaggle/input]
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, resolve_paths  # noqa: E402
from src import dataset as ds  # noqa: E402
from src.preprocessing import contrast_mask  # noqa: E402

# Fixed display size per panel. SOCOFing altered images vary in size (Zcut/CR/Obl
# change the canvas), so panels are resized to a common box for the grid -- the
# preprocessing itself still runs on each image's NATIVE pixels.
PANEL_W, PANEL_H = 192, 208   # ~2x SOCOFing's ~96x103 (near-undistorted common case)
SEP = 4            # white separator (px) between panels
PANELS = [         # (Stages attribute or 'raw', column label)
    ("raw", "raw"),
    ("denoised", "denoised"),
    ("laplacian", "laplacian"),
    ("mask_binary", "mask"),
    ("soft_mask", "soft-mask"),
    ("sharpened", "sharpened"),
    ("final", "final"),
]


def _to_bgr_u8(arr: np.ndarray) -> np.ndarray:
    """Normalise any stage array to a fixed-size 3-channel uint8 panel."""
    if arr.dtype != np.uint8:  # soft_mask is float [0,1]
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    big = cv2.resize(arr, (PANEL_W, PANEL_H), interpolation=cv2.INTER_NEAREST)
    return cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)


def _row(raw: np.ndarray, stages) -> np.ndarray:
    """One montage row: the panels for a single image, horizontally joined."""
    cells = []
    for attr, _ in PANELS:
        arr = raw if attr == "raw" else getattr(stages, attr)
        cells.append(_to_bgr_u8(arr))
    sep = np.full((PANEL_H, SEP, 3), 255, np.uint8)
    joined = cells[0]
    for cell in cells[1:]:
        joined = np.hstack([joined, sep, cell])
    return joined


def _header(row_w: int) -> np.ndarray:
    """A label strip naming each column, aligned to the panel grid."""
    strip = np.full((22, row_w, 3), 245, np.uint8)
    for i, (_, label) in enumerate(PANELS):
        x = i * (PANEL_W + SEP) + 4
        cv2.putText(strip, label, (x, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (0, 0, 0), 1, cv2.LINE_AA)
    return strip


def _montage(records, cfg, title: str, out_path: Path) -> int:
    rows = []
    for rec in records:
        img = cv2.imread(rec.path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"    skip (unreadable): {rec.filename}")
            continue
        _, stages = contrast_mask(img, cfg, return_stages=True)
        rows.append(_row(img, stages))
    if not rows:
        print(f"  {title}: no readable images, skipped")
        return 0
    grid = np.vstack(rows)
    grid = np.vstack([_header(grid.shape[1]), grid])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), grid)
    print(f"  {title}: {len(rows)} rows -> {out_path}")
    return len(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="Contrast-mask montages (preprocessing gate).")
    ap.add_argument("--config", default=None)
    ap.add_argument("--levels", default="Easy,Medium,Hard")
    ap.add_argument("--dataset-root", default=None)
    ap.add_argument("--input-root", default="/kaggle/input")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.dataset_root:
        active = cfg["paths"]["active_profile"]
        cfg["paths"]["profiles"][active]["dataset_root"] = args.dataset_root
    paths = resolve_paths(cfg, input_root=args.input_root)

    m = cfg["preprocessing"]["montage"]
    rng = random.Random(m["seed"])
    out_dir = Path(paths["results_dir"]) / "figures" / "preprocessing"

    print("== contrast-mask montages ==")
    gallery, *_ = ds.build_gallery(paths["real_dir"])
    reals = list(gallery.values())
    sample = rng.sample(reals, min(m["n_real"], len(reals)))
    _montage(sample, cfg, "Real", out_dir / "montage_real.png")

    for level in [s for s in args.levels.split(",") if s.strip()]:
        level_dir = paths["level_dirs"].get(level)
        if not level_dir or not Path(level_dir).exists():
            print(f"  {level}: level dir missing, skipped")
            continue
        probes, *_ = ds.build_probes(level_dir, gallery)
        if not probes:
            print(f"  {level}: no probes, skipped")
            continue
        sample = rng.sample(probes, min(m["n_per_level"], len(probes)))
        _montage(sample, cfg, level, out_dir / f"montage_{level.lower()}.png")

    print(f"\n  review the PNGs under {out_dir} before the full matrix.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
