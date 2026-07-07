"""NBIS spike -- run AFTER scaffold/parser, BEFORE building the model wrapper.

Goal: prove the minutiae pipeline works end-to-end on Kaggle and characterize it,
so we can decide whether NBIS is viable on ~96x103 SOCOFing images before investing
in the full wrapper.

It will:
  1. Ensure mindtct + bozorth3 exist (builds them via scripts/build_nbis.sh if not).
  2. Empirically find an input format mindtct accepts (mindtct documents only WSQ
     and ANSI/NIST -- SOCOFing is BMP, so we try wsq -> png -> pgm -> bmp).
  3. Extract .xyt templates for a genuine pair (a probe + its true Real) and an
     impostor Real, logging minutiae counts (relevant at this low resolution).
  4. Run bozorth3 one-to-one for genuine and impostor; print scores.
  5. Write a JSON report to <logs_dir>/nbis_spike.json.

Usage:
    python scripts/spike_nbis.py [--config configs/default.yaml]
                                 [--install-dir /kaggle/working/nbis/install]
                                 [--level Easy]
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, resolve_paths  # noqa: E402
from src import dataset as ds  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_FORMATS = ["wsq", "png", "pgm", "bmp"]  # order = preference


def run(cmd, timeout=300):
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return proc.returncode, proc.stdout, proc.stderr


def ensure_binaries(install_dir: Path) -> tuple[Path, Path]:
    mindtct = install_dir / "bin" / "mindtct"
    bozorth3 = install_dir / "bin" / "bozorth3"
    if mindtct.exists() and bozorth3.exists():
        print(f"  binaries present: {mindtct}")
        return mindtct, bozorth3
    print("  binaries missing -> running scripts/build_nbis.sh (this can take minutes)...")
    rc = subprocess.call(["bash", str(REPO_ROOT / "scripts" / "build_nbis.sh"),
                          str(install_dir)])
    if rc != 0 or not (mindtct.exists() and bozorth3.exists()):
        raise SystemExit(f"NBIS build failed (rc={rc}). Surface the build.log and stop.")
    return mindtct, bozorth3


def _ensure_wsq_support() -> bool:
    """The `wsq` PyPI package registers a Pillow plugin for .wsq save/load."""
    try:
        import wsq  # noqa: F401
        return True
    except ImportError:
        print("  installing 'wsq' (Pillow WSQ plugin) ...")
        subprocess.call([sys.executable, "-m", "pip", "install", "-q", "wsq"])
        try:
            import wsq  # noqa: F401
            return True
        except ImportError:
            print("  WARNING: could not import 'wsq'; skipping WSQ format.")
            return False


def convert_image(src: Path, dst: Path, fmt: str) -> Path | None:
    """Write `src` to `dst` in the requested format; return dst or None on failure."""
    try:
        if fmt == "bmp":
            shutil.copy(src, dst)
            return dst
        from PIL import Image
        img = Image.open(src).convert("L")
        if fmt == "wsq":
            if not _ensure_wsq_support():
                return None
            import wsq  # noqa: F401  (registers the plugin)
            img.save(dst, format="WSQ")
        else:  # png / pgm
            img.save(dst)
        return dst
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"    convert {fmt} failed: {exc}")
        return None


def extract_template(img_path: Path, workdir: Path, mindtct: Path,
                     formats=INPUT_FORMATS):
    """Try formats until mindtct yields a non-empty .xyt. Returns (xyt, fmt, n)."""
    stem = Path(img_path).stem
    for fmt in formats:
        conv = convert_image(Path(img_path), workdir / f"{stem}.{fmt}", fmt)
        if conv is None:
            continue
        oroot = workdir / f"{stem}__{fmt}"
        rc, out, err = run([str(mindtct), str(conv), str(oroot)])
        xyt = Path(str(oroot) + ".xyt")
        if rc == 0 and xyt.exists():
            n = sum(1 for _ in open(xyt))
            print(f"    mindtct OK via '{fmt}': {n} minutiae -> {xyt.name}")
            return xyt, fmt, n
        print(f"    mindtct failed via '{fmt}' (rc={rc}): {err.strip()[:120]}")
    return None, None, 0


def match_pair(probe_xyt: Path, gallery_xyt: Path, bozorth3: Path) -> int:
    rc, out, err = run([str(bozorth3), str(probe_xyt), str(gallery_xyt)])
    if rc != 0:
        print(f"    bozorth3 failed (rc={rc}): {err.strip()[:120]}")
        return -1
    m = re.search(r"-?\d+", out)
    return int(m.group()) if m else -1


def main() -> int:
    ap = argparse.ArgumentParser(description="NBIS (mindtct+bozorth3) spike.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--install-dir", default="/kaggle/working/nbis/install")
    ap.add_argument("--level", default="Easy")
    ap.add_argument("--dataset-root", default=None)
    ap.add_argument("--input-root", default="/kaggle/input")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.dataset_root:
        active = cfg["paths"]["active_profile"]
        cfg["paths"]["profiles"][active]["dataset_root"] = args.dataset_root
    paths = resolve_paths(cfg, input_root=args.input_root)
    report: dict = {"level": args.level}

    print("== NBIS spike ==")
    t0 = time.time()
    mindtct, bozorth3 = ensure_binaries(Path(args.install_dir))
    report["build_seconds"] = round(time.time() - t0, 1)
    report["mindtct_path"] = str(mindtct)
    report["bozorth3_path"] = str(bozorth3)
    report["on_path"] = bool(shutil.which("mindtct"))

    # Pick a genuine pair (probe + its true Real) and an impostor Real.
    gallery, *_ = ds.build_gallery(paths["real_dir"])
    probes, *_ = ds.build_probes(paths["level_dirs"][args.level], gallery)
    if not probes:
        raise SystemExit(f"No probes found for level {args.level}; run verify_dataset first.")
    probe = probes[0]
    genuine_real = gallery[probe.identity]
    impostor_real = next(r for idt, r in gallery.items() if idt != probe.identity)
    print(f"  probe   : {probe.filename} (identity {probe.identity_str})")
    print(f"  genuine : {genuine_real.filename}")
    print(f"  impostor: {impostor_real.filename}")

    work = Path(paths["cache_dir"]) / "nbis_spike"
    work.mkdir(parents=True, exist_ok=True)

    print("  extracting templates...")
    xyt_probe, fmt, n_probe = extract_template(Path(probe.path), work, mindtct)
    if xyt_probe is None:
        raise SystemExit("mindtct accepted NO tried format. Surface this before proceeding.")
    # Reuse the format that worked for the rest.
    xyt_gen, _, n_gen = extract_template(Path(genuine_real.path), work, mindtct, [fmt])
    xyt_imp, _, n_imp = extract_template(Path(impostor_real.path), work, mindtct, [fmt])

    genuine_score = match_pair(xyt_probe, xyt_gen, bozorth3)
    impostor_score = match_pair(xyt_probe, xyt_imp, bozorth3)

    report.update({
        "working_input_format": fmt,
        "minutiae": {"probe": n_probe, "genuine": n_gen, "impostor": n_imp},
        "genuine_score": genuine_score,
        "impostor_score": impostor_score,
        "sanity_genuine_gt_impostor": genuine_score > impostor_score,
    })

    print("\n== RESULT ==")
    print(f"  working input format : {fmt}")
    print(f"  minutiae (probe/gen/imp): {n_probe}/{n_gen}/{n_imp}")
    print(f"  genuine score  : {genuine_score}")
    print(f"  impostor score : {impostor_score}")
    print(f"  sanity (gen>imp): {genuine_score > impostor_score}")
    if min(n_probe, n_gen, n_imp) < cfg["models"].get("nbis", {}).get("min_minutiae_warn", 10):
        print("  WARNING: low minutiae count -- expected at SOCOFing's ~200 dpi resolution.")

    out = Path(paths["logs_dir"]) / "nbis_spike.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\n  report -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
