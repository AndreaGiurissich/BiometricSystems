"""MinutiaeNet (deep) spike -- candidate REPLACEMENT for the NBIS minutiae model.

Why this exists
---------------
The NBIS spike (scripts/spike_nbis.py) proved the classical MINDTCT/BOZORTH3
pipeline is *plumbing-correct* but *degenerate* on SOCOFing: ~96x103 px images
yield only 4-6 minutiae and 0 match scores, because MINDTCT is calibrated for
~500 dpi full-finger captures. Any classical extractor hits the same resolution
wall. A pretrained *deep* minutiae extractor (frozen inference -- still satisfies
the no-training rule) is the one family that can plausibly recover usable
minutiae at this resolution without an explicit upscale step.

This spike mirrors spike_nbis.py so the two are directly comparable. It will:
  1. Ensure `fingerflow` is importable (pip-install if missing); log its version.
  2. Locate the pretrained weights (CoarseNet/FineNet/ClassifyNet/CoreNet +
     a VerifyNet matcher for the chosen precision). These are NOT pip-bundled --
     if absent the spike STOPS with download/attach instructions (no fake URLs,
     no invented numbers).
  3. Extract minutiae for the SAME probe / genuine-real / impostor-real triple
     spike_nbis used, logging counts -- the headline number to compare vs NBIS.
  4. Best-effort: run the VerifyNet matcher genuine vs impostor and check the
     sanity inequality. This stage degrades gracefully (the matcher feature
     layout has a core-distance column whose exact encoding must be confirmed on
     first run) -- a matcher failure still reports the all-important counts.
  5. Write <logs_dir>/minutiaenet_spike.json, shaped like nbis_spike.json.

This is a GATE / investigation step. It does NOT commit the NBIS->MinutiaeNet
protocol change: configs/default.yaml, CONSTRAINTS.md and README stay frozen until the
spike proves viability AND a supervisor signs off on the model-set change.

Usage:
    python scripts/spike_minutiaenet.py [--config configs/default.yaml]
                                        [--models-dir DIR] [--precision 10]
                                        [--level Easy]

`--models-dir` must contain the fingerflow weight files. On Kaggle the clean way
is to attach them as a dataset, e.g. /kaggle/input/minutiaenet-weights, and pass
that. Download links live in the fingerflow README:
    https://github.com/jakubarendac/fingerflow
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# fingerflow targets Keras 2 (it passes weights= to Conv2D). Modern Kaggle ships
# Keras 3, which rejects that API. Route tensorflow.keras to the Keras-2 shim
# (`tf-keras`) -- MUST be set before TensorFlow/fingerflow is imported.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, resolve_paths  # noqa: E402
from src import dataset as ds  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]

# VerifyNet ships per fixed precision = number of minutiae in the feature vector.
VALID_PRECISIONS = (10, 14, 20, 24, 30)

# Expected weight filenames. fingerflow distributes these via Google Drive (see
# the README) -- names can vary, so we resolve by glob/substring below rather
# than hardcoding, and surface whatever we actually find.
WEIGHT_KEYS = {
    "coarse_net": ["coarse", "CoarseNet"],
    "fine_net": ["fine", "FineNet"],
    "classify_net": ["classify", "ClassifyNet"],
    "core_net": ["core", "CoreNet"],
    # "verf" tolerates fingerflow's misspelled distributions (VerfifyNet/VerfiyNet).
    "verify_net": ["verify", "verf", "VerifyNet"],
}


def ensure_fingerflow() -> str:
    """Import fingerflow (pip-install on first run). Returns its version string.

    Also ensures the Keras-2 shim `tf-keras` is present so TF_USE_LEGACY_KERAS
    actually has a Keras 2 to route to (Kaggle ships only Keras 3 by default).
    """
    try:
        import tf_keras  # noqa: F401  (provides the legacy Keras 2 backend)
    except ImportError:
        print("  installing 'tf-keras' (Keras 2 shim for fingerflow)...")
        subprocess.call([sys.executable, "-m", "pip", "install", "-q", "tf-keras"])
    try:
        import fingerflow
    except ImportError:
        print("  installing 'fingerflow' (pulls TensorFlow -- can take minutes)...")
        subprocess.call([sys.executable, "-m", "pip", "install", "-q", "fingerflow"])
        import fingerflow  # may still raise -> surfaced to the caller
    return getattr(fingerflow, "__version__", "unknown")


def install_compat_shims() -> list:
    """Restore APIs removed from modern NumPy/SciPy that fingerflow's vendored
    MinutiaeNet code (circa 2018) + CoreNet still call. Surgical aliases -- NO
    downgrades, so the rest of the Kaggle stack is untouched. Must run before the
    Extractor is built (the model graph calls these at construction time).

      - scipy.signal.gaussian -> scipy.signal.windows.gaussian   (removed SciPy 1.13)
      - np.bool/int/float      -> Python builtins                (removed NumPy 1.24)
      - np.product/... etc.    -> NumPy 2.0 replacements          (removed NumPy 2.0)
    """
    import warnings
    applied: list = []
    import numpy as np
    import scipy.signal

    if not hasattr(scipy.signal, "gaussian"):
        from scipy.signal.windows import gaussian
        scipy.signal.gaussian = gaussian
        applied.append("scipy.signal.gaussian")

    # Bare scalar aliases removed in NumPy 1.24 (some re-added in 2.0). Probing
    # the name can emit a cosmetic FutureWarning -- silence it.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, builtin in (("bool", bool), ("int", int), ("float", float),
                              ("object", object), ("str", str)):
            if not hasattr(np, name):
                setattr(np, name, builtin)
                applied.append(f"np.{name}")

    # Functions/aliases removed in NumPy 2.0 -> their current equivalents.
    np2_aliases = {
        "product": "prod", "cumproduct": "cumprod", "sometrue": "any",
        "alltrue": "all", "round_": "round", "float_": "float64",
        "unicode_": "str_", "infty": "inf", "NaN": "nan", "NAN": "nan",
        "in1d": "isin", "trapz": "trapezoid",
    }
    for old, new in np2_aliases.items():
        if not hasattr(np, old) and hasattr(np, new):
            setattr(np, old, getattr(np, new))
            applied.append(f"np.{old}")

    # np.lib.pad was removed in NumPy 2.0 -> top-level np.pad. Used in the
    # extraction forward pass (get_maps_stft).
    if not hasattr(np.lib, "pad"):
        np.lib.pad = np.pad
        applied.append("np.lib.pad")

    # scikit-image >=0.19 dropped the `multichannel` kwarg (now `channel_axis`).
    # fingerflow's vendored smooth_dir_map calls filters.gaussian(multichannel=).
    # Wrap to translate the old kwarg; harmless on modern skimage.
    try:
        import skimage.filters as skf
        if not getattr(skf.gaussian, "_ff_patched", False):
            _orig_gaussian = skf.gaussian

            def _gaussian_compat(*a, **k):
                if "multichannel" in k:
                    mc = k.pop("multichannel")
                    k.setdefault("channel_axis", -1 if mc else None)
                return _orig_gaussian(*a, **k)

            _gaussian_compat._ff_patched = True
            skf.gaussian = _gaussian_compat
            applied.append("skimage.filters.gaussian(multichannel)")
    except ImportError:
        pass
    return applied


def resolve_weights(models_dir: Path, precision: int) -> dict:
    """Map each network to a concrete weight file under models_dir.

    Resolution is by case-insensitive substring match so we tolerate the exact
    filenames fingerflow ships. VerifyNet is additionally filtered by precision
    (e.g. a file mentioning '10') when several VerifyNet files coexist.
    Missing entries are reported as None -- the caller decides whether to stop.
    """
    files = [p for p in models_dir.rglob("*") if p.is_file()] if models_dir.exists() else []
    resolved: dict = {}
    for key, needles in WEIGHT_KEYS.items():
        cands = [p for p in files
                 if any(n.lower() in p.name.lower() for n in needles)]
        if key == "verify_net" and len(cands) > 1:
            pref = [p for p in cands if str(precision) in p.name]
            cands = pref or cands
        resolved[key] = str(cands[0]) if cands else None
    return resolved


def _missing_weights_msg(models_dir: Path, resolved: dict) -> str:
    missing = [k for k, v in resolved.items() if v is None]
    return (
        "Pretrained weights not found.\n"
        f"  models-dir : {models_dir}\n"
        f"  missing    : {missing}\n"
        "  fix: download the CoarseNet / FineNet / ClassifyNet / CoreNet and a\n"
        "       VerifyNet model from the fingerflow README\n"
        "       (https://github.com/jakubarendac/fingerflow), place them in\n"
        "       --models-dir (on Kaggle: attach them as a dataset), and re-run.\n"
        "  NOTE: not inventing weights or scores -- stopping so this is visible."
    )


def _to_feature_vectors(minutiae, core, precision: int):
    """Best-effort assembly of a (precision, cols) matrix for VerifyNet.

    fingerflow's matcher wants the minutiae columns plus a 'distance to core'
    column, padded/truncated to exactly `precision` rows (highest score first).
    The exact dtype encoding of the categorical 'class' column and the core
    schema must be confirmed against the installed package on first run -- this
    helper is intentionally defensive and is allowed to raise; the caller treats
    a raise as "matcher stage unavailable" and still reports minutiae counts.
    """
    import numpy as np
    import pandas as pd

    df = minutiae.copy()
    # core may be a DataFrame (x1,y1,x2,y2,...) or absent; fall back to centroid.
    if core is not None and len(core) > 0:
        c = core.iloc[0]
        cx = float(c.get("x1", c.get("x", df["x"].mean())))
        cy = float(c.get("y1", c.get("y", df["y"].mean())))
    else:
        cx, cy = float(df["x"].mean()), float(df["y"].mean())
    df["dist_core"] = np.hypot(df["x"] - cx, df["y"] - cy)
    df = df.sort_values("score", ascending=False).head(precision)
    # coerce the categorical class to a numeric code if needed
    if df["class"].dtype == object:
        df["class"] = pd.Categorical(df["class"]).codes
    feat = df[["x", "y", "angle", "score", "class", "dist_core"]].to_numpy(dtype="float32")
    if feat.shape[0] < precision:  # pad with zero rows
        pad = np.zeros((precision - feat.shape[0], feat.shape[1]), dtype="float32")
        feat = np.vstack([feat, pad])
    return feat


def main() -> int:
    ap = argparse.ArgumentParser(description="MinutiaeNet (fingerflow) spike.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--models-dir", default="/kaggle/input/minutiaenet-weights")
    ap.add_argument("--precision", type=int, default=10, choices=VALID_PRECISIONS,
                    help="VerifyNet precision = #minutiae; SOCOFing is low-res -> 10")
    ap.add_argument("--level", default="Easy")
    ap.add_argument("--upscale", default="1.0",
                    help="comma-separated cubic-upsample factors to SWEEP before "
                         "extraction (e.g. '1,2,4,6'). Tests whether ~500dpi-"
                         "calibrated extractors recover minutiae on SOCOFing's "
                         "~96px images. Models load once; each factor reuses them.")
    ap.add_argument("--dataset-root", default=None)
    ap.add_argument("--input-root", default="/kaggle/input")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.dataset_root:
        active = cfg["paths"]["active_profile"]
        cfg["paths"]["profiles"][active]["dataset_root"] = args.dataset_root
    paths = resolve_paths(cfg, input_root=args.input_root)
    models_dir = Path(args.models_dir)
    report: dict = {"level": args.level, "precision": args.precision,
                    "upscale": args.upscale, "models_dir": str(models_dir)}

    print("== MinutiaeNet spike (fingerflow) ==")
    t0 = time.time()
    version = ensure_fingerflow()
    report["fingerflow_version"] = version
    report["install_seconds"] = round(time.time() - t0, 1)
    print(f"  fingerflow {version}")

    shims = install_compat_shims()
    report["compat_shims"] = shims
    if shims:
        print(f"  compat shims applied: {', '.join(shims)}")

    weights = resolve_weights(models_dir, args.precision)
    report["weights"] = weights
    if any(v is None for v in weights.values()):
        msg = _missing_weights_msg(models_dir, weights)
        print("  " + msg.replace("\n", "\n  "))
        report["status"] = "missing_weights"
        _dump(report, paths)
        raise SystemExit(1)

    # Same probe / genuine / impostor triple as the NBIS spike, for comparability.
    gallery, *_ = ds.build_gallery(paths["real_dir"])
    probes, *_ = ds.build_probes(paths["level_dirs"][args.level], gallery)
    if not probes:
        raise SystemExit(f"No probes for level {args.level}; run verify_dataset first.")
    probe = probes[0]
    genuine_real = gallery[probe.identity]
    impostor_real = next(r for idt, r in gallery.items() if idt != probe.identity)
    print(f"  probe   : {probe.filename} (identity {probe.identity_str})")
    print(f"  genuine : {genuine_real.filename}")
    print(f"  impostor: {impostor_real.filename}")

    import cv2  # noqa: E402  (only needed once we know we'll run)
    from fingerflow.extractor import Extractor

    print("  loading Extractor (CoarseNet+FineNet+ClassifyNet+CoreNet)...")
    extractor = Extractor(weights["coarse_net"], weights["fine_net"],
                          weights["classify_net"], weights["core_net"])

    def extract(rec, scale):
        img = cv2.imread(rec.path)  # fingerflow expects a 3D (BGR) array
        if img is None:
            raise SystemExit(f"cv2 could not read {rec.path}")
        h0, w0 = img.shape[:2]
        if scale != 1.0:
            # NBIS and MinutiaeNet are calibrated for ~500 dpi full-finger
            # captures; SOCOFing is ~96x103. Upsample so ridge structure spans
            # the detectors' receptive fields. Cubic interpolation, same op for
            # gallery+probes. (Investigation -- not a committed preprocessing step.)
            img = cv2.resize(img, (int(w0 * scale), int(h0 * scale)),
                             interpolation=cv2.INTER_CUBIC)
        result = extractor.extract_minutiae(img)
        # extract_minutiae returns a dict: {'core': <df>, 'minutiae': <df>}.
        if isinstance(result, dict):
            minutiae = result.get("minutiae")
            core = result.get("core")
        else:  # fallbacks: attribute object, or a bare DataFrame
            minutiae = getattr(result, "minutiae", result)
            core = getattr(result, "core", None)
        n = int(len(minutiae)) if minutiae is not None else 0
        cols = list(getattr(minutiae, "columns", []))
        print(f"    [x{scale:g}] {rec.filename}: {n} minutiae "
              f"[{w0}x{h0} -> {img.shape[1]}x{img.shape[0]}]  cols={cols}")
        return minutiae, core, n

    factors = [float(t) for t in str(args.upscale).split(",") if t.strip()] or [1.0]
    warn = cfg["models"].get("nbis", {}).get("min_minutiae_warn", 10)

    print(f"  extracting minutiae (sweep over upscale factors {factors})...")
    sweep: dict = {}
    best = None
    for scale in factors:
        mp, cp, n_p = extract(probe, scale)
        mg, cg, n_g = extract(genuine_real, scale)
        mi, ci, n_i = extract(impostor_real, scale)
        key = f"x{scale:g}"
        sweep[key] = {"probe": n_p, "genuine": n_g, "impostor": n_i}
        if best is None or min(n_p, n_g, n_i) > best["min"]:
            best = {"scale": scale, "min": min(n_p, n_g, n_i),
                    "vecs": ((mp, cp), (mg, cg), (mi, ci))}
    report["sweep"] = sweep
    report["minutiae"] = sweep[f"x{best['scale']:g}"]  # best factor = headline

    # Best-effort matcher on the best factor (counts already decide viability).
    genuine_score = impostor_score = None
    matcher_error = None
    if best["min"] >= 3:
        try:
            from fingerflow.matcher import Matcher
            matcher = Matcher(args.precision, weights["verify_net"])
            (mp, cp), (mg, cg), (mi, ci) = best["vecs"]
            fp = _to_feature_vectors(mp, cp, args.precision)
            fg = _to_feature_vectors(mg, cg, args.precision)
            fi = _to_feature_vectors(mi, ci, args.precision)
            genuine_score = float(matcher.verify(fp, fg))
            impostor_score = float(matcher.verify(fp, fi))
        except Exception as exc:  # pragma: no cover - first-run API confirmation
            matcher_error = f"{type(exc).__name__}: {exc}"
            print(f"  matcher stage unavailable ({matcher_error}); counts still valid.")
    else:
        matcher_error = "skipped (best factor has too few minutiae)"

    sanity = (genuine_score is not None and impostor_score is not None
              and genuine_score > impostor_score)
    report.update({
        "genuine_score": genuine_score,
        "impostor_score": impostor_score,
        "matcher_error": matcher_error,
        "sanity_genuine_gt_impostor": sanity,
    })

    print("\n== RESULT ==")
    print("  upscale sweep -- minutiae (probe/gen/imp):")
    for key, v in sweep.items():
        usable = "   <-- usable" if min(v.values()) >= warn else ""
        print(f"    {key:>5}: {v['probe']}/{v['genuine']}/{v['impostor']}{usable}")
    print(f"  baseline: NBIS native 6/5/4, MinutiaeNet native 2/2/2")
    print(f"  genuine score  : {genuine_score}")
    print(f"  impostor score : {impostor_score}")
    print(f"  sanity (gen>imp): {sanity}")
    if best["min"] < warn:
        print(f"  VERDICT: best factor x{best['scale']:g} still below {warn} "
              "minutiae -- minutiae matching is NON-VIABLE on SOCOFing.")
    else:
        print(f"  VERDICT: x{best['scale']:g} recovers usable minutiae -- viable "
              "WITH an upsampling step (needs sign-off; apply to gallery+probes).")

    _dump(report, paths)
    return 0


def _dump(report: dict, paths: dict) -> None:
    out = Path(paths["logs_dir"]) / "minutiaenet_spike.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\n  report -> {out}")


if __name__ == "__main__":
    raise SystemExit(main())
