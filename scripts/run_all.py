"""One-command full pipeline: run every model x level x condition, then
synthesize tables/figures and paired significance.

Profile-aware (Kaggle or local): set the dataset via --dataset-root or the
`local` profile (SOCOFING_PROFILE=local, dataset under data/socofing/SOCOFing/).
Models are reused across levels (DINOv2 weights load once); empty levels and
unbuildable models (e.g. DINOv2 without torch) are skipped with a warning.

Examples:
    # Kaggle, everything
    python scripts/run_all.py
    # local, no GPU -> skip DINOv2, fewer SIFT workers
    SOCOFING_PROFILE=local python scripts/run_all.py --models gabor,sift \
        --dataset-root data/socofing/SOCOFing --workers 4
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.config import load_config, resolve_paths  # noqa: E402
from src import dataset as ds  # noqa: E402
from src import manifest as mf  # noqa: E402
from src import pipeline as pl  # noqa: E402


def _probes_for(level, gallery, cfg, paths, model_name, full, limit):
    level_dir = paths["level_dirs"].get(level)
    if not level_dir or not Path(level_dir).exists():
        return None
    probes, *_ = ds.build_probes(level_dir, gallery)
    if not probes:
        return None
    if not full:
        probes = mf.build_manifest(probes, cfg)
        cap = cfg["subsampling"].get("per_model_n_probes", {}).get(model_name)
        if cap:
            probes = probes[:int(cap)]
    if limit:
        probes = probes[:limit]
    return probes


def main() -> int:
    ap = argparse.ArgumentParser(description="Full pipeline in one command.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--models", default="gabor,sift,dinov2")
    ap.add_argument("--levels", default="Easy,Medium,Hard")
    ap.add_argument("--condition", default="both",
                    choices=["baseline", "preprocessed", "both"])
    ap.add_argument("--full", action="store_true", help="Tier B: all probes")
    ap.add_argument("--limit", type=int, default=None, help="cap probes (smoke)")
    ap.add_argument("--workers", type=int, default=None, help="pairwise workers")
    ap.add_argument("--no-synthesize", action="store_true")
    ap.add_argument("--no-significance", action="store_true")
    ap.add_argument("--dataset-root", default=None)
    ap.add_argument("--input-root", default="/kaggle/input")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.dataset_root:
        active = cfg["paths"]["active_profile"]
        cfg["paths"]["profiles"][active]["dataset_root"] = args.dataset_root
    paths = resolve_paths(cfg, input_root=args.input_root)

    gallery, *_ = ds.build_gallery(paths["real_dir"])
    ids = sorted(gallery.keys())
    if not ids:
        raise SystemExit(f"empty gallery at {paths['real_dir']} -- check dataset/profile")
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    levels = [lv.strip() for lv in args.levels.split(",") if lv.strip()]
    conditions = (["baseline", "preprocessed"] if args.condition == "both"
                  else [args.condition])
    workers = args.workers if args.workers is not None else int(
        cfg.get("runtime", {}).get("num_workers", 1))
    print(f"== run_all == gallery={len(ids)} models={models} levels={levels} "
          f"conditions={conditions} workers={workers}")

    for name in models:
        try:
            model = pl.build_model(name, cfg)
        except Exception as exc:  # e.g. torch missing for dinov2
            print(f"\n!! skipping {name}: {type(exc).__name__}: {exc}")
            continue
        for level in levels:
            probes = _probes_for(level, gallery, cfg, paths, name, args.full, args.limit)
            if probes is None:
                print(f"  skip {name}/{level}: no probes for this level")
                continue
            for c in conditions:
                pl.run_condition(name, level, c, gallery, ids, probes, model,
                                 cfg, paths, workers, full=args.full)

    # downstream digests read from disk; run as subprocesses (env carries profile)
    common = [sys.executable]
    cfg_args = (["--config", args.config] if args.config else []) + \
               ["--input-root", args.input_root]
    if not args.no_synthesize:
        print("\n== synthesize ==")
        subprocess.call(common + [str(REPO / "scripts" / "synthesize.py")] + cfg_args)
    if not args.no_significance:
        print("\n== significance ==")
        subprocess.call(common + [str(REPO / "scripts" / "significance.py")] + cfg_args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
