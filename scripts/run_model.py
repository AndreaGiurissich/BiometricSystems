"""Run any model end-to-end (identification + verification) on SOCOFing.

    python scripts/run_model.py --model gabor  --level Easy --condition both
    python scripts/run_model.py --model dinov2 --level Easy --condition both

Shared logic lives in src/pipeline.py; this is just the CLI + data loading.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, resolve_paths  # noqa: E402
from src import dataset as ds  # noqa: E402
from src import manifest as mf  # noqa: E402
from src import pipeline as pl  # noqa: E402


def run_cli(default_model: str | None = None) -> int:
    ap = argparse.ArgumentParser(description="End-to-end model runner.")
    ap.add_argument("--model", default=default_model, required=default_model is None,
                    choices=["gabor", "dinov2", "sift"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--level", default="Easy")
    ap.add_argument("--condition", default="both",
                    choices=["baseline", "preprocessed", "both"])
    ap.add_argument("--full", action="store_true",
                    help="Tier B: all probes (default: Tier-A seeded subsample)")
    ap.add_argument("--no-cap", action="store_true",
                    help="ignore the per-model probe cap (e.g. run SIFT on the "
                         "full 2000-probe manifest instead of its nested 500)")
    ap.add_argument("--limit", type=int, default=None, help="cap probes (smoke test)")
    ap.add_argument("--workers", type=int, default=None,
                    help="parallel workers for pairwise scoring (default: "
                         "runtime.num_workers; 1 = serial)")
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
    probes, *_ = ds.build_probes(paths["level_dirs"][args.level], gallery)
    if not probes:
        raise SystemExit(f"No probes for level {args.level}; run verify_dataset first.")
    probes = probes if args.full else mf.build_manifest(probes, cfg)
    # Per-model nested cap (e.g. SIFT's 500): a prefix of the balanced manifest.
    if not args.full and not args.no_cap:
        cap = cfg["subsampling"].get("per_model_n_probes", {}).get(args.model)
        if cap:
            probes = probes[:int(cap)]
    if args.limit:
        probes = probes[:args.limit]
    print(f"model={args.model}  gallery={len(ids)}  probes={len(probes)}  "
          f"({'full' if args.full else 'Tier-A subsample'})")

    conditions = (["baseline", "preprocessed"] if args.condition == "both"
                  else [args.condition])
    workers = args.workers if args.workers is not None else int(
        cfg.get("runtime", {}).get("num_workers", 1))
    summaries = pl.run(args.model, args.level, conditions, gallery, ids, probes,
                       cfg, paths, workers=workers, full=args.full)

    if len(summaries) == 2:
        b, p = summaries
        print("\n== baseline vs preprocessed ==")
        print(f"  Rank-1 : {b['identification']['rank_1']:.4f} -> "
              f"{p['identification']['rank_1']:.4f}")
        print(f"  EER    : {b['verification']['eer']:.4f} -> "
              f"{p['verification']['eer']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli())
