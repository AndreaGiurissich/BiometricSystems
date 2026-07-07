"""Paired significance of the baseline-vs-preprocessed effect, per model x level.

Joins each model/level's baseline and preprocessed scores.jsonl by probe stem
(paired), then reports:
  identification -> McNemar on Rank-1 hits (does preprocessing change who is
                    identified at rank 1?)
  verification   -> paired bootstrap CI + p for Delta EER and Delta AUC.

Writes results/significance.csv and prints a table. This is the inferential layer
that turns point deltas into "significant on Medium/Hard, not on Easy".

Usage:
    python scripts/significance.py [--config ...] [--n-boot 2000] [--glob ...]
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, resolve_paths  # noqa: E402
from src import evaluation as ev  # noqa: E402
from src import stats as st  # noqa: E402

MODEL_ORDER = ["sift", "gabor", "dinov2"]
LEVEL_ORDER = ["Easy", "Medium", "Hard"]


def _load(jsonl: Path):
    out = {}
    if jsonl.exists():
        for line in jsonl.open(encoding="utf-8"):
            line = line.strip()
            if line:
                r = json.loads(line)
                out[r["stem"]] = r
    return out


def _stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def analyze_pair(base, prep, n_boot, seed, alt=None):
    stems = sorted(set(base) & set(prep))           # paired: present in both
    if alt is not None:                             # restrict to one alteration type
        stems = [s for s in stems if base[s]["alt"] == alt]
    if not stems:
        return None
    hb = np.array([base[s]["rank"] == 1 for s in stems])
    hp = np.array([prep[s]["rank"] == 1 for s in stems])
    b = int(np.sum(hb & ~hp))
    c = int(np.sum(~hb & hp))
    mc = st.mcnemar(b, c)

    base_v = [(base[s]["genuine"], base[s]["impostors"]) for s in stems]
    prep_v = [(prep[s]["genuine"], prep[s]["impostors"]) for s in stems]
    eer = st.paired_bootstrap(base_v, prep_v,
                              lambda g, im: ev.eer(g, im, True), n_boot, seed)
    auc = st.paired_bootstrap(base_v, prep_v,
                              lambda g, im: ev.roc_auc(g, im, True), n_boot, seed)
    return {
        "n_paired": len(stems),
        "rank1_base": float(hb.mean()), "rank1_prep": float(hp.mean()),
        "mcnemar_b": b, "mcnemar_c": c, "mcnemar_p": mc["pvalue"],
        "d_eer": eer["delta"], "d_eer_lo": eer["ci_low"], "d_eer_hi": eer["ci_high"],
        "d_eer_p": eer["pvalue"],
        "d_auc": auc["delta"], "d_auc_lo": auc["ci_low"], "d_auc_hi": auc["ci_high"],
        "d_auc_p": auc["pvalue"],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Paired significance tests.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--results-dir", default=None,
                    help="read raw/ and write significance.csv here (e.g. a "
                         "downloaded results folder); no dataset/profile needed")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--glob", default=None)
    ap.add_argument("--input-root", default="/kaggle/input")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg.get("subsampling", {}).get("seed", 1234))
    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        results_dir = Path(resolve_paths(cfg, input_root=args.input_root)["results_dir"])
    raw = Path(args.glob) if args.glob else results_dir / "raw"

    rows = []
    for tier, suffix in (("A", ""), ("B", "_full")):
        for model in MODEL_ORDER:
            for level in LEVEL_ORDER:
                base = _load(raw / f"{model}_{level}_baseline{suffix}" / "scores.jsonl")
                prep = _load(raw / f"{model}_{level}_preprocessed{suffix}" / "scores.jsonl")
                if not base or not prep:
                    continue
                alts = sorted({base[s]["alt"] for s in (set(base) & set(prep))})
                for alt in ["ALL"] + alts:           # overall + per alteration type
                    res = analyze_pair(base, prep, args.n_boot, seed,
                                       None if alt == "ALL" else alt)
                    if res is None:
                        continue
                    res.update({"model": model, "level": level, "tier": tier, "alt": alt})
                    rows.append(res)

    if not rows:
        raise SystemExit(f"no baseline/preprocessed pairs found under {raw}")

    results_dir.mkdir(parents=True, exist_ok=True)
    out = results_dir / "significance.csv"
    cols = ["model", "level", "tier", "alt", "n_paired", "rank1_base", "rank1_prep",
            "mcnemar_b", "mcnemar_c", "mcnemar_p",
            "d_eer", "d_eer_lo", "d_eer_hi", "d_eer_p",
            "d_auc", "d_auc_lo", "d_auc_hi", "d_auc_p"]
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in cols})

    print(f"== paired significance (n_boot={args.n_boot}) ==")
    print("  model  level  T  Rank-1 base->prep  McNemar p      dEER [95% CI] p          dAUC p")
    print("  " + "-" * 90)
    for r in rows:
        if r.get("alt") != "ALL":          # console = overview; per-alt is in the CSV
            continue
        print(f"  {r['model']:<6} {r['level']:<6} {r['tier']}  "
              f"{r['rank1_base']:.3f}->{r['rank1_prep']:.3f}  "
              f"p={r['mcnemar_p']:.3g} {_stars(r['mcnemar_p']):<3}  "
              f"dEER {r['d_eer']:+.4f} [{r['d_eer_lo']:+.4f},{r['d_eer_hi']:+.4f}] "
              f"{_stars(r['d_eer_p'])}  "
              f"dAUC {r['d_auc']:+.4f} {_stars(r['d_auc_p'])}")
    print(f"\n  -> {out}  (per-alteration rows included; console shows ALL only)")
    print("  (dEER negative = preprocessing lowers EER = better; * p<.05 ** .01 *** .001)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
