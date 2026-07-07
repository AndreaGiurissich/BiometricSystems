"""Digest run scores.jsonl files into a compact, pasteable report.

The per-probe detail (rank, genuine + impostor scores, alteration type) lives in
each experiment's scores.jsonl, which is too large to read by hand. This script
re-derives identification + verification metrics from those files -- overall and
broken down per alteration type (Obl/CR/Zcut) -- and runs a few sanity checks, so
the jsonl content can be reviewed without pasting thousands of lines.

Usage:
    python scripts/analyze_scores.py [--glob 'results/raw/gabor_*'] [--config ...]
    python scripts/analyze_scores.py results/raw/gabor_Easy_baseline ...
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, resolve_paths  # noqa: E402
from src import evaluation as ev  # noqa: E402


def _load(jsonl: Path):
    recs = []
    for line in jsonl.open(encoding="utf-8"):
        line = line.strip()
        if line:
            recs.append(json.loads(line))
    return recs


def _metrics(recs, ranks_to_report, far_targets):
    ranks = np.array([r["rank"] for r in recs])
    genuine = np.array([r["genuine"] for r in recs], dtype=float)
    impostor = np.array([s for r in recs for s in r["impostors"]], dtype=float)
    out = {"n": len(recs)}
    for k in ranks_to_report:
        out[f"rank_{k}"] = float(np.mean(ranks <= k))
    out["mrr"] = float(np.mean(1.0 / ranks))
    out["eer"] = ev.eer(genuine, impostor, higher_is_better=True)
    out["auc"] = ev.roc_auc(genuine, impostor, higher_is_better=True)
    for t in far_targets:
        out[f"far@frr{t}"] = ev.far_at_frr(genuine, impostor, t, higher_is_better=True)
    out["_genuine"] = genuine
    out["_impostor"] = impostor
    return out


def _sanity(m) -> str:
    g, im = m["_genuine"], m["_impostor"]
    # how often the genuine score is the best of {genuine, its impostors}: not
    # directly available here (impostors pooled), so report distribution overlap.
    return (f"genuine {g.mean():.3f}+/-{g.std():.3f}  "
            f"impostor {im.mean():.3f}+/-{im.std():.3f}  "
            f"sep(mean) {g.mean() - im.mean():+.3f}  "
            f"genuine<impostor.mean: {float(np.mean(g < im.mean())):.1%}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Digest scores.jsonl into per-alteration metrics.")
    ap.add_argument("dirs", nargs="*", help="experiment dirs (default: --glob)")
    ap.add_argument("--glob", default=None, help="glob for experiment dirs")
    ap.add_argument("--config", default=None)
    ap.add_argument("--input-root", default="/kaggle/input")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ec = cfg["evaluation"]
    ranks_to_report = ec.get("ranks_to_report", [1, 5, 10])
    far_targets = ec.get("far_at_frr", [0.01])

    dirs = list(args.dirs)
    if not dirs:
        pattern = args.glob
        if pattern is None:
            paths = resolve_paths(cfg, input_root=args.input_root)
            pattern = str(Path(paths["results_dir"]) / "raw" / "*")
        dirs = sorted(glob.glob(pattern))
    if not dirs:
        raise SystemExit("no experiment dirs found (pass dirs or --glob)")

    for d in dirs:
        jsonl = Path(d) / "scores.jsonl"
        if not jsonl.exists():
            print(f"skip (no scores.jsonl): {d}")
            continue
        recs = _load(jsonl)
        overall = _metrics(recs, ranks_to_report, far_targets)
        print(f"\n== {Path(d).name} ==  (n={overall['n']})")
        print(f"  overall  Rank-1 {overall['rank_1']:.4f}  "
              f"Rank-5 {overall.get('rank_5', float('nan')):.4f}  "
              f"MRR {overall['mrr']:.4f}  EER {overall['eer']:.4f}  "
              f"AUC {overall['auc']:.4f}  FAR@FRR={far_targets[0]} "
              f"{overall[f'far@frr{far_targets[0]}']:.4f}")
        print(f"  sanity   {_sanity(overall)}")
        alts = sorted({r["alt"] for r in recs})
        for a in alts:
            sub = [r for r in recs if r["alt"] == a]
            m = _metrics(sub, ranks_to_report, far_targets)
            print(f"  alt {a:<5} n={m['n']:<5} Rank-1 {m['rank_1']:.4f}  "
                  f"EER {m['eer']:.4f}  AUC {m['auc']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
