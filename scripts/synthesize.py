"""Synthesize all experiment summaries into report-ready tables and figures.

Reads every results/raw/<model>_<level>_<condition>/summary.json (+ cmc.npy) and
produces, under <results_dir>:
  summary.csv            one row per model x level x condition (overall metrics)
  per_alteration.csv     one row per model x level x condition x alteration
  figures/rank1_by_level.png         Rank-1 vs difficulty, per model (base/pre)
  figures/eer_by_level.png           EER vs difficulty, per model (base/pre)
  figures/preprocessing_delta.png    Rank-1 gain from preprocessing, per model
  figures/cmc_by_level.png           CMC curves per model, one panel per level
  figures/per_alteration_hard.png    Hard Rank-1 per model per alteration

It also prints a compact cross-model comparison table.

Usage:
    python scripts/synthesize.py [--config ...] [--glob 'results/raw/*']
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, resolve_paths  # noqa: E402

MODEL_ORDER = ["sift", "gabor", "dinov2"]
LEVEL_ORDER = ["Easy", "Medium", "Hard"]
COND_ORDER = ["baseline", "preprocessed"]
COLORS = {"sift": "#1f77b4", "gabor": "#2ca02c", "dinov2": "#d62728"}


def _first(d: dict, prefix: str):
    for k, v in d.items():
        if k.startswith(prefix):
            return k.replace(prefix, ""), v
    return None, None


def load_records(dirs):
    recs = {}
    for d in dirs:
        sj = Path(d) / "summary.json"
        if not sj.exists():
            continue
        s = json.loads(sj.read_text(encoding="utf-8"))
        idn, ver = s.get("identification", {}), s.get("verification", {})
        far_t, far_v = _first(ver, "far_at_frr_")
        frr_t, frr_v = _first(ver, "frr_at_far_")
        cmc_path = Path(d) / "cmc.npy"
        rec = {
            "model": s["model"], "level": s["level"], "condition": s["condition"],
            "tier": s.get("tier", "A"), "n_probes": s.get("n_probes"),
            "rank_1": idn.get("rank_1"), "rank_5": idn.get("rank_5"),
            "rank_10": idn.get("rank_10"), "mrr": idn.get("mrr"),
            "eer": ver.get("eer"), "auc": ver.get("auc"),
            "far_at_frr": far_v, "far_at_frr_target": far_t,
            "frr_at_far": frr_v, "frr_at_far_target": frr_t,
            "per_alteration": s.get("per_alteration", {}),
            "cmc": np.load(cmc_path) if cmc_path.exists() else None,
        }
        recs[(s["model"], s["level"], s["condition"], rec["tier"])] = rec
    return recs


def write_csvs(recs, out_dir):
    rows = sorted(recs.values(), key=lambda r: (
        r["tier"],
        MODEL_ORDER.index(r["model"]) if r["model"] in MODEL_ORDER else 9,
        LEVEL_ORDER.index(r["level"]) if r["level"] in LEVEL_ORDER else 9,
        r["condition"]))
    cols = ["model", "level", "condition", "tier", "n_probes", "rank_1", "rank_5",
            "rank_10", "mrr", "eer", "auc", "far_at_frr", "frr_at_far"]
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    with (out_dir / "per_alteration.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "level", "condition", "tier", "alt", "n",
                    "rank_1", "eer", "auc"])
        for r in rows:
            for alt, pa in r["per_alteration"].items():
                w.writerow([r["model"], r["level"], r["condition"], r["tier"], alt,
                            pa.get("n"), pa.get("rank_1"), pa.get("eer"), pa.get("auc")])
    print(f"  wrote summary.csv ({len(rows)} rows) + per_alteration.csv")


def _present(recs, key):
    return [m for m in (MODEL_ORDER if key == "model" else LEVEL_ORDER)
            if any(r[key] == m for r in recs.values())]


def _line_by_level(recs, metric, ylabel, title, out_path):
    models = _present(recs, "model")
    levels = _present(recs, "level")
    fig, ax = plt.subplots(figsize=(6, 4))
    for m in models:
        for cond, ls in (("baseline", "-"), ("preprocessed", "--")):
            ys = [recs.get((m, lv, cond), {}).get(metric) for lv in levels]
            if all(y is None for y in ys):
                continue
            ax.plot(levels, ys, ls, color=COLORS.get(m, "k"), marker="o",
                    label=f"{m} ({cond[:4]})")
    ax.set_xlabel("alteration level"); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)


def _delta_fig(recs, out_path):
    models = _present(recs, "model"); levels = _present(recs, "level")
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(levels)); w = 0.8 / max(len(models), 1)
    for i, m in enumerate(models):
        deltas = []
        for lv in levels:
            b = recs.get((m, lv, "baseline"), {}).get("rank_1")
            p = recs.get((m, lv, "preprocessed"), {}).get("rank_1")
            deltas.append((p - b) * 100 if (b is not None and p is not None) else 0)
        ax.bar(x + i * w, deltas, w, label=m, color=COLORS.get(m, "k"))
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x + w * (len(models) - 1) / 2); ax.set_xticklabels(levels)
    ax.set_ylabel("Rank-1 gain from preprocessing (pp)")
    ax.set_title("Preprocessing effect by model"); ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)


def _cmc_fig(recs, out_path, top=30):
    levels = _present(recs, "level"); models = _present(recs, "model")
    fig, axes = plt.subplots(1, len(levels), figsize=(4 * len(levels), 3.6),
                             squeeze=False)
    for j, lv in enumerate(levels):
        ax = axes[0][j]
        for m in models:
            rec = recs.get((m, lv, "baseline"))
            if rec is None or rec["cmc"] is None:
                continue
            cmc = rec["cmc"][:top]
            ax.plot(np.arange(1, len(cmc) + 1), cmc, color=COLORS.get(m, "k"),
                    marker="o", ms=3, label=m)
        ax.set_title(f"CMC — {lv} (baseline)"); ax.set_xlabel("rank")
        ax.set_ylim(0, 1.02); ax.grid(True, alpha=0.3)
        if j == 0:
            ax.set_ylabel("identification rate")
        ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)


def _per_alt_fig(recs, level, out_path):
    models = _present(recs, "model")
    alts = sorted({a for r in recs.values() if r["level"] == level
                   for a in r["per_alteration"]})
    if not alts:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(alts)); w = 0.8 / max(len(models), 1)
    for i, m in enumerate(models):
        rec = recs.get((m, level, "baseline"))
        ys = [rec["per_alteration"].get(a, {}).get("rank_1", 0) if rec else 0
              for a in alts]
        ax.bar(x + i * w, ys, w, label=m, color=COLORS.get(m, "k"))
    ax.set_xticks(x + w * (len(models) - 1) / 2); ax.set_xticklabels(alts)
    ax.set_ylabel("Rank-1"); ax.set_ylim(0, 1.02)
    ax.set_title(f"{level} Rank-1 by alteration (baseline)"); ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)


def print_table(recs):
    print("\n  model   level   cond           Rank-1   EER     AUC    FAR@FRR")
    print("  " + "-" * 62)
    for m in _present(recs, "model"):
        for lv in _present(recs, "level"):
            for c in COND_ORDER:
                r = recs.get((m, lv, c))
                if not r:
                    continue
                print(f"  {m:<7} {lv:<7} {c:<13} {r['rank_1']:.4f}  "
                      f"{r['eer']:.4f}  {r['auc']:.4f}  {r['far_at_frr']:.4f}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Synthesize summaries into tables + figures.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--results-dir", default=None,
                    help="read raw/ and write outputs here (e.g. a downloaded "
                         "results folder); skips the dataset/profile entirely")
    ap.add_argument("--glob", default=None)
    ap.add_argument("--input-root", default="/kaggle/input")
    args = ap.parse_args()

    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        paths = resolve_paths(load_config(args.config), input_root=args.input_root)
        results_dir = Path(paths["results_dir"])
    pattern = args.glob or str(results_dir / "raw" / "*")
    dirs = sorted(glob.glob(pattern))

    recs = load_records(dirs)
    if not recs:
        raise SystemExit(f"no summary.json found under {pattern}")
    tiers = sorted({k[3] for k in recs})
    print(f"== synthesize: {len(recs)} experiments (tiers: {', '.join(tiers)}) ==")

    fig_dir = results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    write_csvs(recs, results_dir)  # CSVs carry every tier (tier column)

    # Figures are the Tier-A cross-model comparison (all 3 models; SIFT is A-only).
    tierA = {(m, lv, c): r for (m, lv, c, t), r in recs.items() if t == "A"}
    if tierA:
        _line_by_level(tierA, "rank_1", "Rank-1", "Identification vs difficulty (Tier A)",
                       fig_dir / "rank1_by_level.png")
        _line_by_level(tierA, "eer", "EER", "Verification EER vs difficulty (Tier A)",
                       fig_dir / "eer_by_level.png")
        _delta_fig(tierA, fig_dir / "preprocessing_delta.png")
        _cmc_fig(tierA, fig_dir / "cmc_by_level.png")
        _per_alt_fig(tierA, "Hard", fig_dir / "per_alteration_hard.png")
        print(f"  wrote 5 Tier-A figures -> {fig_dir}")
        print_table(tierA)
    if "B" in tiers:
        print("\n  Tier B (full-probe best estimate) rows are in summary.csv "
              "(tier=B); figures stay Tier A for a fair 3-model comparison.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
