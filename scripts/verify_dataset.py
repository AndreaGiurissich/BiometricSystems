"""On-disk dataset verification -- run this FIRST on Kaggle, before any model.

What it does:
  1. Walk the input root (default /kaggle/input) up to 2 levels deep and print
     the real structure, so we confirm the dataset path before trusting the
     configured dataset_root.
  2. Build the gallery from Real/ and assert (warn) it has 6000 finger-identities.
  3. For each altered level: count probes, per-(level, alteration) breakdown,
     detect orphans (probe identity missing from gallery -> breaks closed set),
     and print a casing histogram of alteration suffixes over the first N files.
  4. Write a JSON summary to <logs_dir>/dataset_verification.json.

Nothing here loads a model. Usage:
    python scripts/verify_dataset.py [--config configs/default.yaml]
                                     [--input-root /kaggle/input]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, resolve_paths  # noqa: E402
from src import dataset as ds  # noqa: E402


def _print_header(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify SOCOFing on-disk layout.")
    parser.add_argument("--config", default=None, help="Path to YAML config.")
    parser.add_argument("--input-root", default="/kaggle/input",
                        help="Root to walk for dataset discovery.")
    parser.add_argument("--dataset-root", default=None,
                        help="Override dataset_root (skips the configured path).")
    parser.add_argument("--max-depth", type=int, default=2)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.dataset_root:
        active = cfg["paths"]["active_profile"]
        cfg["paths"]["profiles"][active]["dataset_root"] = args.dataset_root
    paths = resolve_paths(cfg, input_root=args.input_root)
    summary: dict = {"profile": paths["profile"], "warnings": []}
    if paths.get("autodetected_root"):
        msg = (f"auto-detected dataset_root = {paths['autodetected_root']} "
               f"(configured path had no Real/); set this in configs/default.yaml")
        print(f"  NOTE: {msg}")
        summary["warnings"].append(msg)
        summary["autodetected_root"] = paths["autodetected_root"]

    # --- 1. Discover the actual structure under the input root ----------------
    _print_header(f"1. Structure under {args.input_root} (<= {args.max_depth} levels)")
    tree = ds.walk_summary(args.input_root, max_depth=args.max_depth)
    if not tree:
        print(f"  (nothing found at {args.input_root} -- is the dataset attached?)")
        summary["warnings"].append(f"input-root empty: {args.input_root}")
    for node in tree:
        indent = "  " * (node["depth"] + 1)
        print(f"{indent}{node['path']}  "
              f"[{node['n_subdirs']} dirs, {node['n_files']} files]  "
              f"e.g. {node['examples']}")
    summary["input_tree"] = tree

    # --- 2. Gallery from Real/ -----------------------------------------------
    _print_header(f"2. Gallery (Real) from {paths['real_dir']}")
    if not Path(paths["real_dir"]).exists():
        print(f"  ERROR: real_dir does not exist: {paths['real_dir']}")
        print("  Fix dataset_root in the config (see section 1) and re-run.")
        summary["warnings"].append(f"missing real_dir: {paths['real_dir']}")
        _dump(summary, paths)
        return 1

    gallery, skipped, collisions, unexpected_altered = ds.build_gallery(paths["real_dir"])
    expected = cfg["dataset"]["expected_real_count"]
    print(f"  identities (templates): {len(gallery)}  (expected {expected})")
    print(f"  unparsed filenames    : {len(skipped)}")
    print(f"  identity collisions   : {len(collisions)}")
    print(f"  reals with alt suffix : {len(unexpected_altered)}")
    if len(gallery) != expected:
        summary["warnings"].append(
            f"gallery size {len(gallery)} != expected {expected}")
    if skipped:
        print(f"    e.g. unparsed: {skipped[:5]}")
    if collisions:
        print(f"    e.g. collisions: {collisions[:3]}")
    summary["gallery"] = {
        "n_identities": len(gallery),
        "expected": expected,
        "n_skipped": len(skipped),
        "skipped_examples": skipped[:10],
        "n_collisions": len(collisions),
        "n_reals_with_alt": len(unexpected_altered),
    }

    # --- 3. Probes per level --------------------------------------------------
    summary["levels"] = {}
    hist_limit = cfg["dataset"]["casing_histogram_limit"]
    for level, level_dir in paths["level_dirs"].items():
        _print_header(f"3. Level '{level}' from {level_dir}")
        if not Path(level_dir).exists():
            print(f"  ERROR: level dir missing: {level_dir}")
            summary["warnings"].append(f"missing level dir: {level_dir}")
            continue

        probes, p_skipped, orphans, non_altered = ds.build_probes(level_dir, gallery)
        breakdown = ds.breakdown_counts(probes)
        casing, unexpected_tok, examined = ds.alt_casing_histogram(level_dir, hist_limit)

        print(f"  probes (valid)        : {len(probes)}")
        print(f"  per alteration        : {dict(breakdown['alt'])}")
        print(f"  per hand              : {dict(breakdown['hand'])}")
        print(f"  per finger            : {dict(breakdown['finger'])}")
        print(f"  unparsed              : {len(p_skipped)}  e.g. {p_skipped[:3]}")
        print(f"  orphans (no gallery)  : {len(orphans)}  e.g. {orphans[:3]}")
        print(f"  non-altered in folder : {len(non_altered)}  e.g. {non_altered[:3]}")
        print(f"  casing histogram over first {examined} altered files:")
        for tok, n in casing.most_common():
            flag = "  <-- UNEXPECTED" if tok.lower() not in ds._ALT_CANON else ""
            print(f"      {tok!r}: {n}{flag}")
        if orphans:
            summary["warnings"].append(f"{level}: {len(orphans)} orphan probes")
        if unexpected_tok:
            summary["warnings"].append(
                f"{level}: unexpected alt tokens {dict(unexpected_tok)}")

        summary["levels"][level] = {
            "n_probes": len(probes),
            "by_alteration": dict(breakdown["alt"]),
            "by_hand": dict(breakdown["hand"]),
            "by_finger": dict(breakdown["finger"]),
            "n_unparsed": len(p_skipped),
            "n_orphans": len(orphans),
            "n_non_altered": len(non_altered),
            "casing_histogram": dict(casing),
            "unexpected_tokens": dict(unexpected_tok),
            "casing_examined": examined,
        }

    # --- 4. Persist + verdict -------------------------------------------------
    _dump(summary, paths)
    _print_header("VERDICT")
    if summary["warnings"]:
        print("  Completed WITH warnings:")
        for w in summary["warnings"]:
            print(f"    - {w}")
    else:
        print("  All checks passed: parser and closed-set assumptions hold.")
    return 0


def _dump(summary: dict, paths: dict) -> None:
    out = Path(paths["logs_dir"]) / "dataset_verification.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\n  Summary written to {out}")


if __name__ == "__main__":
    raise SystemExit(main())
