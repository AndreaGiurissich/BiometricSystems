"""Zip + timestamp the results/ directory so it survives Kaggle's ephemeral disk.

Kaggle wipes /kaggle/working between sessions, so after a run bundle everything
worth keeping -- CSVs, figures, per-experiment summary.json / scores.jsonl /
cmc.npy -- into a single timestamped archive you can download. A small MANIFEST
(file list + a config snapshot + library versions + git commit if available) is
embedded for reproducibility.

Usage:
    python scripts/save_results.py [--config ...] [--results-dir DIR]
                                   [--out-dir DIR] [--name NAME]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, resolve_paths  # noqa: E402


def _lib_versions() -> dict:
    vers = {}
    for mod in ("numpy", "cv2", "yaml", "matplotlib", "torch"):
        try:
            vers[mod] = __import__(mod).__version__
        except Exception:
            vers[mod] = None
    return vers


def _git_commit(repo_root: Path):
    try:
        out = subprocess.run(["git", "-C", str(repo_root), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Zip + timestamp results/ for download.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--results-dir", default=None, help="override results dir")
    ap.add_argument("--out-dir", default=None, help="where to write the zip "
                    "(default: parent of results dir)")
    ap.add_argument("--name", default="results", help="archive name prefix")
    ap.add_argument("--input-root", default="/kaggle/input")
    args = ap.parse_args()

    cfg = load_config(args.config)
    paths = resolve_paths(cfg, input_root=args.input_root)
    repo_root = Path(__file__).resolve().parents[1]
    results_dir = Path(args.results_dir) if args.results_dir else Path(paths["results_dir"])
    if not results_dir.exists():
        raise SystemExit(f"results dir not found: {results_dir} (nothing to save)")

    out_dir = Path(args.out_dir) if args.out_dir else results_dir.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_path = out_dir / f"{args.name}_{stamp}.zip"

    files = [p for p in results_dir.rglob("*") if p.is_file()]
    manifest = {
        "created": stamp,
        "results_dir": str(results_dir),
        "n_files": len(files),
        "git_commit": _git_commit(repo_root),
        "lib_versions": _lib_versions(),
        "files": [str(p.relative_to(results_dir)) for p in files],
    }

    print(f"== save_results ==\n  bundling {len(files)} files from {results_dir}")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in files:
            zf.write(p, arcname=str(Path("results") / p.relative_to(results_dir)))
        zf.writestr("results/MANIFEST.json", json.dumps(manifest, indent=2))
        zf.writestr("results/run_config.yaml", _dump_cfg(cfg))

    size_mb = zip_path.stat().st_size / 1e6
    print(f"  -> {zip_path}  ({size_mb:.1f} MB)")
    print("  download this from the Kaggle output panel before the session ends.")
    return 0


def _dump_cfg(cfg) -> str:
    import yaml
    return yaml.safe_dump(cfg, sort_keys=False)


if __name__ == "__main__":
    raise SystemExit(main())
