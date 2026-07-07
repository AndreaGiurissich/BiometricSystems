"""Back-compat alias: Gabor end-to-end == run_model.py --model gabor.

The runner is now model-agnostic (src/pipeline.py); this thin wrapper keeps the
familiar `python scripts/run_gabor.py ...` entry point working. New work should
prefer `scripts/run_model.py --model gabor`.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_model import run_cli  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run_cli(default_model="gabor"))
