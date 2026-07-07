"""Paired significance tests (McNemar + bootstrap) on synthetic inputs.

    python -m pytest tests/test_stats.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import evaluation as ev  # noqa: E402
from src import stats as st  # noqa: E402


def test_mcnemar_symmetric_is_ns():
    assert st.mcnemar(10, 10)["pvalue"] == 1.0
    assert st.mcnemar(0, 0)["pvalue"] == 1.0


def test_mcnemar_exact_vs_chi2_selection():
    small = st.mcnemar(20, 2)
    assert small["method"] == "exact" and small["pvalue"] < 0.001
    big = st.mcnemar(60, 30)
    assert big["method"] == "chi2_cc" and big["pvalue"] < 0.01


def _probes(n, g_mean, i_mean, seed):
    rng = np.random.default_rng(seed)
    return [(float(rng.normal(g_mean, 0.12)), list(rng.normal(i_mean, 0.12, 100)))
            for _ in range(n)]


def test_bootstrap_detects_improvement():
    base = _probes(300, 0.60, 0.45, 0)
    prep = _probes(300, 0.72, 0.33, 1)   # better separation
    r = st.paired_bootstrap(base, prep, lambda g, im: ev.eer(g, im, True),
                            n_boot=400, seed=1)
    assert r["delta"] < 0 and r["ci_high"] < 0      # EER drops, CI excludes 0
    assert r["pvalue"] < 0.05


def test_bootstrap_identical_is_zero():
    base = _probes(200, 0.6, 0.4, 2)
    r = st.paired_bootstrap(base, base, lambda g, im: ev.eer(g, im, True),
                            n_boot=200, seed=3)
    assert abs(r["delta"]) < 1e-9
    assert r["ci_low"] <= 0 <= r["ci_high"]


def test_bootstrap_rejects_misaligned():
    base = _probes(10, 0.6, 0.4, 4)
    with pytest.raises(ValueError):
        st.paired_bootstrap(base, base[:5], lambda g, im: ev.eer(g, im, True))
