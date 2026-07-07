"""Metric tests on synthetic scores (no data, no model).

Locks down the identification ranking (incl. pessimistic tie handling) and the
verification metrics against analytically known cases: perfect separation ->
EER 0 / AUC 1; identical distributions -> EER ~0.5 / AUC ~0.5.

    python -m pytest tests/test_evaluation.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import evaluation as ev  # noqa: E402


# --- identification --------------------------------------------------------

def test_rank_of_match_basic():
    # true mate has the top score -> rank 1
    scores = [0.9, 0.5, 0.3]
    assert ev.rank_of_match(scores, [True, False, False]) == 1
    # true mate is third best -> rank 3
    assert ev.rank_of_match(scores, [False, False, True]) == 3


def test_rank_pessimistic_ties():
    # true mate tied with two higher-or-equal others -> worst-case rank 3
    scores = [0.8, 0.8, 0.8, 0.1]
    assert ev.rank_of_match(scores, [True, False, False, False]) == 3


def test_rank_distance_mode():
    # lower = better; true mate has the smallest distance -> rank 1
    scores = [0.1, 0.4, 0.9]
    assert ev.rank_of_match(scores, [True, False, False],
                            higher_is_better=False) == 1


def test_identification_metrics():
    ranks = [1, 1, 2, 6]            # 4 probes
    m = ev.identification_metrics(ranks, n_gallery=10, ranks_to_report=(1, 5, 10))
    assert m["rank_1"] == 0.5       # 2/4 at rank 1
    assert m["rank_5"] == 0.75      # 3/4 within rank 5
    assert m["rank_10"] == 1.0      # all within rank 10
    assert np.isclose(m["mrr"], np.mean([1, 1, 0.5, 1 / 6]))
    assert m["cmc"][-1] == 1.0      # CMC saturates at 1


# --- verification ----------------------------------------------------------

def test_perfect_separation():
    genuine = np.linspace(0.6, 1.0, 50)
    impostor = np.linspace(0.0, 0.4, 50)
    assert ev.eer(genuine, impostor) == 0.0
    assert np.isclose(ev.roc_auc(genuine, impostor), 1.0, atol=1e-6)
    assert ev.far_at_frr(genuine, impostor, 0.01) == 0.0


def test_identical_distributions():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 2000)
    y = rng.normal(0, 1, 2000)
    assert abs(ev.eer(x, y) - 0.5) < 0.05
    assert abs(ev.roc_auc(x, y) - 0.5) < 0.05


def test_distance_orientation_matches_similarity():
    """Same data, flipped sign + higher_is_better=False -> identical EER."""
    rng = np.random.default_rng(1)
    g = rng.normal(1.0, 1.0, 500)
    im = rng.normal(0.0, 1.0, 500)
    e_sim = ev.eer(g, im, higher_is_better=True)
    e_dist = ev.eer(-g, -im, higher_is_better=False)
    assert np.isclose(e_sim, e_dist, atol=1e-9)


def test_verification_metrics_keys():
    g = np.linspace(0.5, 1.0, 100)
    im = np.linspace(0.0, 0.5, 100)
    m = ev.verification_metrics(g, im, far_at_frr_targets=(0.01,),
                                frr_at_far_targets=(0.01,))
    assert {"eer", "auc", "far_at_frr_0.01", "frr_at_far_0.01"} <= set(m)
    assert 0.0 <= m["eer"] <= 1.0 and 0.0 <= m["auc"] <= 1.0
