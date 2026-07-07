"""Paired significance tests for the baseline-vs-preprocessed comparison.

Both conditions are evaluated on the SAME probes, so the comparison is paired and
the right tests are paired ones:

  mcnemar(b, c)                 -> identification: did Rank-1 hits change? (exact
                                   binomial for few discordant pairs, else the
                                   continuity-corrected chi-square)
  paired_bootstrap(base, prep)  -> verification: CI + p for the DELTA of a metric
                                   (EER, AUC) by resampling PROBES (not scores),
                                   rebuilding genuine+impostor for both conditions
                                   from the same resampled probes.

Pure numpy / stdlib -- no statsmodels/scipy dependency.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np


def mcnemar(b: int, c: int) -> Dict:
    """McNemar test on discordant counts.

    b = #probes the baseline got right and preprocessing got wrong
    c = #probes preprocessing got right and baseline got wrong
    (concordant pairs carry no information). Two-sided.
    """
    n = int(b + c)
    if n == 0:
        return {"b": b, "c": c, "n_discordant": 0, "pvalue": 1.0, "method": "none"}
    if n < 25:  # exact binomial (standard guidance for few discordant pairs)
        k = min(b, c)
        tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
        return {"b": b, "c": c, "n_discordant": n,
                "pvalue": min(1.0, 2.0 * tail), "method": "exact"}
    stat = (abs(b - c) - 1) ** 2 / n              # continuity-corrected chi-square
    pvalue = math.erfc(math.sqrt(stat / 2.0))      # chi2 df=1 survival
    return {"b": b, "c": c, "n_discordant": n, "statistic": stat,
            "pvalue": pvalue, "method": "chi2_cc"}


def _impostor_matrix(per_probe: Sequence[Sequence[float]]):
    """Stack impostor scores into a (n, k) array when every probe has the same
    count (the common case: 100/probe); else return None (use the slow path)."""
    counts = {len(x) for x in per_probe}
    if len(counts) == 1:
        return np.asarray(per_probe, dtype=float)
    return None


def paired_bootstrap(base: List[Tuple[float, Sequence[float]]],
                     prep: List[Tuple[float, Sequence[float]]],
                     metric: Callable[[np.ndarray, np.ndarray], float],
                     n_boot: int = 2000, seed: int = 1234, ci: float = 0.95) -> Dict:
    """CI + p for metric(prep) - metric(base) via paired probe resampling.

    base / prep are aligned per-probe lists of (genuine_score, [impostor_scores]).
    Each bootstrap draws probe indices with replacement and rebuilds both
    conditions from the SAME indices, so the pairing is preserved.
    metric(genuine, impostor) -> float (e.g. evaluation.eer / roc_auc).
    """
    n = len(base)
    if n != len(prep) or n == 0:
        raise ValueError("base and prep must be aligned, non-empty")
    bg = np.array([g for g, _ in base], dtype=float)
    pg = np.array([g for g, _ in prep], dtype=float)
    bi = _impostor_matrix([im for _, im in base])
    pi = _impostor_matrix([im for _, im in prep])
    bi_list = [np.asarray(im, dtype=float) for _, im in base]
    pi_list = [np.asarray(im, dtype=float) for _, im in prep]

    def imp(idx, mat, lst):
        return mat[idx].ravel() if mat is not None else np.concatenate([lst[j] for j in idx])

    full = np.arange(n)
    point = (metric(pg, imp(full, pi, pi_list))
             - metric(bg, imp(full, bi, bi_list)))

    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot)
    for t in range(n_boot):
        idx = rng.integers(0, n, n)
        deltas[t] = (metric(pg[idx], imp(idx, pi, pi_list))
                     - metric(bg[idx], imp(idx, bi, bi_list)))
    lo = float(np.percentile(deltas, 100 * (1 - ci) / 2))
    hi = float(np.percentile(deltas, 100 * (1 + ci) / 2))
    # two-sided bootstrap p: twice the smaller tail mass around 0
    p = 2.0 * min(float(np.mean(deltas >= 0)), float(np.mean(deltas <= 0)))
    return {"delta": float(point), "ci_low": lo, "ci_high": hi,
            "pvalue": min(1.0, p), "n_boot": n_boot, "ci": ci}
