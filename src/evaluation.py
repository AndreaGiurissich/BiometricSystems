"""Metrics for closed-set identification and verification.

Pure functions over score arrays -- no data, no model, no I/O -- so they unit-test
anywhere. Scores are similarities (higher = more similar, e.g. Gabor cosine);
pass ``higher_is_better=False`` for distances.

Identification (rank-based, cross-model comparable):
    rank_of_match, cmc_curve, identification_metrics  -> Rank-1/5/10, CMC, MRR
Verification (per-model native score scale; never cross-normalized):
    roc_rates, eer, roc_auc, far_at_frr, frr_at_far, verification_metrics
"""
from __future__ import annotations

from typing import Dict, Iterable, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Identification (1:N, closed set)
# ---------------------------------------------------------------------------


def rank_of_match(scores: Sequence[float], is_true: Sequence[bool],
                  higher_is_better: bool = True) -> int:
    """1-based rank of the (single) true-mate gallery entry within ``scores``.

    Ties are resolved pessimistically: if k gallery entries share the true
    mate's score, the true mate is placed last among them (worst-case rank).
    """
    scores = np.asarray(scores, dtype=float)
    is_true = np.asarray(is_true, dtype=bool)
    if is_true.sum() != 1:
        raise ValueError("expected exactly one true-mate entry per probe")
    true_score = scores[is_true][0]
    if higher_is_better:
        better = np.sum(scores > true_score)
        ties = np.sum(scores == true_score) - 1  # exclude the true mate itself
    else:
        better = np.sum(scores < true_score)
        ties = np.sum(scores == true_score) - 1
    return int(better + ties + 1)


def cmc_curve(ranks: Sequence[int], n_gallery: int) -> np.ndarray:
    """Cumulative Match Characteristic: cmc[k-1] = fraction of probes with rank<=k."""
    ranks = np.asarray(ranks, dtype=int)
    counts = np.bincount(ranks, minlength=n_gallery + 1)[1:]  # ranks are 1-based
    return np.cumsum(counts) / len(ranks)


def identification_metrics(ranks: Sequence[int], n_gallery: int,
                           ranks_to_report: Iterable[int] = (1, 5, 10)
                           ) -> Dict[str, float]:
    """Rank-k accuracies, MRR, and the full CMC curve from per-probe ranks."""
    ranks = np.asarray(ranks, dtype=int)
    cmc = cmc_curve(ranks, n_gallery)
    out: Dict[str, float] = {}
    for k in ranks_to_report:
        out[f"rank_{k}"] = float(cmc[min(k, n_gallery) - 1])
    out["mrr"] = float(np.mean(1.0 / ranks))
    out["cmc"] = cmc
    return out


# ---------------------------------------------------------------------------
# Verification (genuine vs impostor score distributions)
# ---------------------------------------------------------------------------


def _oriented(genuine, impostor, higher_is_better: bool):
    """Return arrays oriented so that LARGER means more genuine."""
    g = np.asarray(genuine, dtype=float)
    im = np.asarray(impostor, dtype=float)
    if not higher_is_better:
        g, im = -g, -im
    return g, im


def roc_rates(genuine, impostor, higher_is_better: bool = True
              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Thresholds with FAR and FRR. Accept iff score >= threshold.

    FAR(t) = P(impostor accepted) = mean(impostor >= t)
    FRR(t) = P(genuine rejected)  = mean(genuine  <  t)
    """
    g, im = _oriented(genuine, impostor, higher_is_better)
    g = np.sort(g)   # np.sort copies -> the caller's arrays are never mutated
    im = np.sort(im)
    thr = np.unique(np.concatenate([g, im]))
    far = (len(im) - np.searchsorted(im, thr, side="left")) / len(im)
    frr = np.searchsorted(g, thr, side="left") / len(g)
    return thr, far, frr


def eer(genuine, impostor, higher_is_better: bool = True) -> float:
    """Equal Error Rate: the rate where FAR and FRR cross."""
    _, far, frr = roc_rates(genuine, impostor, higher_is_better)
    diff = far - frr
    idx = np.where(np.diff(np.sign(diff)) != 0)[0]
    if len(idx) == 0:
        i = int(np.argmin(np.abs(diff)))
        return float((far[i] + frr[i]) / 2)
    i = idx[0]
    # linear interpolation of the crossing between i and i+1
    d0, d1 = diff[i], diff[i + 1]
    alpha = 0.0 if d1 == d0 else d0 / (d0 - d1)
    far_c = far[i] + alpha * (far[i + 1] - far[i])
    frr_c = frr[i] + alpha * (frr[i + 1] - frr[i])
    return float((far_c + frr_c) / 2)


def roc_auc(genuine, impostor, higher_is_better: bool = True) -> float:
    """Area under the ROC, computed exactly as the Mann-Whitney statistic:
    AUC = P(genuine > impostor) + 0.5 * P(genuine == impostor).

    This avoids the trapezoid tie artifacts that arise when many scores collide
    at the same FAR (common with discrete/identical score grids)."""
    g, im = _oriented(genuine, impostor, higher_is_better)
    im_sorted = np.sort(im)
    wins = np.searchsorted(im_sorted, g, side="left")          # impostor < g
    ties = np.searchsorted(im_sorted, g, side="right") - wins  # impostor == g
    return float((wins.sum() + 0.5 * ties.sum()) / (len(g) * len(im)))


def far_at_frr(genuine, impostor, frr_target: float,
               higher_is_better: bool = True) -> float:
    """FAR at the operating point where FRR equals ``frr_target``.

    Convention: linear interpolation of the FAR-vs-FRR curve (np.interp). On a
    vertical ROC segment (several thresholds sharing one FRR but different FAR)
    the returned value sits between the segment's endpoints -- a fixed, documented
    choice rather than a worst-/best-case selection.
    """
    _, far, frr = roc_rates(genuine, impostor, higher_is_better)
    order = np.argsort(frr)  # frr increasing
    return float(np.interp(frr_target, frr[order], far[order]))


def frr_at_far(genuine, impostor, far_target: float,
               higher_is_better: bool = True) -> float:
    """FRR at the operating point where FAR equals ``far_target`` (same
    linear-interpolation convention as far_at_frr)."""
    _, far, frr = roc_rates(genuine, impostor, higher_is_better)
    order = np.argsort(far)  # far increasing
    return float(np.interp(far_target, far[order], frr[order]))


def verification_metrics(genuine, impostor, higher_is_better: bool = True,
                         far_at_frr_targets: Sequence[float] = (0.01,),
                         frr_at_far_targets: Sequence[float] = (0.01,)
                         ) -> Dict[str, float]:
    """EER, AUC, and FAR@FRR / FRR@FAR at the requested operating points."""
    out: Dict[str, float] = {
        "eer": eer(genuine, impostor, higher_is_better),
        "auc": roc_auc(genuine, impostor, higher_is_better),
    }
    for t in far_at_frr_targets:
        out[f"far_at_frr_{t}"] = far_at_frr(genuine, impostor, t, higher_is_better)
    for t in frr_at_far_targets:
        out[f"frr_at_far_{t}"] = frr_at_far(genuine, impostor, t, higher_is_better)
    return out
