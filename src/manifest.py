"""Probe subsampling and impostor sampling -- deterministic and seeded.

Both the Tier-A probe manifest and the per-probe impostor sets are reproducible
from the config seeds alone, independent of iteration order, so a run can be
resumed or re-created exactly.

  build_manifest(probes, cfg)      -> stratified, seeded, nested probe subsample
  sample_impostor_ids(id, ids, cfg)-> N seeded non-self gallery identities/probe
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Dict, List, Sequence

import numpy as np

from src.dataset import FpRecord, Identity


def _stable_seed(base_seed: int, key: str) -> int:
    """A process-stable integer seed (Python's hash() is salted -> unusable)."""
    digest = hashlib.md5(f"{base_seed}:{key}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _id_key(identity: Identity) -> str:
    return f"{identity[0]}_{identity[1]}_{identity[2]}"


def _stratum(rec: FpRecord, key: str) -> str:
    if key == "alteration_type":
        return rec.alt or "Real"
    if key == "hand":
        return rec.hand
    if key == "finger":
        return rec.finger
    raise ValueError(f"unknown stratify_by: {key!r}")


def build_manifest(probes: Sequence[FpRecord], cfg: Dict) -> List[FpRecord]:
    """Stratified, seeded subsample of ``probes`` for Tier A.

    Stratified by ``subsampling.stratify_by`` with equal balance per stratum
    (falling back to availability when a stratum is short), then round-robin
    interleaved so that ANY prefix stays balanced -- this gives the nested
    property the smaller per-model sets (e.g. SIFT's 500) rely on.
    """
    sub = cfg["subsampling"]
    if not sub.get("enabled", True):
        return list(probes)
    n_total = int(sub["n_probes"])
    seed = int(sub["seed"])
    key = sub.get("stratify_by", "alteration_type")

    # Group, deterministic order within group, then a seeded shuffle.
    groups: Dict[str, List[FpRecord]] = defaultdict(list)
    for p in probes:
        groups[_stratum(p, key)].append(p)
    rng = np.random.default_rng(seed)
    shuffled: Dict[str, List[FpRecord]] = {}
    for k in sorted(groups):
        recs = sorted(groups[k], key=lambda r: r.filename)
        shuffled[k] = [recs[i] for i in rng.permutation(len(recs))]

    # Equal allocation per stratum, capped by availability, remainder filled
    # round-robin among strata that still have items.
    strata = sorted(shuffled)
    take = {k: 0 for k in strata}
    remaining = {k: len(shuffled[k]) for k in strata}
    target = min(n_total, sum(remaining.values()))
    while sum(take.values()) < target:
        progressed = False
        for k in strata:
            if sum(take.values()) >= target:
                break
            if take[k] < remaining[k]:
                take[k] += 1
                progressed = True
        if not progressed:
            break

    # Round-robin interleave for nested-balanced prefixes.
    out: List[FpRecord] = []
    pointers = {k: 0 for k in strata}
    while len(out) < target:
        for k in strata:
            if len(out) >= target:
                break
            if pointers[k] < take[k]:
                out.append(shuffled[k][pointers[k]])
                pointers[k] += 1
    return out


def sample_impostor_ids(probe_identity: Identity,
                        gallery_ids: Sequence[Identity],
                        cfg: Dict) -> List[Identity]:
    """Return N seeded non-self gallery identities for one probe.

    Per-probe deterministic: the seed mixes the global verification seed with the
    probe identity, so the impostor set is identical regardless of run order.
    """
    v = cfg["verification"]
    n = int(v["n_impostors_per_probe"])
    seed = int(v["seed"])
    # Sort the pool by a stable key so the sample does not depend on the order
    # the gallery identities were passed in (reproducibility).
    pool = sorted((g for g in gallery_ids if g != probe_identity), key=_id_key)
    if not pool:
        return []
    n = min(n, len(pool))
    rng = np.random.default_rng(_stable_seed(seed, _id_key(probe_identity)))
    idx = rng.choice(len(pool), size=n, replace=False)
    return [pool[i] for i in idx]
