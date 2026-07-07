"""Subsampling + impostor-sampling tests (synthetic records, no real data).

Locks down the determinism, stratification balance, nested-prefix property, and
the per-probe impostor contract the verification protocol depends on.

    python -m pytest tests/test_manifest.py -v
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import FpRecord  # noqa: E402
from src import manifest as mf  # noqa: E402

ALTS = ("Obl", "CR", "Zcut")


def _rec(subject: int, finger: str, alt: str) -> FpRecord:
    name = f"{subject}__M_Left_{finger}_finger_{alt}.BMP"
    return FpRecord(path=name, filename=name, subject_id=subject, gender="M",
                    hand="Left", finger=finger, alt=alt, alt_raw=alt)


@pytest.fixture()
def probes():
    fingers = ("thumb", "index", "middle", "ring", "little")
    return [_rec(s, f, a) for s in range(1, 201) for f in fingers for a in ALTS]


@pytest.fixture()
def cfg():
    return {
        "subsampling": {"enabled": True, "n_probes": 300, "seed": 1234,
                        "stratify_by": "alteration_type"},
        "verification": {"n_impostors_per_probe": 100, "seed": 777},
    }


def test_manifest_size_and_determinism(probes, cfg):
    m1 = mf.build_manifest(probes, cfg)
    m2 = mf.build_manifest(probes, cfg)
    assert len(m1) == 300
    assert [r.filename for r in m1] == [r.filename for r in m2]  # deterministic


def test_manifest_stratified_balance(probes, cfg):
    m = mf.build_manifest(probes, cfg)
    counts = Counter(r.alt for r in m)
    assert set(counts) == set(ALTS)
    assert max(counts.values()) - min(counts.values()) <= 1  # equal balance


def test_manifest_nested_prefix_balanced(probes, cfg):
    """A nested prefix (e.g. SIFT's smaller set) stays stratified."""
    m = mf.build_manifest(probes, cfg)
    counts = Counter(r.alt for r in m[:60])
    assert max(counts.values()) - min(counts.values()) <= 1


def test_manifest_no_duplicates(probes, cfg):
    m = mf.build_manifest(probes, cfg)
    assert len({r.filename for r in m}) == len(m)


def test_impostors_contract(cfg):
    gallery_ids = [(s, "Left", f) for s in range(1, 50)
                   for f in ("thumb", "index", "middle")]
    probe_id = (1, "Left", "index")
    imp = mf.sample_impostor_ids(probe_id, gallery_ids, cfg)
    assert len(imp) == 100
    assert probe_id not in imp                      # never self
    assert len(set(imp)) == 100                     # distinct
    assert all(i in gallery_ids for i in imp)       # from the gallery


def test_impostors_deterministic_per_probe(cfg):
    gallery_ids = [(s, "Left", f) for s in range(1, 50)
                   for f in ("thumb", "index", "middle")]
    pid = (3, "Left", "thumb")
    a = mf.sample_impostor_ids(pid, gallery_ids, cfg)
    b = mf.sample_impostor_ids(pid, list(reversed(gallery_ids)), cfg)
    assert a == b  # independent of gallery iteration order
