"""Parser / gallery / probe tests on a synthetic SOCOFing-like tree.

These need no real data, so they run anywhere Python + pytest exist (including a
Kaggle cell). They lock down the filename grammar, identity join, closed-set
orphan detection, and the casing histogram before any model is built.

    python -m pytest tests/test_dataset.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import dataset as ds  # noqa: E402


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x00")  # content is irrelevant; only the name is parsed


@pytest.fixture()
def tree(tmp_path: Path) -> Path:
    """A miniature dataset: 2 identities x (1 real + altered variants)."""
    real = tmp_path / "Real"
    easy = tmp_path / "Altered" / "Altered-Easy"

    # Two complete finger-identities.
    _touch(real / "1__M_Left_index_finger.BMP")
    _touch(real / "1__M_Left_middle_finger.BMP")

    # Altered variants (note mixed casing on the suffix to exercise the histogram).
    _touch(easy / "1__M_Left_index_finger_Obl.BMP")
    _touch(easy / "1__M_Left_index_finger_CR.BMP")
    _touch(easy / "1__M_Left_index_finger_Zcut.BMP")
    _touch(easy / "1__M_Left_middle_finger_obl.BMP")   # lowercase token
    _touch(easy / "1__M_Left_middle_finger_ZCUT.BMP")  # uppercase token

    # An orphan: identity has no Real template -> must be flagged, not matched.
    _touch(easy / "999__F_Right_thumb_finger_CR.BMP")

    # Noise files that must be skipped, not parsed.
    _touch(real / "Thumbs.db")
    _touch(easy / "readme.txt")
    return tmp_path


def test_parse_real():
    rec = ds.parse_filename("1__M_Left_index_finger.BMP")
    assert rec is not None
    assert rec.subject_id == 1
    assert rec.gender == "M"
    assert rec.hand == "Left"
    assert rec.finger == "index"
    assert rec.alt is None
    assert rec.identity == (1, "Left", "index")


def test_parse_altered_canonicalizes_casing():
    rec = ds.parse_filename("12__F_Right_thumb_finger_zcut.BMP")
    assert rec is not None
    assert rec.alt == "Zcut"          # canonical
    assert rec.alt_raw == "zcut"      # as-seen
    assert rec.identity == (12, "Right", "thumb")


def test_parse_rejects_garbage():
    assert ds.parse_filename("Thumbs.db") is None
    assert ds.parse_filename("1_M_Left_index_finger.BMP") is None  # single underscore
    assert ds.parse_filename("1__X_Left_index_finger.BMP") is None  # bad gender


def test_gallery_build(tree: Path):
    gallery, skipped, collisions, unexpected = ds.build_gallery(tree / "Real")
    assert len(gallery) == 2
    assert (1, "Left", "index") in gallery
    assert collisions == []
    assert unexpected == []
    assert "Thumbs.db" in skipped


def test_gallery_collision(tmp_path: Path):
    _touch(tmp_path / "1__M_Left_index_finger.BMP")
    _touch(tmp_path / "1__F_Left_index_finger.BMP")  # same identity, diff gender
    gallery, _, collisions, _ = ds.build_gallery(tmp_path)
    assert len(gallery) == 1            # identity ignores gender
    assert len(collisions) == 1


def test_probes_and_orphans(tree: Path):
    gallery, *_ = ds.build_gallery(tree / "Real")
    probes, skipped, orphans, non_altered = ds.build_probes(
        tree / "Altered" / "Altered-Easy", gallery)
    assert len(probes) == 5            # 3 index + 2 middle
    assert len(orphans) == 1           # the 999 thumb has no Real
    assert "999__F_Right_thumb_finger_CR.BMP" in orphans
    assert "readme.txt" in skipped
    assert non_altered == []
    # every probe joins to an existing gallery identity
    assert all(p.identity in gallery for p in probes)


def test_casing_histogram(tree: Path):
    casing, unexpected, examined = ds.alt_casing_histogram(
        tree / "Altered" / "Altered-Easy", limit=100)
    assert examined == 6               # 6 altered files (readme.txt ignored)
    assert casing["Obl"] == 1
    assert casing["obl"] == 1
    assert casing["ZCUT"] == 1
    assert unexpected == {}            # all tokens are known alterations


def test_breakdown_counts(tree: Path):
    gallery, *_ = ds.build_gallery(tree / "Real")
    probes, *_ = ds.build_probes(tree / "Altered" / "Altered-Easy", gallery)
    counts = ds.breakdown_counts(probes)
    assert counts["alt"]["Obl"] == 2
    assert counts["alt"]["CR"] == 1
    assert counts["alt"]["Zcut"] == 2
    assert counts["finger"]["index"] == 3
    assert counts["finger"]["middle"] == 2
