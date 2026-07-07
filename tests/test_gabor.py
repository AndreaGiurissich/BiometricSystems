"""Gabor descriptor tests on synthetic images.

No real data needed -- runs anywhere Python + pytest + cv2 exist. Locks down the
descriptor contract (length from config, L2 norm, determinism, size-invariance)
and the discriminative behaviour we rely on for matching (self-similarity = 1;
same-orientation textures score higher than orthogonal ones).

    python -m pytest tests/test_gabor.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config  # noqa: E402
from src.models.gabor import GaborModel  # noqa: E402


@pytest.fixture(scope="module")
def model():
    return GaborModel(load_config(None))


def _stripes(orientation: str, size=(103, 96), period=8) -> np.ndarray:
    """A binary stripe pattern, 'h' = horizontal ridges, 'v' = vertical."""
    h, w = size
    if orientation == "v":
        base = ((np.arange(w) // period) % 2) * 255
        img = np.tile(base, (h, 1))
    else:
        base = ((np.arange(h) // period) % 2) * 255
        img = np.tile(base[:, None], (1, w))
    return img.astype(np.uint8)


def test_descriptor_length_and_norm(model):
    d = model.extract(_stripes("v"))
    assert d.shape == (model.descriptor_length,)        # 8*4*8*8 = 2048
    assert d.dtype == np.float32
    assert np.isclose(np.linalg.norm(d), 1.0, atol=1e-5)


def test_deterministic(model):
    img = _stripes("h")
    assert np.array_equal(model.extract(img), model.extract(img))


def test_size_invariant_length(model):
    """Different image sizes -> same descriptor length (canonical resize)."""
    small = model.extract(_stripes("v", size=(103, 96)))
    big = model.extract(_stripes("v", size=(220, 180)))
    assert small.shape == big.shape == (model.descriptor_length,)


def test_self_similarity_is_one(model):
    d = model.extract(_stripes("v"))
    assert np.isclose(model.score(d, d), 1.0, atol=1e-5)


def test_same_orientation_beats_orthogonal(model):
    """Matching texture orientation should score higher than orthogonal."""
    v1 = model.extract(_stripes("v", period=8))
    v2 = model.extract(_stripes("v", period=10))   # same orientation, near scale
    h1 = model.extract(_stripes("h", period=8))
    assert model.score(v1, v2) > model.score(v1, h1)


def test_score_symmetric(model):
    a = model.extract(_stripes("v"))
    b = model.extract(_stripes("h"))
    assert np.isclose(model.score(a, b), model.score(b, a), atol=1e-6)
