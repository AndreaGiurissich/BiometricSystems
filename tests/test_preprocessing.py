"""Contrast-mask preprocessing tests on synthetic images.

No real data needed -- runs anywhere Python + pytest + cv2 exist (incl. a Kaggle
cell). Locks down the operator's contract (dtype/shape/range, determinism, the
flat-image identity, soft-mask range) and that it reads the SHIPPED config params,
before the operator is wired into any model pipeline.

    python -m pytest tests/test_preprocessing.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config  # noqa: E402
from src.preprocessing import Stages, contrast_mask  # noqa: E402


@pytest.fixture(scope="module")
def cfg():
    """The real shipped config -- also asserts its preprocessing keys exist."""
    return load_config(None)


@pytest.fixture()
def fp_like():
    """A small ridge-like synthetic image (alternating stripes + noise)."""
    rng = np.random.default_rng(0)
    h, w = 103, 96  # SOCOFing-ish
    stripes = (np.sin(np.arange(w) * 0.6) > 0).astype(np.uint8) * 180 + 20
    img = np.tile(stripes, (h, 1)).astype(np.float32)
    img += rng.normal(0, 8, size=(h, w))  # sensor noise
    return np.clip(img, 0, 255).astype(np.uint8)


def test_output_dtype_shape_range(cfg, fp_like):
    out = contrast_mask(fp_like, cfg)
    assert out.dtype == np.uint8
    assert out.shape == fp_like.shape
    assert out.min() >= 0 and out.max() <= 255


def test_accepts_color_input(cfg, fp_like):
    color = np.repeat(fp_like[:, :, None], 3, axis=2)  # fake BGR
    out = contrast_mask(color, cfg)
    assert out.shape == fp_like.shape  # collapsed to single channel
    assert out.dtype == np.uint8


def test_flat_image_is_unchanged(cfg):
    """No contrast -> mask is 0 everywhere -> blend returns the original."""
    flat = np.full((40, 40), 128, dtype=np.uint8)
    out = contrast_mask(flat, cfg)
    assert np.array_equal(out, flat)


def test_deterministic(cfg, fp_like):
    assert np.array_equal(contrast_mask(fp_like, cfg), contrast_mask(fp_like, cfg))


def test_stages_contract(cfg, fp_like):
    final, stages = contrast_mask(fp_like, cfg, return_stages=True)
    assert isinstance(stages, Stages)
    assert np.array_equal(final, stages.final)
    # soft mask is a float probability map; everything else is uint8 imagery.
    assert stages.soft_mask.dtype == np.float32
    assert stages.soft_mask.min() >= 0.0 and stages.soft_mask.max() <= 1.0
    for arr in (stages.denoised, stages.laplacian, stages.mask_binary,
                stages.sharpened):
        assert arr.dtype == np.uint8
        assert arr.shape == fp_like.shape
    # binary mask really is binary.
    assert set(np.unique(stages.mask_binary)).issubset({0, 255})


def test_sharpens_where_mask_is_active(cfg, fp_like):
    """On a textured image the operator should change pixels (not a no-op)."""
    out = contrast_mask(fp_like, cfg)
    assert not np.array_equal(out, fp_like)
