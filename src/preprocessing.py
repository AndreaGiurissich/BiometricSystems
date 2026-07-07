"""Contrast-mask preprocessing (Moresca/Suozzi/Santini, face-detection paper).

NO facial alignment. Paper defaults are applied VERBATIM -- every parameter is
read from ``configs/default.yaml`` (the ``preprocessing`` block), nothing is
hardcoded here. The operator sharpens only high-contrast regions (fingerprint
ridges) through a soft mask, leaving flat/background areas untouched so noise is
not amplified.

Two tracks that merge:

    Mask track : denoise -> Laplacian -> blur -> threshold -> blur  => soft mask in [0,1]
    Sharp track: unsharp mask on the ORIGINAL                       => sharpened
    Blend      : final = mask * sharpened + (1 - mask) * original

The Gaussian denoise (step 1) feeds ONLY the mask computation; it never appears
in the output directly.

Two places where the published recipe / config is silent and a faithful default
was chosen (surface these at montage review, change only on sign-off):
  * Laplacian: computed in float32 and rectified (|.|) so both rising and falling
    ridge edges contribute to the contrast map (cv2's default CV_8U would clip
    the negative lobe and drop half the edges).
  * Unsharp: standard unsharp masking, final = (1+amount)*orig - amount*blur.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class Stages:
    """Intermediate stages of the pipeline, for montages / debugging.

    All arrays are uint8 except ``soft_mask`` (float32 in [0, 1]).
    """
    denoised: np.ndarray
    laplacian: np.ndarray
    mask_binary: np.ndarray
    soft_mask: np.ndarray
    sharpened: np.ndarray
    final: np.ndarray


def _to_gray_u8(image: np.ndarray) -> np.ndarray:
    """Coerce any input to a single-channel uint8 image."""
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def _ksize(value) -> Tuple[int, int]:
    """YAML stores kernel sizes as [w, h] lists; cv2 wants an (w, h) tuple."""
    return (int(value[0]), int(value[1]))


def contrast_mask(image: np.ndarray, cfg: Dict[str, Any],
                  return_stages: bool = False):
    """Apply the contrast-mask pipeline to one image.

    Parameters
    ----------
    image : np.ndarray
        Grayscale or BGR image (uint8 or convertible). SOCOFing is grayscale.
    cfg : dict
        The full config; only ``cfg['preprocessing']`` is read.
    return_stages : bool
        If True, also return a :class:`Stages` with every intermediate.

    Returns
    -------
    np.ndarray (uint8)  -- the processed image, or ``(final, Stages)``.
    """
    p = cfg["preprocessing"]
    orig = _to_gray_u8(image)
    orig_f = orig.astype(np.float32)

    # --- Mask track -------------------------------------------------------
    # Step 1: Gaussian denoise (mask computation only).
    gd = p["gaussian_denoise"]
    denoised = cv2.GaussianBlur(orig, _ksize(gd["ksize"]), gd["sigma"])

    # Step 2: Laplacian -> contrast/edge magnitude map.
    lap = cv2.Laplacian(denoised, cv2.CV_32F, ksize=int(p["laplacian"]["ksize"]))
    lap_u8 = np.clip(np.abs(lap), 0, 255).astype(np.uint8)

    # Step 3a: blur the contrast map so thin edge responses fuse into regions.
    mb1 = p["mask_blur1"]
    blurred_lap = cv2.GaussianBlur(lap_u8, _ksize(mb1["ksize"]), mb1["sigma"])

    # Step 3b: threshold -> binary mask (where to sharpen / where not to).
    mt = p["mask_threshold"]
    _, mask_bin = cv2.threshold(blurred_lap, mt["value"], mt["maxval"],
                                cv2.THRESH_BINARY)

    # Step 3c: blur the binary mask -> soft mask in [0, 1] (smooth transition).
    mb2 = p["mask_blur2"]
    soft = cv2.GaussianBlur(mask_bin, _ksize(mb2["ksize"]), mb2["sigma"])
    soft = soft.astype(np.float32) / 255.0

    # --- Sharp track ------------------------------------------------------
    # Step 4: unsharp mask on the ORIGINAL.
    us = p["unsharp"]
    amount = float(us["amount"])
    us_blur = cv2.GaussianBlur(orig, _ksize(us["ksize"]), us["sigma"]).astype(np.float32)
    sharpened_f = (1.0 + amount) * orig_f - amount * us_blur

    # --- Blend ------------------------------------------------------------
    # Step 5: final = mask * sharpened + (1 - mask) * original.
    final_f = soft * sharpened_f + (1.0 - soft) * orig_f
    final = np.clip(final_f, 0, 255).astype(np.uint8)

    if not return_stages:
        return final
    return final, Stages(
        denoised=denoised,
        laplacian=lap_u8,
        mask_binary=mask_bin,
        soft_mask=soft,
        sharpened=np.clip(sharpened_f, 0, 255).astype(np.uint8),
        final=final,
    )
