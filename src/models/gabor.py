"""Gabor texture descriptor (frozen, inference-only).

A bank of Gabor filters (``orientations`` x ``frequencies``) is convolved with
the image; the mean absolute response is pooled over a fixed ``grid`` of cells,
giving a fixed-length descriptor regardless of image size. Two descriptors are
compared by **cosine similarity**.

All structural parameters come from ``cfg['models']['gabor']`` (config is the
single source of truth):
    orientations, frequencies (cycles/px), grid [rows, cols], response, l2_normalize

The kernel/envelope params Gabor also needs (the published spec is silent on
them) are pinned in the same config block -- ``sigma_factor``, ``gamma``,
``psi``, ``kernel_sigma_support``, ``canonical_size`` -- with standard defaults
applied if a key is absent:
  * ``sigma = sigma_factor * wavelength``   (sigma_factor=0.56 ~ 1-octave bandwidth)
  * each kernel is zero-meaned to remove the DC term so flat regions give ~0.
  * images are resized to ``canonical_size`` first, so a frequency in cycles/px
    means the same physical scale on every (variably-sized) SOCOFing image.

Preprocessing (baseline vs contrast-mask) is applied UPSTREAM by the pipeline;
``extract`` simply takes whatever image it is given.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

# Fallback defaults, used only if the config omits the key (config wins).
_SIGMA_FACTOR = 0.56
_GAMMA = 0.5
_PSI = 0.0
_KERNEL_SIGMA_SUPPORT = 3.0
_CANONICAL_SIZE = (96, 103)


def _to_gray_f32(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image.astype(np.float32)


def _grid_means(resp: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Mean of each cell of a rows x cols grid over a 2D response map."""
    out: List[float] = []
    for band in np.array_split(resp, rows, axis=0):
        for cell in np.array_split(band, cols, axis=1):
            out.append(float(cell.mean()))
    return np.asarray(out, dtype=np.float32)


class GaborModel:
    """Frozen Gabor texture descriptor + cosine scorer."""

    def __init__(self, cfg: Dict[str, Any]):
        g = cfg["models"]["gabor"]
        self.orientations: int = int(g["orientations"])
        self.frequencies: List[float] = [float(f) for f in g["frequencies"]]
        self.grid: Tuple[int, int] = (int(g["grid"][0]), int(g["grid"][1]))
        self.response: str = g.get("response", "mean_abs")
        self.l2_normalize: bool = bool(g.get("l2_normalize", True))
        self.sigma_factor: float = float(g.get("sigma_factor", _SIGMA_FACTOR))
        self.gamma: float = float(g.get("gamma", _GAMMA))
        self.psi: float = float(g.get("psi", _PSI))
        self.kernel_sigma_support: float = float(
            g.get("kernel_sigma_support", _KERNEL_SIGMA_SUPPORT))
        cs = g.get("canonical_size", _CANONICAL_SIZE)
        self.canonical_size: Tuple[int, int] = (int(cs[0]), int(cs[1]))
        if self.response != "mean_abs":
            raise ValueError(f"unsupported gabor response: {self.response!r}")
        self.kernels: List[np.ndarray] = self._build_bank()

    def _build_bank(self) -> List[np.ndarray]:
        kernels: List[np.ndarray] = []
        for o in range(self.orientations):
            theta = o * np.pi / self.orientations
            for freq in self.frequencies:
                wavelength = 1.0 / freq
                sigma = self.sigma_factor * wavelength
                ksize = int(2 * np.ceil(self.kernel_sigma_support * sigma) + 1)
                k = cv2.getGaborKernel((ksize, ksize), sigma, theta, wavelength,
                                       self.gamma, self.psi, ktype=cv2.CV_32F)
                k -= k.mean()  # zero-mean -> no DC response on flat regions
                kernels.append(k)
        return kernels

    @property
    def descriptor_length(self) -> int:
        return self.orientations * len(self.frequencies) * self.grid[0] * self.grid[1]

    def extract(self, image: np.ndarray) -> np.ndarray:
        """Return the L2-normalized Gabor descriptor for one image."""
        gray = _to_gray_f32(image)
        gray = cv2.resize(gray, self.canonical_size, interpolation=cv2.INTER_AREA)
        rows, cols = self.grid
        feats: List[np.ndarray] = []
        for kernel in self.kernels:
            resp = np.abs(cv2.filter2D(gray, cv2.CV_32F, kernel))
            feats.append(_grid_means(resp, rows, cols))
        vec = np.concatenate(feats).astype(np.float32)
        if self.l2_normalize:
            norm = float(np.linalg.norm(vec))
            if norm > 0:
                vec = vec / norm
        return vec

    @staticmethod
    def score(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two descriptors (higher = more similar)."""
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.dot(a, b) / denom) if denom > 0 else 0.0
