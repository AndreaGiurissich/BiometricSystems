"""DINOv2 ViT-B/14 embedding model (frozen, inference-only).

The pretrained DINOv2 backbone is loaded via torch.hub (needs Internet enabled on
Kaggle), run in eval/no-grad mode -- no weights are updated. Each image is
resized to ``image_size``, the grayscale is replicated to 3 channels, ImageNet-
normalized, and passed through the net; the CLS embedding (768-D for ViT-B/14) is
L2-normalized and compared by cosine.

All params come from ``cfg['models']['dinov2']``. Preprocessing (baseline vs
contrast-mask) is applied to the grayscale image UPSTREAM, before this model.

Reproducibility note: inference is not bit-exact across GPUs/driver versions
(cuDNN kernels are not pinned to deterministic algorithms, and fp16 amplifies
small differences). For rank-based and EER metrics the effect is negligible, but
exact embeddings may differ on different hardware -- worth a footnote in reports.
"""
from __future__ import annotations

from typing import Dict, List

import cv2
import numpy as np

_EMBED_DIM = {"dinov2_vits14": 384, "dinov2_vitb14": 768,
              "dinov2_vitl14": 1024, "dinov2_vitg14": 1536}


class Dinov2Model:
    """Frozen DINOv2 CLS-embedding extractor + cosine scorer."""

    def __init__(self, cfg: Dict):
        import torch  # imported lazily so non-DINO runs don't need torch

        d = cfg["models"]["dinov2"]
        self.torch = torch
        self.image_size = int(d.get("image_size", 224))
        self.l2_normalize = bool(d.get("l2_normalize", True))
        self.batch_size = int(d.get("batch_size", 64))
        self.fp16 = bool(d.get("fp16", True))
        model_name = d.get("model", "dinov2_vitb14")
        self.descriptor_length = _EMBED_DIM.get(model_name, 768)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_half = self.fp16 and self.device == "cuda"
        self.net = torch.hub.load(d.get("hub_repo", "facebookresearch/dinov2"),
                                  model_name)
        self.net.eval().to(self.device)
        if self.use_half:
            self.net.half()
        for p in self.net.parameters():
            p.requires_grad_(False)

        mean = torch.tensor(d.get("norm_mean", [0.485, 0.456, 0.406])).view(1, 3, 1, 1)
        std = torch.tensor(d.get("norm_std", [0.229, 0.224, 0.225])).view(1, 3, 1, 1)
        self.mean = mean.to(self.device)
        self.std = std.to(self.device)

    def _to_tensor(self, image: np.ndarray):
        """Grayscale image -> normalized (1, 3, H, W) tensor on device."""
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(image, (self.image_size, self.image_size),
                         interpolation=cv2.INTER_CUBIC)
        x = self.torch.from_numpy(img.astype(np.float32) / 255.0)[None, None]
        x = x.repeat(1, 3, 1, 1).to(self.device)      # replicate gray -> 3 channels
        return (x - self.mean) / self.std

    def _forward(self, batch):
        if self.use_half:
            batch = batch.half()
        with self.torch.no_grad():
            feat = self.net(batch)                    # (B, D) CLS embedding
        return feat.float().cpu().numpy().astype(np.float32)

    def _normalize(self, vecs: np.ndarray) -> np.ndarray:
        if not self.l2_normalize:
            return vecs
        norms = np.linalg.norm(vecs, axis=-1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms

    def extract(self, image: np.ndarray) -> np.ndarray:
        vec = self._forward(self._to_tensor(image))[0]
        return self._normalize(vec[None, :])[0]

    def extract_batch(self, images: List[np.ndarray]) -> np.ndarray:
        """Optional batched path (faster on GPU) for many images at once."""
        out = []
        for i in range(0, len(images), self.batch_size):
            batch = self.torch.cat([self._to_tensor(im) for im in images[i:i + self.batch_size]])
            out.append(self._forward(batch))
        return self._normalize(np.vstack(out))

    @staticmethod
    def score(a: np.ndarray, b: np.ndarray) -> float:
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.dot(a, b) / denom) if denom > 0 else 0.0
