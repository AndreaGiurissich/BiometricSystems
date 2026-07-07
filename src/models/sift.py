"""SIFT + RANSAC matcher (frozen, inference-only).

Unlike Gabor/DINOv2 (fixed-length descriptor + cosine), SIFT scores a PAIR of
images: detect keypoints/descriptors on each, match with a Lowe ratio test, fit
a RANSAC homography, and use the inlier count as the similarity score. This makes
it O(N x gallery) -- hence Tier A runs it on a nested probe subset.

Interface for the pipeline:
    score_type = "pairwise"
    .extract(image) -> {'pts': (K,2) float32, 'desc': (K,128) float32}
    .score(a, b)    -> float inlier count (0 if < min_matches good matches)

All params come from ``cfg['models']['sift']``. RANSAC is seeded via
cv2.setRNGSeed for reproducibility.
"""
from __future__ import annotations

from typing import Dict

import cv2
import numpy as np


class SiftModel:
    score_type = "pairwise"

    def __init__(self, cfg: Dict):
        s = cfg["models"]["sift"]
        self.sift = cv2.SIFT_create(
            nfeatures=int(s.get("nfeatures", 400)),
            contrastThreshold=float(s.get("contrast_threshold", 0.04)),
            edgeThreshold=float(s.get("edge_threshold", 10)),
            sigma=float(s.get("sigma", 1.6)),
        )
        self.lowe_ratio = float(s.get("lowe_ratio", 0.75))
        self.min_matches = int(s.get("min_matches", 4))
        self.score_kind = s.get("score", "inliers")
        ransac = s.get("ransac", {})
        self.reproj_thresh = float(ransac.get("reproj_thresh", 5.0))
        self.max_iters = int(ransac.get("max_iters", 2000))
        cv2.setRNGSeed(int(ransac.get("seed", 42)))  # reproducible RANSAC

        if s.get("matcher", "flann") == "flann":
            self.matcher = cv2.FlannBasedMatcher(
                dict(algorithm=1, trees=5), dict(checks=50))  # KDTREE for SIFT floats
        else:
            self.matcher = cv2.BFMatcher(cv2.NORM_L2)

    def extract(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kp, desc = self.sift.detectAndCompute(image, None)
        pts = (np.array([k.pt for k in kp], dtype=np.float32)
               if kp else np.zeros((0, 2), dtype=np.float32))
        desc = desc if desc is not None else np.zeros((0, 128), dtype=np.float32)
        return {"pts": pts, "desc": desc.astype(np.float32)}

    def score(self, a: Dict[str, np.ndarray], b: Dict[str, np.ndarray],
              pair_seed: int = None) -> float:
        """Inlier count after Lowe ratio test + RANSAC homography.

        ``pair_seed`` (if given) reseeds OpenCV's global RNG immediately before
        findHomography, so the RANSAC inlier count for a (probe, gallery) pair is
        independent of how many other pairs were scored first -- making serial,
        parallel and resumed runs identical. Without it, results depend on
        execution order (cv2's RNG is global and advances per findHomography call).
        """
        da, db = a["desc"], b["desc"]
        if len(da) < 2 or len(db) < 2:
            return 0.0
        good = []
        for pair in self.matcher.knnMatch(da, db, k=2):
            if len(pair) == 2 and pair[0].distance < self.lowe_ratio * pair[1].distance:
                good.append(pair[0])
        if len(good) < self.min_matches:
            return 0.0  # homography impossible -> score 0
        src = a["pts"][[m.queryIdx for m in good]]
        dst = b["pts"][[m.trainIdx for m in good]]
        if pair_seed is not None:
            cv2.setRNGSeed(int(pair_seed))
        _, mask = cv2.findHomography(src, dst, cv2.RANSAC, self.reproj_thresh,
                                     maxIters=self.max_iters)
        if mask is None:
            return 0.0
        return float(int(mask.sum()))
