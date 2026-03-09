"""
Sparse semantic voxel map with weighted feature aggregation.

Each voxel stores a running weighted average of SigLIP patch features
from all depth-unprojected observations that land in it.  Features are
rollout-weighted (see FeatureExtractor.extract_dense) so they carry
contextual attention information before being stored.

Cosine similarity against a SigLIP text embedding gives per-voxel
language alignment scores for zero-shot VLN.

Usage
-----
    svm = SemanticVoxelMap(cfg)
    # After FeatureExtractor.extract_dense(rgb):
    svm.update(depth, siglip_patches, pose, intrinsics)
    # Periodically remove surface-inconsistent voxels:
    svm.prune(mapper)
    # Score against text:
    positions, scores = svm.query_similarity(text_emb)
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from src.mapping.configs import SemanticVoxelMapConfig

_GL_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])

# Stride for signed 1D voxel key encoding
# Max building ~50m at 0.1m => ±500 vox indices; stride 10000 is safe
_STRIDE = 10_000


class SemanticVoxelMap:
    """
    Sparse map: voxel (ix, iy, iz) -> (feature_sum, weight).

    Features are L2-normalised rollout-weighted SigLIP patch vectors.
    On query, each voxel's mean feature is re-normalised before cosine scoring.
    """

    def __init__(self, cfg: SemanticVoxelMapConfig) -> None:
        self._cfg = cfg
        # Tuple (ix, iy, iz) -> accumulated (D,) feature sum and observation count
        self._feat_sum: Dict[Tuple[int, int, int], np.ndarray] = {}
        self._weight: Dict[Tuple[int, int, int], float] = {}
        # Cache for get_features() — invalidated by update/prune
        self._dirty = True
        self._cached_pos: Optional[np.ndarray] = None
        self._cached_feats: Optional[np.ndarray] = None

    def update(
        self,
        depth: np.ndarray,
        feat_map: np.ndarray,
        pose: np.ndarray,
        intrinsics: np.ndarray,
    ) -> int:
        """
        Fuse one RGB-D frame into the semantic voxel map.

        Parameters
        ----------
        depth    : (H, W) float32, metres; 0 or >= max_range treated as invalid
        feat_map : (ph, pw, D) float32, L2-normalised SigLIP patch features
        pose     : (4, 4) camera-to-world, OpenGL convention (Y-up, -Z fwd)
        intrinsics : (3, 3) camera intrinsics [fx,0,cx; 0,fy,cy; 0,0,1]

        Returns
        -------
        Number of voxels created or updated this frame.
        """
        H, W = depth.shape
        ph, pw, D = feat_map.shape
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]

        # OpenGL -> OpenCV: flip Y and Z axes of camera orientation
        pose_cv = pose @ _GL_TO_CV  # [4, 4]

        u_g, v_g = np.meshgrid(np.arange(W), np.arange(H))
        valid = (depth > 0.0) & (depth < self._cfg.max_range)
        if not valid.any():
            return 0

        u_v = u_g[valid]
        v_v = v_g[valid]
        d_v = depth[valid]  # [N]

        # Unproject to camera frame (OpenCV: +Z forward, +Y down)
        x_c = (u_v - cx) / fx * d_v
        y_c = (v_v - cy) / fy * d_v
        pts_c = np.stack([x_c, y_c, d_v, np.ones(len(d_v))], axis=0)  # [4, N]

        # Transform to world
        pts_w = (pose_cv @ pts_c)[:3].T  # [N, 3]

        # Nearest-neighbour lookup from pixel coords into (ph, pw) patch grid
        pi = (v_v * ph / H).astype(np.int32).clip(0, ph - 1)
        pj = (u_v * pw / W).astype(np.int32).clip(0, pw - 1)
        feats = feat_map[pi, pj]  # [N, D]

        # Voxelise
        vox = np.floor(pts_w / self._cfg.voxel_size).astype(np.int32)  # [N, 3]

        # Encode to signed 1D keys for grouping
        S = _STRIDE
        keys_1d = (
            vox[:, 0].astype(np.int64) * S * S
            + vox[:, 1].astype(np.int64) * S
            + vox[:, 2].astype(np.int64)
        )  # [N]

        # Group by unique voxel — get first-occurrence coords for decoding
        unique_keys, first_idx, inverse = np.unique(
            keys_1d, return_index=True, return_inverse=True
        )
        U = len(unique_keys)

        # Vectorised feature sum per unique voxel
        feat_sums = np.zeros((U, D), dtype=np.float32)
        np.add.at(feat_sums, inverse, feats)  # [U, D]
        counts = np.bincount(inverse, minlength=U).astype(np.float32)  # [U]

        # Representative voxel coords for each unique key
        vox_unique = vox[first_idx]  # [U, 3]

        # Update map
        for ui in range(U):
            k = (int(vox_unique[ui, 0]), int(vox_unique[ui, 1]), int(vox_unique[ui, 2]))
            if k in self._feat_sum:
                self._feat_sum[k] += feat_sums[ui]
                self._weight[k] += counts[ui]
            else:
                self._feat_sum[k] = feat_sums[ui].copy()
                self._weight[k] = float(counts[ui])

        self._dirty = True
        return U

    def prune(self, mapper) -> int:
        """
        Remove voxels whose centre the occupancy map no longer marks as occupied.

        Parameters
        ----------
        mapper : WaveMapper

        Returns
        -------
        Number of voxels removed.
        """
        vs = self._cfg.voxel_size
        to_del = [
            k for k in self._feat_sum
            if not mapper.is_occupied(np.array(k, dtype=float) * vs + vs * 0.5)
        ]
        for k in to_del:
            del self._feat_sum[k]
            del self._weight[k]
        if to_del:
            self._dirty = True
        return len(to_del)

    def get_features(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return all voxel positions and their mean normalised features.

        Returns
        -------
        positions : (N, 3) float32 world-frame voxel centres
        features  : (N, D) float32 L2-normalised
        """
        if not self._dirty and self._cached_pos is not None:
            return self._cached_pos, self._cached_feats

        N = len(self._feat_sum)
        if N == 0:
            empty_p = np.empty((0, 3), dtype=np.float32)
            empty_f = np.empty((0, self._cfg.feature_dim), dtype=np.float32)
            return empty_p, empty_f

        keys = list(self._feat_sum.keys())
        vs = self._cfg.voxel_size
        positions = (np.array(keys, dtype=np.float32) * vs + vs * 0.5)  # [N, 3] centres
        feat_sums = np.stack([self._feat_sum[k] for k in keys])  # [N, D]
        weights = np.array([self._weight[k] for k in keys], dtype=np.float32)  # [N]

        means = feat_sums / weights[:, None]  # [N, D]
        norms = np.linalg.norm(means, axis=1, keepdims=True).clip(min=1e-8)
        features = (means / norms).astype(np.float32)  # [N, D] L2-normalised

        self._cached_pos = positions
        self._cached_feats = features
        self._dirty = False
        return positions, features

    def query_similarity(
        self, text_emb: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Score all voxels against a normalised text embedding.

        Parameters
        ----------
        text_emb : (D,) float32 L2-normalised SigLIP text embedding

        Returns
        -------
        positions : (N, 3)
        scores    : (N,) cosine similarity in [-1, 1]
        """
        positions, features = self.get_features()
        if len(features) == 0:
            return positions, np.empty(0, dtype=np.float32)
        scores = features @ text_emb  # [N]  cosine (both normalised)
        return positions, scores.astype(np.float32)

    @property
    def num_voxels(self) -> int:
        return len(self._feat_sum)
