"""
FrontierRegionMap: flat map of semantic ray-frontier regions.

Based on RayFronts (Alama et al., 2025), Sec. III-E.

Each frontier region accumulates semantic features from OOR pixels — pixels where
depth >= max_range, i.e. the camera is looking INTO unexplored space. Features are
organised into direction angle bins (one bin = one discretised viewing direction),
enabling multi-object storage without feature collision.

update_features() pipeline per step:
  1. Compute OOR mask (depth >= max_range) and erode to avoid boundary leakage
  2. Sample rays from OOR pixels: direction = camera->pixel projected to world frame
  3. Associate each ray to nearest frontier via d_cost (orthogonal + radial distance)
  4. Accumulate the OOR pixel feature into the frontier's (theta_bin, phi_bin) slot
  5. Score: max cosine similarity over all ray bins of each frontier
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

_GL_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])


def _erode_mask(mask: np.ndarray, r: int) -> np.ndarray:
    """Erode a boolean mask by r pixels using an integral-image window sum."""
    if r <= 0:
        return mask
    H, W = mask.shape
    padded = np.pad(mask.astype(np.int32), r, constant_values=0)
    # SAT with 1-indexed border so i0 can be 0 without negative indexing
    integral = np.zeros((H + 2*r + 1, W + 2*r + 1), dtype=np.int32)
    integral[1:, 1:] = padded.cumsum(axis=0).cumsum(axis=1)
    size = 2 * r + 1
    i0, i1 = np.arange(H), np.arange(H) + size  # i1 max = H-1+size = H+2r (valid)
    j0, j1 = np.arange(W), np.arange(W) + size
    win_sum = (
        integral[np.ix_(i1, j1)]
        - integral[np.ix_(i0, j1)]
        - integral[np.ix_(i1, j0)]
        + integral[np.ix_(i0, j0)]
    )
    return win_sum == size * size
_region_counter = itertools.count()

# Max OOR pixels to cast rays from per frame (random subsample if exceeded)
_MAX_RAYS = 500


@dataclass
class FrontierRegionsConfig:
    merge_radius: float = 0.8       # voxels within this distance -> same region
    max_range: float = 4.0          # depth >= this -> OOR pixel
    erosion_radius: int = 3         # pixels: shrink OOR mask to avoid boundary leakage
    assoc_max_dist: float = 4.0     # m: skip frontiers farther than this from camera
    assoc_ortho_dist: float = 1.0   # m: beta — max orthogonal distance for ray-frontier match
    angle_bin_size: float = 30.0    # psi (degrees): angular resolution for direction bins
    invalidate_radius: float = 1.0  # m: remove regions when agent is this close
    visited_filter_dist: float = 1.4  # m: avoid suppressing doorway/frontier regions after passing nearby


class _FrontierRegion:
    def __init__(self, pt: np.ndarray) -> None:
        self.id: int = next(_region_counter)
        self.member_pts: List[np.ndarray] = [pt.copy()]
        self.centroid: np.ndarray = pt.copy()
        # rays: maps (theta_bin, phi_bin) -> (feature [D], total_weight, direction [3])
        self.rays: Dict[Tuple[int, int], Tuple[np.ndarray, float, np.ndarray]] = {}

    @property
    def has_feature(self) -> bool:
        return len(self.rays) > 0

    def _recompute_centroid(self) -> None:
        self.centroid = np.mean(self.member_pts, axis=0).astype(np.float32)


class FrontierRegionMap:
    """
    Flat map of semantic ray-frontier regions.

    Call sync() every time the frontier set is refreshed (periodic).
    Call update_features() every step to cast OOR rays and accumulate features.
    """

    def __init__(self, cfg: FrontierRegionsConfig) -> None:
        self._cfg = cfg
        self._regions: List[_FrontierRegion] = []
        self._visited_xz: List[np.ndarray] = []

    def sync(self, frontier_pts: np.ndarray) -> None:
        """
        Update regions to match the current set of frontier voxels.

        Each frontier_pt is assigned to the nearest existing region centroid
        if within merge_radius, otherwise a new region is created.
        Regions with no member_pts in the new set are removed.
        Accumulated ray features are preserved across sync calls.
        """
        if frontier_pts is None or len(frontier_pts) == 0:
            self._regions.clear()
            return

        region_members: List[List[np.ndarray]] = [[] for _ in self._regions]
        unassigned: List[np.ndarray] = []

        centroids = (
            np.array([r.centroid for r in self._regions], dtype=np.float32)
            if self._regions else None
        )

        for pt in frontier_pts:
            if centroids is not None:
                dists = np.linalg.norm(centroids - pt, axis=1)
                best_idx = int(np.argmin(dists))
                if dists[best_idx] < self._cfg.merge_radius:
                    region_members[best_idx].append(pt.copy())
                    continue
            unassigned.append(pt.copy())

        kept: List[_FrontierRegion] = []
        for reg, members in zip(self._regions, region_members):
            if members:
                reg.member_pts = members
                reg._recompute_centroid()
                kept.append(reg)

        # Cluster unassigned points greedily so the first sync() doesn't create
        # one region per voxel.
        new_centroids: List[np.ndarray] = []
        new_member_lists: List[List[np.ndarray]] = []
        for pt in unassigned:
            placed = False
            for i, nc in enumerate(new_centroids):
                if float(np.linalg.norm(nc - pt)) < self._cfg.merge_radius:
                    new_member_lists[i].append(pt.copy())
                    new_centroids[i] = np.mean(new_member_lists[i], axis=0).astype(np.float32)
                    placed = True
                    break
            if not placed:
                new_centroids.append(pt.copy().astype(np.float32))
                new_member_lists.append([pt.copy()])

        for pts_list in new_member_lists:
            reg = _FrontierRegion(pts_list[0])
            reg.member_pts = pts_list
            reg._recompute_centroid()
            kept.append(reg)

        self._regions = kept

    def update_features(
        self,
        depth: np.ndarray,
        feat_map: np.ndarray,
        pose: np.ndarray,
        intrinsics: np.ndarray,
    ) -> int:
        """
        Cast semantic rays from OOR pixels and accumulate features into frontier
        angle bins (RayFronts Sec. III-E: Observe -> Associate -> Discretize & Accumulate).

        Returns the number of (ray, frontier) associations made this frame.
        """
        if not self._regions:
            return 0

        H, W = depth.shape
        ph, pw = feat_map.shape[:2]

        # 1. OOR mask: pixels where the camera looks into unexplored space
        oor_mask = depth >= self._cfg.max_range

        # 2. Erode to prevent semantic leakage at object boundaries
        oor_mask = _erode_mask(oor_mask, self._cfg.erosion_radius)

        oor_vi, oor_ui = np.where(oor_mask)
        if len(oor_vi) == 0:
            return 0

        # 3. Subsample OOR pixels if too many
        if len(oor_vi) > _MAX_RAYS:
            idx = np.random.choice(len(oor_vi), _MAX_RAYS, replace=False)
            oor_vi = oor_vi[idx]
            oor_ui = oor_ui[idx]

        # 4. Ray directions in world frame
        fx = float(intrinsics[0, 0])
        fy = float(intrinsics[1, 1])
        cx = float(intrinsics[0, 2])
        cy = float(intrinsics[1, 2])

        cam_pos = pose[:3, 3]
        R_cv = (pose @ _GL_TO_CV)[:3, :3]  # OpenCV camera -> world

        ray_dirs_cam = np.stack([
            (oor_ui - cx) / fx,
            (oor_vi - cy) / fy,
            np.ones(len(oor_ui)),
        ], axis=1).astype(np.float32)  # [N, 3]
        norms = np.linalg.norm(ray_dirs_cam, axis=1, keepdims=True)
        ray_dirs_cam /= np.where(norms > 1e-8, norms, 1.0)
        ray_dirs_world = (R_cv @ ray_dirs_cam.T).T  # [N, 3]

        # 5. Features at OOR pixels (nearest patch)
        pi = np.minimum((oor_vi * ph / H).astype(int), ph - 1)
        pj = np.minimum((oor_ui * pw / W).astype(int), pw - 1)
        ray_feats = feat_map[pi, pj].astype(np.float32)  # [N, D]
        feat_norms = np.linalg.norm(ray_feats, axis=1, keepdims=True)
        valid = feat_norms[:, 0] > 1e-8
        ray_dirs_world = ray_dirs_world[valid]
        ray_feats = ray_feats[valid] / feat_norms[valid]
        N = len(ray_feats)
        if N == 0:
            return 0

        # 6. Frontier geometry: precompute once (same cam_pos for all rays)
        centroids = np.array([reg.centroid for reg in self._regions], dtype=np.float32)  # [M, 3]
        vecs = centroids - cam_pos  # [M, 3]
        d_orig = np.linalg.norm(vecs, axis=1)  # [M]
        dist_ok = d_orig <= self._cfg.assoc_max_dist  # [M]

        # 7. Vectorised association filtering: [N, M]
        dots = ray_dirs_world @ vecs.T  # [N, M]
        proj = dots[:, :, np.newaxis] * ray_dirs_world[:, np.newaxis, :]  # [N, M, 3]
        d_ortho = np.linalg.norm(vecs[np.newaxis, :, :] - proj, axis=2)  # [N, M]

        in_front = dots > 0
        close_enough = d_ortho <= self._cfg.assoc_ortho_dist
        valid_pairs = in_front & close_enough & dist_ok[np.newaxis, :]  # [N, M]

        psi_rad = np.deg2rad(self._cfg.angle_bin_size)
        count = 0

        for n in range(N):
            valid_n = valid_pairs[n]
            if not valid_n.any():
                continue

            valid_idx = np.where(valid_n)[0]
            d_ortho_n = d_ortho[n, valid_idx]
            d_orig_n = d_orig[valid_idx]

            # d_cost: normalised per-ray as in the paper
            max_do = max(float(d_ortho_n.max()), 1e-8)
            max_dr = max(float(d_orig_n.max()), 1e-8)
            costs = (d_ortho_n / max_do + d_orig_n / max_dr) / 2.0

            best_local = int(np.argmin(costs))
            best_global = int(valid_idx[best_local])
            weight = max(0.0, 1.0 - float(costs[best_local]))
            if weight <= 0.0:
                continue

            # Discretize ray direction into (theta, phi) angle bin
            d = ray_dirs_world[n]
            theta = float(np.arctan2(d[1], d[0]))           # [-pi, pi]
            phi   = float(np.arccos(np.clip(d[2], -1.0, 1.0)))  # [0, pi]
            bin_key = (int(np.floor(theta / psi_rad)), int(np.floor(phi / psi_rad)))

            feat = ray_feats[n]
            reg = self._regions[best_global]

            if bin_key not in reg.rays:
                reg.rays[bin_key] = (feat.copy(), weight, d.copy())
            else:
                old_feat, old_w, old_dir = reg.rays[bin_key]
                total = old_w + weight
                new_feat = (old_w * old_feat + weight * feat) / total
                new_dir = (old_w * old_dir + weight * d) / total
                dn = float(np.linalg.norm(new_dir))
                new_dir = new_dir / dn if dn > 1e-8 else new_dir
                reg.rays[bin_key] = (new_feat.astype(np.float32), total, new_dir.astype(np.float32))

            count += 1

        return count

    def invalidate_near(self, pos3d: np.ndarray) -> None:
        """Remove all regions whose centroid is within invalidate_radius of pos3d.

        Uses XZ-only distance so that frontier centroids at camera height (y~1.5)
        are correctly removed when compared against floor-level agent positions.
        """
        r = self._cfg.invalidate_radius
        self._regions = [
            reg for reg in self._regions
            if float(np.linalg.norm(reg.centroid[[0, 2]] - pos3d[[0, 2]])) >= r
        ]

    def record_visit(self, pos3d: np.ndarray) -> None:
        """Record an agent position so nearby frontier regions are excluded from score_all."""
        self._visited_xz.append(pos3d[[0, 2]].copy())

    def _is_novel(self, centroid: np.ndarray) -> bool:
        xz = centroid[[0, 2]]
        r = self._cfg.visited_filter_dist
        return all(float(np.linalg.norm(xz - v)) >= r for v in self._visited_xz)

    def score_all(
        self,
        text_emb: np.ndarray,
        anchor_embs: Optional[np.ndarray] = None,
        anchor_alpha: float = 0.3,
    ) -> List[Tuple[np.ndarray, np.ndarray, float, int]]:
        """
        Score all regions. Each region's score = max cosine similarity over all its
        ray bins. Direction = the ray direction of the best-scoring bin.
        Regions too close to any recorded visit position are excluded.

        If anchor_embs (N_anchors, D) is provided, adds a spatial context bonus:
            score = target_sim + anchor_alpha * mean_over_anchors(max_sim(anchor_i))
        This guides exploration toward regions where both the target and its
        spatially-related anchor objects are likely present.

        Returns list of (centroid, direction, score, region_id), highest score first.
        """
        results = []
        for reg in self._regions:
            if not self._is_novel(reg.centroid):
                continue
            if not reg.has_feature:
                continue
            best_score = -np.inf
            best_dir: Optional[np.ndarray] = None
            for feat, _, direction in reg.rays.values():
                norm = float(np.linalg.norm(feat))
                if norm < 1e-8:
                    continue
                s = float((feat / norm) @ text_emb)
                if s > best_score:
                    best_score = s
                    best_dir = direction
            if best_dir is None:
                continue
            # Anchor context bonus: mean over anchors of max sim across ray bins
            if anchor_embs is not None and len(anchor_embs) > 0:
                anchor_sims = []
                for a_emb in anchor_embs:
                    best_a = -np.inf
                    for feat, _, _ in reg.rays.values():
                        norm = float(np.linalg.norm(feat))
                        if norm < 1e-8:
                            continue
                        a_s = float((feat / norm) @ a_emb)
                        if a_s > best_a:
                            best_a = a_s
                    if best_a > -np.inf:
                        anchor_sims.append(best_a)
                if anchor_sims:
                    best_score += anchor_alpha * float(np.mean(anchor_sims))
            results.append((reg.centroid.copy(), best_dir.astype(np.float32), best_score, reg.id))
        results.sort(key=lambda x: x[2], reverse=True)
        return results

    @property
    def active_regions(self) -> List[_FrontierRegion]:
        return list(self._regions)

    def summary(self) -> str:
        n = len(self._regions)
        m = sum(1 for r in self._regions if r.has_feature)
        return f"FrontierRegionMap: {n} regions ({m} with features)"
