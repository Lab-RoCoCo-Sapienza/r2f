"""
3D frontier detection from wavemap occupancy.

A voxel is a frontier if it is:
  - Free (log-odds < free_threshold)
  - Has at least min_unknown_neighbors face-neighbors that are unknown
    (log-odds in (free_threshold, occ_threshold), i.e. not yet observed)
  - Within the configured height band (avoids floor/ceiling noise)

The output is subsampled to a coarser grid (subsample_cell) to keep the
frontier set manageable.

Usage
-----
    detector = FrontierDetector(cfg, voxel_size=mapper._res)
    # After mapper.update_occupancy():
    frontier_pts = detector.detect(mapper)   # (N, 3) world-frame positions
"""

from __future__ import annotations

import numpy as np

from src.mapping.configs import FrontierDetectorConfig
from src.mapping.wavemap import WaveMapper

# Stride for 1D index encoding — must be larger than the voxel index range
# (query space ±20m at 0.1m = ±200 voxels → stride of 1000 is safe)
_STRIDE = 1_000


class FrontierDetector:
    """
    Detects frontier voxels from a WaveMapper's cached free/occupied grids.

    Detection is fully vectorised: no per-voxel Python loop.
    """

    def __init__(self, cfg: FrontierDetectorConfig, voxel_size: float) -> None:
        self._cfg = cfg
        self._res = voxel_size

        # 6-connected face-neighbor offsets in voxel-index space
        self._face_offsets = np.array(
            [[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]],
            dtype=np.int64,
        )

    def detect(self, mapper: WaveMapper, floor_y: float = 0.0) -> np.ndarray:
        """
        Return (N, 3) world-frame frontier voxel centres.

        Requires mapper.update_occupancy() to have been called first.

        floor_y : agent body Y (world frame) — height filter is applied relative to this.
        """
        free = mapper.free_pts
        occ = mapper.occ_pts

        if free is None or len(free) == 0:
            return np.empty((0, 3), dtype=np.float32)

        # Keep unfiltered free set for free-neighbor lookup (includes floor/ceiling)
        free_all = free

        # Height filter relative to floor — restrict candidates to navigable band
        h_min = floor_y + self._cfg.height_min
        h_max = floor_y + self._cfg.height_max
        h_mask = (free[:, 1] >= h_min) & (free[:, 1] <= h_max)
        free = free[h_mask]

        if len(free) == 0:
            return np.empty((0, 3), dtype=np.float32)

        # Convert world positions to integer voxel indices
        res = self._res
        free_idx = np.round(free / res).astype(np.int64)       # [M, 3] height-filtered
        free_all_idx = np.round(free_all / res).astype(np.int64)  # [M_all, 3] all free

        S = _STRIDE

        # Known set: free + occupied (for unknown-neighbor test)
        known_idx = free_all_idx
        if occ is not None and len(occ) > 0:
            occ_idx = np.round(occ / res).astype(np.int64)
            known_idx = np.vstack([free_all_idx, occ_idx])

        known_codes = (
            known_idx[:, 0] * S * S +
            known_idx[:, 1] * S +
            known_idx[:, 2]
        )
        known_codes_sorted = np.sort(known_codes)

        # Free set codes (for free-neighbor test)
        free_all_codes = (
            free_all_idx[:, 0] * S * S +
            free_all_idx[:, 1] * S +
            free_all_idx[:, 2]
        )
        free_all_codes_sorted = np.sort(free_all_codes)

        # For each candidate free voxel compute the 6 neighbor codes — [M, 6]
        nb_idx = free_idx[:, np.newaxis, :] + self._face_offsets[np.newaxis, :, :]  # [M, 6, 3]
        nb_codes = (
            nb_idx[:, :, 0] * S * S +
            nb_idx[:, :, 1] * S +
            nb_idx[:, :, 2]
        )  # [M, 6]
        nb_flat = nb_codes.reshape(-1)  # [M*6]

        # Unknown-neighbor count
        ins = np.searchsorted(known_codes_sorted, nb_flat)
        cap = len(known_codes_sorted) - 1
        matched = known_codes_sorted[np.minimum(ins, cap)]
        is_known = (matched == nb_flat).reshape(len(free_idx), 6)
        n_unknown = (~is_known).sum(axis=1)  # [M]

        # Free-neighbor count — rejects voxels embedded in walls
        ins_f = np.searchsorted(free_all_codes_sorted, nb_flat)
        cap_f = len(free_all_codes_sorted) - 1
        matched_f = free_all_codes_sorted[np.minimum(ins_f, cap_f)]
        is_free = (matched_f == nb_flat).reshape(len(free_idx), 6)
        n_free = is_free.sum(axis=1)  # [M]

        is_frontier = (
            (n_unknown >= self._cfg.min_unknown_neighbors) &
            (n_free >= self._cfg.min_free_neighbors)
        )

        frontier_pts = free[is_frontier]

        if len(frontier_pts) == 0:
            return np.empty((0, 3), dtype=np.float32)

        return self._subsample(frontier_pts)

    def _subsample(self, pts: np.ndarray) -> np.ndarray:
        """Keep the first representative per coarser-grid cell."""
        cell = self._cfg.subsample_cell
        cell_idx = np.floor(pts / cell).astype(np.int64)

        S = 100_000
        codes = (
            cell_idx[:, 0] * S * S +
            cell_idx[:, 1] * S +
            cell_idx[:, 2]
        )
        _, first_occ = np.unique(codes, return_index=True)
        return pts[first_occ]
