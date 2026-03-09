"""
Online voxel mapping via wavemap (hashed chunked wavelet octree).

Adapted from FrontierNet/mapping/wavemap.py.

Usage
-----
    mapper = WaveMapper(cfg, width=640, height=480, intrinsics=obs.intrinsics)
    mapper.integrate(obs.depth, obs.pose)   # call each step
    mapper.update_occupancy()               # call when you need fresh voxels
    grid = mapper.get_occupancy()           # {"occupied": (N,3), "free": (M,3)}

Adapter API (single-point and neighborhood queries)
----------------------------------------------------
    mapper.is_free(x)                       # bool
    mapper.is_occupied(x)                   # bool
    mapper.is_observed(x)                   # bool (log-odds != prior)
    mapper.neighbor_counts(v, r)            # (n_free, n_occ, n_unknown)
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pywavemap as wave
from pywavemap import InterpolationMode

from src.mapping.configs import MappingConfig


class WaveMapper:
    """
    Wraps pywavemap to maintain an online occupancy map from posed depth frames.

    Parameters
    ----------
    cfg:        MappingConfig with voxel_size, range limits, and thresholds.
    width:      depth image width in pixels.
    height:     depth image height in pixels.
    intrinsics: 3x3 camera intrinsic matrix K (from obs.intrinsics).
    """

    def __init__(
        self,
        cfg: MappingConfig,
        width: int,
        height: int,
        intrinsics: np.ndarray,
    ) -> None:
        fx = float(intrinsics[0, 0])
        fy = float(intrinsics[1, 1])
        cx = float(intrinsics[0, 2])
        cy = float(intrinsics[1, 2])

        self._occ_thr = cfg.occ_threshold
        self._free_thr = cfg.free_threshold
        self._res = cfg.voxel_size

        self._map = wave.Map.create({
            "type": "hashed_chunked_wavelet_octree",
            "min_cell_width": {"meters": self._res / 2.0},
        })

        self._pipeline = wave.Pipeline(self._map)
        self._pipeline.add_operation(
            {"type": "threshold_map", "once_every": {"seconds": 5.0}}
        )
        self._pipeline.add_integrator(
            "integrator",
            {
                "projection_model": {
                    "type": "pinhole_camera_projector",
                    "width": width,
                    "height": height,
                    "fx": fx,
                    "fy": fy,
                    "cx": cx,
                    "cy": cy,
                },
                "measurement_model": {
                    "type": "continuous_ray",
                    "range_sigma": {"meters": 0.05},
                    "scaling_free": 0.2,
                    "scaling_occupied": 0.4,
                },
                "integration_method": {
                    "type": "hashed_chunked_wavelet_integrator",
                    "min_range": {"meters": cfg.min_range},
                    "max_range": {"meters": cfg.max_range},
                },
            },
        )

        self._query_half_xz = 20.0  # metres: query radius in XZ around the agent
        self._query_y_lo = -1.0    # metres: absolute Y lower bound of query slab
        self._query_y_hi = 2.5     # metres: absolute Y upper bound of query slab

        self.occ_pts = None
        self.free_pts = None

        # obs.pose is OpenGL convention (+Y up, -Z forward).
        # pywavemap expects OpenCV convention (+Y down, +Z forward).
        # Post-multiplying by this matrix flips Y and Z in the camera frame.
        self._GL_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])

    def integrate(self, depth: np.ndarray, pose: np.ndarray) -> None:
        """
        Integrate one depth frame into the map.

        depth: (H, W) float32, metres.
        pose:  (4, 4) camera-to-world SE(3) (obs.pose, OpenGL convention).
        """
        # Convert from Habitat/OpenGL camera frame to OpenCV (wavemap) convention
        pose_cv = pose @ self._GL_TO_CV
        # wavemap expects image as (W, H), so transpose
        image = wave.Image(np.asarray(depth, dtype=np.float32).T)
        self._pipeline.run_pipeline(
            ["integrator"], wave.PosedImage(wave.Pose(pose_cv), image)
        )
        self._map.prune()

    def update_occupancy(self, agent_pos: np.ndarray | None = None) -> None:
        """
        Query the map at every grid point and refresh occ_pts / free_pts.
        Call after integrate() when you need an up-to-date voxel grid.

        agent_pos: (3,) world-frame agent position used to centre the XZ query
                   window. If None, centres on the origin (legacy behaviour).
        """
        cx = float(agent_pos[0]) if agent_pos is not None else 0.0
        cz = float(agent_pos[2]) if agent_pos is not None else 0.0
        r = self._query_half_xz
        res = self._res

        xs = np.arange(cx - r, cx + r, res, dtype=np.float32)
        ys = np.arange(self._query_y_lo, self._query_y_hi, res, dtype=np.float32)
        zs = np.arange(cz - r, cz + r, res, dtype=np.float32)
        gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
        query_space = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1).astype(np.float32)

        log_odds = self._map.interpolate(
            query_space, InterpolationMode.NEAREST
        ).reshape(-1)

        self.occ_pts = query_space[log_odds > self._occ_thr]
        self.free_pts = query_space[log_odds < self._free_thr]

    def get_occupancy(self) -> dict:
        """Return {"occupied": (N,3) or None, "free": (M,3) or None}."""
        return {"occupied": self.occ_pts, "free": self.free_pts}

    # Adapter API — single-point queries

    def _query_log_odds(self, x: np.ndarray) -> float:
        """Query log-odds at a single world-frame point."""
        return float(
            self._map.interpolate(x[np.newaxis], InterpolationMode.NEAREST)[0]
        )

    def is_free(self, x: np.ndarray) -> bool:
        """True if x is in observed free space."""
        return self._query_log_odds(x) < self._free_thr

    def is_occupied(self, x: np.ndarray) -> bool:
        """True if x is in observed occupied space."""
        return self._query_log_odds(x) > self._occ_thr

    def is_observed(self, x: np.ndarray) -> bool:
        """True if x has been observed (log-odds differs from prior of 0)."""
        return abs(self._query_log_odds(x)) > 1e-6

    # Adapter API — neighborhood query

    def neighbor_counts(self, v: np.ndarray, r: float) -> Tuple[int, int, int]:
        """
        Count free, occupied, and unknown voxels within radius r of v.

        Returns
        -------
        (n_free, n_occupied, n_unknown)
        """
        n = int(np.ceil(r / self._res))
        ax = np.arange(-n, n + 1) * self._res
        # Build 3D grid of offsets
        gx, gy, gz = np.meshgrid(ax, ax, ax, indexing="ij")
        offsets = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)  # [K, 3]

        # Filter to sphere
        dists = np.linalg.norm(offsets, axis=1)
        offsets = offsets[dists <= r]
        nb_pts = v + offsets  # [K', 3]

        lo = self._map.interpolate(nb_pts, InterpolationMode.NEAREST)
        n_free = int((lo < self._free_thr).sum())
        n_occ = int((lo > self._occ_thr).sum())
        n_unknown = len(lo) - n_free - n_occ
        return n_free, n_occ, n_unknown
