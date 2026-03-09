"""
FrontierSemanticMapper: coordinates per-step and periodic mapping operations.

Calling contract
----------------
Every step:
    n_updated = mapper.step(depth, rgb, pose, step_idx)

Every N steps (or on demand):
    frontier_pts = mapper.update(agent_pos, step_idx)

This separation is intentional: integrating depth and accumulating visual
features are cheap and must run every step so frontiers are seen from all
angles. Occupancy queries and frontier detection are expensive and only
need to run periodically.
"""

from __future__ import annotations

from typing import Optional
import numpy as np

from src.mapping.configs import FrontierDetectorConfig, MappingConfig
from src.mapping.frontier_detector import FrontierDetector
from src.mapping.wavemap import WaveMapper
from src.rayfront.frontier_regions import FrontierRegionMap, FrontierRegionsConfig
from src.features.extractor import FeatureExtractor


class FrontierSemanticMapper:
    """
    Coordinates wavemap integration, frontier detection, region map maintenance,
    and per-step visual feature accumulation.

    The FeatureExtractor is injected rather than owned — it is a heavy shared
    resource that callers typically reuse across components.
    """

    def __init__(
        self,
        map_cfg: MappingConfig,
        front_cfg: FrontierDetectorConfig,
        regions_cfg: FrontierRegionsConfig,
        sensor_width: int,
        sensor_height: int,
        intrinsics: np.ndarray,
        extractor: Optional[FeatureExtractor],
    ) -> None:
        self._mapper = WaveMapper(map_cfg, sensor_width, sensor_height, intrinsics)
        self._detector = FrontierDetector(front_cfg, voxel_size=map_cfg.voxel_size)
        self._region_map = FrontierRegionMap(regions_cfg)
        self._extractor = extractor
        self._intrinsics = intrinsics

        self.frontier_pts: np.ndarray = np.empty((0, 3), dtype=np.float32)
        self._last_feat_map: np.ndarray | None = None


    def step(
        self,
        depth: np.ndarray,
        rgb: np.ndarray,
        pose: np.ndarray,
        step_idx: int = 0,
    ) -> int:
        """
        Per-step update: integrate depth and accumulate features for all
        frontier regions currently visible from this pose.

        Parameters
        ----------
        depth    : (H, W) float32 metres
        rgb      : (H, W, 3) uint8
        pose     : (4, 4) camera-to-world, OpenGL convention
        step_idx : current exploration step counter

        Returns
        -------
        Number of frontier regions that received a feature update.
        """
        self._mapper.integrate(depth, pose)
        if self._extractor is not None:
            feat_map, _ = self._extractor.extract_dense(rgb)
            self._last_feat_map = feat_map
            return self._region_map.update_features(depth, feat_map, pose, self._intrinsics)
        return 0

    def update(self, agent_pos: np.ndarray, step_idx: int = 0) -> np.ndarray:
        """
        Periodic update: refresh occupancy, detect frontiers, sync region map.
        Call every N steps or whenever the map needs refreshing.

        Parameters
        ----------
        agent_pos : (3,) agent body position (world frame, Y-up)
        step_idx  : current exploration step counter

        Returns
        -------
        (N, 3) frontier voxel centres in world frame.
        """
        self._mapper.update_occupancy(agent_pos=agent_pos)
        self.frontier_pts = self._detector.detect(self._mapper, floor_y=float(agent_pos[1]))
        self._region_map.sync(self.frontier_pts)
        self._region_map.invalidate_near(agent_pos)
        return self.frontier_pts

    @property
    def region_map(self) -> FrontierRegionMap:
        return self._region_map

    @property
    def last_feat_map(self) -> np.ndarray | None:
        """Feature map from the most recent step() call, shape (ph, pw, D), L2-normalised."""
        return self._last_feat_map

    @property
    def free_pts(self):
        return self._mapper.free_pts

    @property
    def occ_pts(self):
        return self._mapper.occ_pts
