"""
Local Controller: reactive waypoint tracking.

Given a target waypoint in world frame, emits one discrete action per step.
Stateless -- no internal memory between calls.

Logic:
  1. Arrival check -> "stop"
  2. Misaligned   -> "turn_left" / "turn_right"
  3. Aligned      -> "move_forward"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from src.simulator.observation import Observation

logger = logging.getLogger(__name__)


@dataclass
class LocalControllerConfig:
    arrival_threshold: float = 0.3   # metres - "close enough" to waypoint
    align_threshold: float = 0.15    # radians (~12 deg) - turn vs go


class LocalController:
    """Reactive local controller for discrete action spaces.

    Stateless: each call to step() is independent.
    """

    def __init__(self, config: LocalControllerConfig) -> None:
        self.cfg = config

    def step(
        self,
        obs: Observation,
        target: np.ndarray,
    ) -> str:
        """Emit one discrete action to move toward *target*.

        Parameters
        ----------
        obs : Observation
            Current RGBD + pose.
        target : (3,) ndarray
            Target waypoint in world frame.

        Returns
        -------
        str
            One of "move_forward", "turn_left", "turn_right", "stop".
        """
        agent_pos = obs.pose[:3, 3]

        # XZ-plane distance (ignore Y)
        diff = target - agent_pos
        diff_xz = np.array([diff[0], diff[2]])
        dist = np.linalg.norm(diff_xz)

        if dist < self.cfg.arrival_threshold:
            logger.debug("Arrived (dist=%.3fm)", dist)
            return "stop"

        heading_error = self._heading_error(obs.pose, target)

        if abs(heading_error) > self.cfg.align_threshold:
            action = "turn_left" if heading_error > 0 else "turn_right"
            logger.debug(
                "Aligning: %s (heading=%.1f deg, dist=%.2fm)",
                action, np.rad2deg(heading_error), dist,
            )
            return action

        logger.debug(
            "Moving forward (heading=%.1f deg, dist=%.2fm)",
            np.rad2deg(heading_error), dist,
        )
        return "move_forward"

    def _heading_error(
        self,
        pose: np.ndarray,
        target: np.ndarray,
    ) -> float:
        """Signed heading error from agent forward to target (radians).

        Positive = target is to the left, negative = to the right.
        Computed in the XZ plane (Y ignored).
        """
        agent_pos = pose[:3, 3]

        forward = -pose[:3, 2]  # camera looks along -Z
        right = pose[:3, 0]     # X axis = right

        to_target = target - agent_pos
        to_target[1] = 0.0
        norm = np.linalg.norm(to_target)
        if norm < 1e-6:
            return 0.0
        to_target /= norm

        fwd_xz = np.array([forward[0], forward[2]])
        right_xz = np.array([right[0], right[2]])
        target_xz = np.array([to_target[0], to_target[2]])

        fwd_norm = np.linalg.norm(fwd_xz)
        if fwd_norm < 1e-6:
            return 0.0
        fwd_xz /= fwd_norm

        right_norm = np.linalg.norm(right_xz)
        if right_norm < 1e-6:
            return 0.0
        right_xz /= right_norm

        dot_fwd = np.dot(target_xz, fwd_xz)
        dot_right = np.dot(target_xz, right_xz)

        # Positive = target is to the left (turn_left)
        return float(np.arctan2(-dot_right, dot_fwd))
