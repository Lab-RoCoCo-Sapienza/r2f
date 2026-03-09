"""
Navigator: navmesh path-following for frontier-guided exploration.

Given a frontier target, computes a collision-free path via the navmesh
pathfinder and follows the waypoints using the local controller for
discrete action generation.

Flow:
  1. set_frontier(frontier, obs) -> find_path to frontier.pos3d
  2. step(obs) each tick -> follow waypoints, detect arrival/stall
  3. arrived or stalled -> caller picks next frontier
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, List, Optional, Protocol

import numpy as np

from src.navigation.local_controller import LocalController, LocalControllerConfig
from src.simulator.observation import Observation


class FrontierTarget(Protocol):
    """Duck-type interface expected by Navigator: anything with pos3d and id."""
    pos3d: np.ndarray
    id: int

logger = logging.getLogger(__name__)

# Type alias for the pathfinder callable
PathFinderFn = Callable[[np.ndarray, np.ndarray], Optional[List[np.ndarray]]]


@dataclass
class NavigatorConfig:
    arrival_threshold: float = 0.4     # metres - close enough to waypoint
    stall_patience: int = 15           # steps without progress before replan
    stall_move_threshold: float = 0.1  # metres - min position change per check
    max_replans: int = 3               # replan attempts before declaring STALLED
    max_steps_per_frontier: int = 80   # hard cap on total steps per frontier (catches sliding loops)


@dataclass
class NavigatorResult:
    """Returned by Navigator.step() each timestep."""
    action: str = "stop"
    arrived: bool = False
    stalled: bool = False
    active: bool = False


class Navigator:
    """
    Navmesh path follower for frontier targets.

    Usage::

        nav = Navigator(ctrl_cfg, nav_cfg, find_path_fn=env.find_path)
        nav.set_frontier(frontier, obs)

        while True:
            result = nav.step(obs)
            if result.arrived or result.stalled:
                break
            obs = env.step(result.action)
    """

    def __init__(
        self,
        controller_config: LocalControllerConfig,
        navigator_config: NavigatorConfig,
        find_path_fn: PathFinderFn,
    ) -> None:
        self._controller = LocalController(controller_config)
        self._find_path = find_path_fn
        self.cfg = navigator_config

        # Path state
        self._waypoints: List[np.ndarray] = []
        self._wp_idx: int = 0
        self._current_frontier: Optional[FrontierTarget] = None

        # Stall detection and replanning
        self._stall_counter: int = 0
        self._last_position: Optional[np.ndarray] = None
        self._replan_count: int = 0
        self._steps_on_frontier: int = 0

    @property
    def current_frontier(self) -> Optional[FrontierTarget]:
        return self._current_frontier

    @property
    def waypoint(self) -> Optional[np.ndarray]:
        """Current waypoint being tracked."""
        if self._wp_idx < len(self._waypoints):
            return self._waypoints[self._wp_idx]
        return None

    def set_frontier(
        self,
        frontier: FrontierTarget,
        obs: Observation,
    ) -> bool:
        """Plan a navmesh path to the frontier.

        Returns True if a path was found, False otherwise.
        """
        self.clear()
        self._current_frontier = frontier

        agent_pos = obs.body_pose[:3, 3]
        goal_pos = frontier.pos3d

        # Reject goals the agent is already standing at to avoid zero-step arrivals
        if self._xz_distance(agent_pos, goal_pos) < self.cfg.arrival_threshold:
            logger.info("Navigator: goal too close to agent, rejecting frontier %d", frontier.id)
            self._current_frontier = None
            return False

        path = self._find_path(agent_pos, goal_pos)
        if path is None or len(path) < 2:
            logger.info(
                "Navigator: no navmesh path to frontier %d at [%.2f, %.2f, %.2f]",
                frontier.id, goal_pos[0], goal_pos[1], goal_pos[2],
            )
            self._current_frontier = None
            return False

        # Skip start position, store remaining waypoints
        self._waypoints = path[1:]

        # Reject if the navmesh-snapped destination is already at the agent's feet.
        # Ceiling frontier voxels often snap to a floor point within arrival_threshold.
        if self._xz_distance(agent_pos, self._waypoints[-1]) < self.cfg.arrival_threshold:
            logger.info(
                "Navigator: snapped destination too close to agent, rejecting frontier %d", frontier.id
            )
            self._current_frontier = None
            self._waypoints = []
            return False

        self._wp_idx = 0
        self._stall_counter = 0
        self._replan_count = 0
        self._steps_on_frontier = 0
        self._last_position = agent_pos.copy()

        logger.info(
            "Navigator: path with %d waypoints to frontier %d",
            len(self._waypoints), frontier.id,
        )
        return True

    def clear(self) -> None:
        """Clear the current target and path."""
        self._current_frontier = None
        self._waypoints = []
        self._wp_idx = 0
        self._stall_counter = 0
        self._last_position = None
        self._replan_count = 0
        self._steps_on_frontier = 0

    def step(self, obs: Observation) -> NavigatorResult:
        """Emit one action toward the current frontier.

        Returns NavigatorResult with action and status flags.
        """
        if self._current_frontier is None or self._wp_idx >= len(self._waypoints):
            return NavigatorResult(action="stop", active=False)

        self._steps_on_frontier += 1
        if self._steps_on_frontier > self.cfg.max_steps_per_frontier:
            logger.info("Navigator: step cap reached (%d), frontier %d",
                        self._steps_on_frontier, self._current_frontier.id)
            return NavigatorResult(action="stop", stalled=True, active=True)

        agent_pos = obs.body_pose[:3, 3]
        wp = self._waypoints[self._wp_idx]

        # Arrival check for current waypoint (XZ distance)
        dist_xz = self._xz_distance(agent_pos, wp)
        if dist_xz < self.cfg.arrival_threshold:
            self._wp_idx += 1
            self._stall_counter = 0

            if self._wp_idx >= len(self._waypoints):
                # Reached final waypoint = frontier reached
                logger.info("Navigator: arrived at frontier %d", self._current_frontier.id)
                return NavigatorResult(action="stop", arrived=True, active=True)

            # Advance to next waypoint
            logger.debug(
                "Navigator: waypoint %d/%d reached, advancing",
                self._wp_idx, len(self._waypoints),
            )

        # Stall detection
        if self._last_position is not None:
            moved = self._xz_distance(agent_pos, self._last_position)
            if moved < self.cfg.stall_move_threshold:
                self._stall_counter += 1
            else:
                self._stall_counter = 0
        self._last_position = agent_pos.copy()

        if self._stall_counter >= self.cfg.stall_patience:
            self._stall_counter = 0
            self._replan_count += 1

            if self._replan_count > self.cfg.max_replans:
                logger.info("Navigator: stalled after %d replans, frontier %d",
                            self._replan_count, self._current_frontier.id)
                return NavigatorResult(action="stop", stalled=True, active=True)

            # Replan from current position
            goal_pos = self._current_frontier.pos3d
            path = self._find_path(agent_pos, goal_pos)
            if path is not None and len(path) >= 2:
                self._waypoints = path[1:]
                self._wp_idx = 0
                logger.info("Navigator: replanned (%d/%d), %d new waypoints",
                            self._replan_count, self.cfg.max_replans,
                            len(self._waypoints))
            else:
                logger.info("Navigator: replan failed, frontier %d",
                            self._current_frontier.id)
                return NavigatorResult(action="stop", stalled=True, active=True)

        # Get discrete action from local controller
        current_wp = self._waypoints[self._wp_idx]
        action = self._controller.step(obs, current_wp)

        return NavigatorResult(action=action, active=True)

    @staticmethod
    def _xz_distance(a: np.ndarray, b: np.ndarray) -> float:
        dx = a[0] - b[0]
        dz = a[2] - b[2]
        return float(np.sqrt(dx * dx + dz * dz))
