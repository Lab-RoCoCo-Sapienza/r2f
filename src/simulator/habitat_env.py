"""
Thin wrapper around ``habitat_sim.Simulator``.

Responsibilities
----------------
* Load a .glb scene + .navmesh and configure RGB / depth sensors.
* Expose a clean ``reset() -> Observation`` / ``step(action) -> Observation``
  interface with standard numpy outputs.
* Convert Habitat's internal quaternion (wxyz) and coordinate conventions into
  the 4x4 SE(3) pose matrices and xyzw quaternions used by the rest of the
  codebase.
* Optionally manage a second agent (agent index 1) carrying a static
  overhead camera for bird's-eye-view visualization.

Coordinate conventions
----------------------
Habitat-sim uses a *right-handed, Y-up* frame:
    +X = right,  +Y = up,  +Z = backward  (camera looks along -Z).

All poses returned by this wrapper are 4x4 homogeneous matrices representing
the *camera-to-world* transform.
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

os.environ.setdefault("MAGNUM_LOG", "quiet")
os.environ.setdefault("HABITAT_SIM_LOG", "quiet")
logging.getLogger("habitat_sim").setLevel(logging.ERROR)

import magnum as mn
import habitat_sim
from habitat_sim.agent import ActionSpec, ActuationSpec, AgentConfiguration

from .configs import SimulatorConfig
from .observation import Observation


# Actions exposed to the rest of the system
ACTIONS: List[str] = ["move_forward", "turn_left", "turn_right", "stop"]


class HabitatEnv:
    """
    Lightweight Habitat-sim environment.

    Parameters
    ----------
    cfg : SimulatorConfig
        All tunables live here so the class stays config-driven.
    """

    def __init__(self, cfg: SimulatorConfig) -> None:
        self._cfg = cfg
        self._step_count: int = 0
        self._bev_enabled: bool = cfg.bev.enabled
        self._overhead_cam_params: Optional[dict] = None

        # Validate paths
        if not cfg.scene_path:
            raise ValueError("SimulatorConfig.scene_path must be set.")
        if not os.path.isfile(cfg.scene_path):
            raise FileNotFoundError(f"Scene not found: {cfg.scene_path}")

        # Auto-derive navmesh path if not provided
        if not cfg.navmesh_path:
            cfg.navmesh_path = str(
                Path(cfg.scene_path).with_suffix(".navmesh")
            )

        # Build habitat-sim configuration
        sim_cfg = self._make_sim_config()
        agent_cfgs = [self._make_agent_config()]

        # If BEV is enabled, add a second agent for the overhead camera
        if self._bev_enabled:
            agent_cfgs.append(self._make_overhead_agent_config())

        self._hab_cfg = habitat_sim.Configuration(sim_cfg, agent_cfgs)

        # Create simulator
        self._sim = habitat_sim.Simulator(self._hab_cfg)

        # Load navmesh (needed for collision and spawning)
        if os.path.isfile(cfg.navmesh_path):
            self._sim.pathfinder.load_nav_mesh(cfg.navmesh_path)
        else:
            print(f"[HabitatEnv] Warning: navmesh not found at "
                  f"{cfg.navmesh_path}, navigation queries will be limited.")

        # Position overhead agent above the scene
        if self._bev_enabled:
            self._position_overhead_agent()

        # Precompute intrinsic matrix (constant across episode)
        self._K = self._build_intrinsics()


    def reset(self) -> Observation:
        """Reset the simulator and return the first observation."""
        self._sim.reset()
        self._step_count = 0

        # sim.reset() resets ALL agents, so re-position the overhead camera
        if self._bev_enabled:
            self._position_overhead_agent()

        return self._get_observation()

    def step(self, action: str) -> Observation:
        """
        Execute a discrete action and return the resulting observation.

        Parameters
        ----------
        action : str
            One of ``ACTIONS`` (move_forward, turn_left, turn_right, stop).

        Returns
        -------
        Observation
        """
        if action not in ACTIONS:
            raise ValueError(
                f"Unknown action '{action}'. Choose from {ACTIONS}."
            )

        if action == "stop":
            pass
        else:
            self._sim.step(action)

        self._step_count += 1
        return self._get_observation()

    def get_agent_pose(self) -> np.ndarray:
        """Return the current agent body pose as a 4x4 matrix (world frame)."""
        state = self._sim.get_agent(0).get_state()
        return self._agent_state_to_pose(state)

    def get_sensor_pose(self) -> np.ndarray:
        """Return the current *sensor* (camera) pose as a 4x4 matrix."""
        state = self._sim.get_agent(0).get_state()
        return self._sensor_state_to_pose(state)

    def set_agent_pose(
        self,
        position: np.ndarray,
        rotation_xyzw: np.ndarray,
    ) -> Observation:
        """
        Teleport the agent to an arbitrary pose and return the observation.

        Parameters
        ----------
        position : (3,) array -- [x, y, z] in world frame.
        rotation_xyzw : (4,) array -- unit quaternion in scipy (xyzw) order.
        """
        agent = self._sim.get_agent(0)
        state = agent.get_state()

        state.position = np.asarray(position, dtype=np.float32)
        state.rotation = self._xyzw_to_habitat_quat(rotation_xyzw)

        agent.set_state(state)
        return self._get_observation()

    @property
    def intrinsics(self) -> np.ndarray:
        """Camera intrinsic matrix K (3x3)."""
        return self._K.copy()

    @property
    def image_size(self) -> tuple:
        """(height, width) of the sensor images."""
        return (self._cfg.sensor.height, self._cfg.sensor.width)

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def overhead_cam_params(self) -> Optional[dict]:
        """
        World-to-pixel mapping parameters for the overhead image.

        Returns None if BEV is disabled. Keys:
            center_x, center_z  -- scene center in world coords
            extent_x, extent_z  -- visible width/depth in metres
            img_w, img_h        -- overhead image size in pixels
        """
        return self._overhead_cam_params

    def sample_navigable_point(self) -> np.ndarray:
        """
        Sample a random navigable position from the navmesh.

        This is an *environment* operation (used for spawning / evaluation),
        NOT agent knowledge.  The agent's own planning must remain map-free.
        """
        return np.asarray(
            self._sim.pathfinder.get_random_navigable_point(),
            dtype=np.float64,
        )

    @property
    def sim(self) -> "habitat_sim.Simulator":
        """Direct access to the underlying Habitat simulator (for evaluation / GT only)."""
        return self._sim

    @property
    def navigable_area(self) -> float:
        """Total navigable area in m^2 (for logging / evaluation only)."""
        return self._sim.pathfinder.navigable_area

    @property
    def scene_bounds(self) -> tuple:
        """(lower, upper) AABB of the navigable scene, each a (3,) array."""
        b = self._sim.pathfinder.get_bounds()
        return (np.array(b[0], dtype=np.float64),
                np.array(b[1], dtype=np.float64))

    def find_path(
        self,
        start: np.ndarray,
        goal: np.ndarray,
    ) -> Optional[List[np.ndarray]]:
        """Find a shortest path on the navmesh between two 3D points.

        Snaps both endpoints to the navmesh before planning so that
        off-mesh queries (e.g. frontier positions from depth) still work.

        Returns a list of (3,) waypoints if a path exists, None otherwise.
        """
        from habitat_sim.nav import ShortestPath
        import magnum as mn

        pf = self._sim.pathfinder

        # Snap endpoints to nearest navmesh surface
        snapped_start = pf.snap_point(mn.Vector3(*start.tolist()))
        snapped_goal = pf.snap_point(mn.Vector3(*goal.tolist()))
        if not np.isfinite(snapped_start).all() or not np.isfinite(snapped_goal).all():
            return None

        path_req = ShortestPath()
        path_req.requested_start = snapped_start
        path_req.requested_end = snapped_goal

        if not pf.find_path(path_req):
            return None
        if len(path_req.points) < 2:
            return None

        return [np.array(p, dtype=np.float64) for p in path_req.points]

    def geodesic_distance(self, start: np.ndarray, goal: np.ndarray) -> Optional[float]:
        """Sum of 3D segment lengths along the shortest navmesh path.

        Returns None if no path exists (caller should fall back to Euclidean).
        """
        path = self.find_path(start, goal)
        if path is None or len(path) < 2:
            return None
        pts = np.array(path, dtype=np.float64)
        return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())

    def snap_point(self, pos3d: np.ndarray) -> np.ndarray:
        """Snap a 3D position to the nearest navmesh point. Returns (3,) float64."""
        import magnum as mn
        snapped = self._sim.pathfinder.snap_point(mn.Vector3(*pos3d.tolist()))
        return np.array(snapped, dtype=np.float64)

    def close(self) -> None:
        """Release simulator resources."""
        self._sim.close()

    # Private helpers

    def _make_sim_config(self) -> habitat_sim.SimulatorConfiguration:
        """Translate our dataclass config into Habitat's SimulatorConfiguration."""
        sc = habitat_sim.SimulatorConfiguration()
        sc.scene_id = self._cfg.scene_path
        sc.gpu_device_id = self._cfg.gpu_device_id
        sc.enable_physics = self._cfg.enable_physics
        sc.allow_sliding = self._cfg.allow_sliding
        sc.random_seed = self._cfg.random_seed

        # Auto-discover HM3D scene dataset config for semantic annotations
        # Configs live in data/hm3d/, scene is in data/hm3d/val/00824-.../
        hm3d_root = Path(self._cfg.scene_path).parent.parent.parent
        for candidate in hm3d_root.glob("hm3d_annotated*scene_dataset_config.json"):
            sc.scene_dataset_config_file = str(candidate)
            break

        return sc

    def _make_agent_config(self) -> AgentConfiguration:
        """Build agent 0 config with RGB + depth sensors and discrete actions."""
        s = self._cfg.sensor

        # RGB sensor
        rgb_spec = habitat_sim.CameraSensorSpec()
        rgb_spec.uuid = "color"
        rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
        rgb_spec.resolution = [s.height, s.width]
        rgb_spec.hfov = s.hfov
        rgb_spec.position = np.array([0.0, s.sensor_height, 0.0])
        rgb_spec.clear_color = mn.Color4(1.0, 1.0, 1.0, 1.0)

        # Depth sensor
        depth_spec = habitat_sim.CameraSensorSpec()
        depth_spec.uuid = "depth"
        depth_spec.sensor_type = habitat_sim.SensorType.DEPTH
        depth_spec.resolution = [s.height, s.width]
        depth_spec.hfov = s.hfov
        depth_spec.position = np.array([0.0, s.sensor_height, 0.0])

        # Action space
        a = self._cfg.action
        action_space: Dict[str, ActionSpec] = {
            "move_forward": ActionSpec(
                "move_forward", ActuationSpec(amount=a.forward_amount)
            ),
            "turn_left": ActionSpec(
                "turn_left", ActuationSpec(amount=a.turn_amount)
            ),
            "turn_right": ActionSpec(
                "turn_right", ActuationSpec(amount=a.turn_amount)
            ),
        }

        # Agent
        agent_cfg = AgentConfiguration()
        agent_cfg.height = self._cfg.agent_height
        agent_cfg.radius = self._cfg.agent_radius
        agent_cfg.sensor_specifications = [rgb_spec, depth_spec]
        agent_cfg.action_space = action_space

        return agent_cfg

    def _make_overhead_agent_config(self) -> AgentConfiguration:
        """
        Build agent 1 config: a single downward-facing RGB sensor.

        The sensor orientation is pitched -90 deg so it looks straight
        down along -Y. The agent itself is positioned above the scene
        by _position_overhead_agent() after the simulator is created.
        """
        b = self._cfg.bev

        import magnum as mn

        rgb_spec = habitat_sim.CameraSensorSpec()
        rgb_spec.uuid = "overhead"
        rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
        rgb_spec.resolution = [b.overhead_height, b.overhead_width]
        rgb_spec.hfov = b.overhead_hfov
        # White background where no geometry is visible
        rgb_spec.clear_color = mn.Color4(1.0, 1.0, 1.0, 1.0)
        # Sensor at agent origin; actual world position set via agent state
        rgb_spec.position = np.array([0.0, 0.0, 0.0])
        # Pitch -90 deg to look straight down (-Y)
        rgb_spec.orientation = np.array([-np.pi / 2.0, 0.0, 0.0])

        agent_cfg = AgentConfiguration()
        agent_cfg.sensor_specifications = [rgb_spec]
        agent_cfg.action_space = {}  # overhead agent never steps
        return agent_cfg

    def _position_overhead_agent(self) -> None:
        """
        Place the overhead agent (agent 1) above the scene center so the
        full navigable area fits within the image.

        Computes camera height from scene bounds and the configured hfov,
        and stores the world-to-pixel mapping parameters used by BEVRenderer.

        The camera is placed high enough that the scene fits in the image,
        and the stored extents reflect the *actual* visible area at the
        ground plane (which may be larger than the scene itself if the
        camera must clear the ceiling).
        """
        import quaternion as qt

        bounds = self._sim.pathfinder.get_bounds()
        lower = np.array(bounds[0], dtype=np.float64)
        upper = np.array(bounds[1], dtype=np.float64)

        margin = self._cfg.bev.overhead_margin
        center = (lower + upper) / 2.0
        scene_extent_x = (upper[0] - lower[0]) + 2.0 * margin
        scene_extent_z = (upper[2] - lower[2]) + 2.0 * margin
        max_scene_extent = max(scene_extent_x, scene_extent_z)

        hfov_rad = np.deg2rad(self._cfg.bev.overhead_hfov)
        tan_half_hfov = np.tan(hfov_rad / 2.0)

        # Minimum distance from camera to ground so the scene fits in hfov:
        #   visible_width = 2 * distance * tan(hfov / 2)
        #   distance = max_scene_extent / (2 * tan(hfov / 2))
        min_distance = max_scene_extent / (2.0 * tan_half_hfov)

        # Camera must be at least above the ceiling (upper[1]) plus a
        # small clearance, but also far enough to fit the full scene.
        ground_y = lower[1]
        cam_y = max(ground_y + min_distance, upper[1] + 0.5)

        # Actual distance from camera to ground determines true visible area
        actual_distance = cam_y - ground_y

        # Visible extents at the ground plane
        # Horizontal (X) extent is governed by hfov
        visible_extent_x = 2.0 * actual_distance * tan_half_hfov
        # Vertical (Z) extent: for square images vfov = hfov; for non-square
        # images compute vfov from aspect ratio
        img_w = self._cfg.bev.overhead_width
        img_h = self._cfg.bev.overhead_height
        tan_half_vfov = (img_h / img_w) * tan_half_hfov
        visible_extent_z = 2.0 * actual_distance * tan_half_vfov

        # Set agent 1 state
        overhead_agent = self._sim.get_agent(1)
        state = overhead_agent.get_state()
        state.position = np.array(
            [center[0], cam_y, center[2]], dtype=np.float32
        )
        state.rotation = qt.quaternion(1, 0, 0, 0)  # identity
        overhead_agent.set_state(state)

        # Store mapping params for BEVRenderer
        self._overhead_cam_params = {
            "center_x": center[0],
            "center_z": center[2],
            "extent_x": visible_extent_x,
            "extent_z": visible_extent_z,
            "img_w": img_w,
            "img_h": img_h,
            "cam_y": cam_y,
            "ground_y": ground_y,
        }

    def _build_intrinsics(self) -> np.ndarray:
        """
        Compute the 3x3 camera intrinsic matrix from the horizontal FoV
        and image resolution.

        Habitat uses a symmetric pinhole model with square pixels, so
        fx == fy.
        """
        W = self._cfg.sensor.width
        H = self._cfg.sensor.height
        hfov_rad = np.deg2rad(self._cfg.sensor.hfov)

        fx = W / (2.0 * np.tan(hfov_rad / 2.0))
        fy = fx  # square pixels
        cx = W / 2.0
        cy = H / 2.0

        K = np.array([
            [fx,  0.0, cx],
            [0.0, fy,  cy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        return K

    # Observation assembly

    def _get_observation(self) -> Observation:
        """Package the current simulator state into an Observation."""
        raw = self._sim.get_sensor_observations(0)

        # RGB: habitat returns RGBA (H,W,4) uint8 -> drop alpha channel
        rgb = raw["color"][:, :, :3]

        # Depth: (H,W) float32, in metres; clip far values
        depth = raw["depth"].copy()
        depth = np.clip(depth, 0.0, self._cfg.depth_clip_far)

        # Camera-to-world SE(3) pose
        pose = self.get_sensor_pose()

        # Agent body pose (Y at floor level, not sensor height)
        body_pose = self.get_agent_pose()

        # Overhead image from agent 1 (if BEV is enabled)
        overhead_rgb = None
        if self._bev_enabled:
            overhead_raw = self._sim.get_sensor_observations(1)
            overhead_rgb = overhead_raw["overhead"][:, :, :3]

        return Observation(
            rgb=rgb,
            depth=depth,
            pose=pose,
            intrinsics=self._K.copy(),
            step_id=self._step_count,
            body_pose=body_pose,
            overhead_rgb=overhead_rgb,
        )

    # Pose conversion helpers

    @staticmethod
    def _habitat_quat_to_xyzw(hq) -> np.ndarray:
        """
        Convert a Habitat / Magnum quaternion (wxyz attribute order)
        to scipy-convention (xyzw) numpy array.
        """
        return np.array([hq.x, hq.y, hq.z, hq.w], dtype=np.float64)

    @staticmethod
    def _xyzw_to_habitat_quat(q_xyzw: np.ndarray):
        """Convert scipy xyzw quaternion to Habitat quaternion object."""
        import quaternion as qt
        return qt.quaternion(q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2])

    @staticmethod
    def _quat_xyzw_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
        """Convert an xyzw quaternion to a 3x3 rotation matrix."""
        from scipy.spatial.transform import Rotation as R
        return R.from_quat(q).as_matrix()

    def _agent_state_to_pose(self, state) -> np.ndarray:
        """Build a 4x4 world-frame pose from an AgentState."""
        q_xyzw = self._habitat_quat_to_xyzw(state.rotation)
        rot = self._quat_xyzw_to_rotation_matrix(q_xyzw)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = rot
        T[:3, 3] = state.position.astype(np.float64)
        return T

    def _sensor_state_to_pose(self, state) -> np.ndarray:
        """
        Build a 4x4 world-frame pose from the first visual sensor.

        The sensor state includes the height offset baked in by Habitat,
        so this gives the actual camera-to-world transform.
        """
        # Pick the first sensor (color and depth are co-located)
        sensor_key = next(iter(state.sensor_states))
        ss = state.sensor_states[sensor_key]

        q_xyzw = self._habitat_quat_to_xyzw(ss.rotation)
        rot = self._quat_xyzw_to_rotation_matrix(q_xyzw)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = rot
        T[:3, 3] = np.asarray(ss.position, dtype=np.float64)
        return T
