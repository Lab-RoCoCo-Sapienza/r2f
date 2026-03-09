"""
Interactive bird's-eye-view viewer using Open3D (Mode 2).

Runs in a background daemon thread so it does not block the main
exploration loop. The main thread communicates with the viewer via a
thread-safe command queue -- all Open3D geometry operations happen
inside the viewer thread.

Reuses functions from vis_utils.py:
    load_mesh, create_camera, create_interactive_vis,
    register_basic_callbacks, create_cylinder_between_points
"""

from __future__ import annotations

import queue
import threading
import time
from typing import List, Optional, Tuple

import numpy as np

from .configs import BEVConfig


class BEVViewer:
    """
    Interactive Open3D 3D viewer for bird's-eye-view visualization.

    Usage
    -----
        viewer = BEVViewer(cfg, scene_mesh_path, scene_bounds)
        viewer.start()

        # each step in the main loop:
        viewer.update_agent(position, heading_rad)
        viewer.add_trajectory_point(position)

        # on shutdown:
        viewer.stop()
    """

    def __init__(
        self,
        cfg: BEVConfig,
        scene_mesh_path: str,
        scene_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> None:
        self._cfg = cfg
        self._scene_mesh_path = scene_mesh_path
        self._scene_bounds = scene_bounds

        self._cmd_queue: queue.Queue = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self) -> None:
        """Launch the viewer in a background daemon thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run_viewer_loop,
            daemon=True,
            name="BEVViewer",
        )
        self._thread.start()

    def stop(self) -> None:
        """Signal shutdown and wait for the thread to finish."""
        self._cmd_queue.put(("shutdown",))
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    def update_agent(self, position: np.ndarray, heading_rad: float) -> None:
        """Update the agent marker position and heading."""
        self._cmd_queue.put(("update_agent", position.copy(), heading_rad))

    def add_trajectory_point(self, position: np.ndarray) -> None:
        """Append a point to the trajectory visualization."""
        self._cmd_queue.put(("add_trajectory_point", position.copy()))

    def reset(self) -> None:
        """Clear trajectory and markers (call on env.reset())."""
        self._cmd_queue.put(("reset",))

    def add_markers(
        self,
        markers: List[Tuple[np.ndarray, Tuple[float, float, float], float]],
    ) -> None:
        """
        Add temporary markers (e.g. frontiers).

        Each marker is (position_3d, color_rgb_normalised, radius_m).
        Previous markers are cleared on each call.
        """
        self._cmd_queue.put(("add_markers", markers))

    # Viewer thread

    def _run_viewer_loop(self) -> None:
        """
        Main loop for the Open3D viewer thread.

        Loads the scene mesh, creates the window with a top-down camera,
        then enters a poll loop that drains the command queue and
        refreshes the renderer.
        """
        # Lazy-import Open3D so the rest of the codebase does not
        # depend on it unless the viewer is actually started.
        import open3d as o3d
        import sys, os
        sys.path.insert(0, os.path.join(
            os.path.dirname(__file__), os.pardir, os.pardir
        ))
        from src.utils.vis_utils import (
            load_mesh,
            create_camera,
            create_interactive_vis,
            register_basic_callbacks,
            create_cylinder_between_points,
        )

        # Load scene mesh
        mesh = load_mesh(self._scene_mesh_path)

        # Open3D loads .glb without applying glTF's Y-up scene transform.
        # Habitat-sim uses Y-up internally, so we must rotate the mesh:
        #   Habitat X =  Open3D X
        #   Habitat Y =  Open3D Z
        #   Habitat Z = -Open3D Y
        # This is a -90 deg rotation about the X axis.
        R_fix = np.array([
            [1.0,  0.0,  0.0],
            [0.0,  0.0,  1.0],
            [0.0, -1.0,  0.0],
        ])
        mesh.rotate(R_fix, center=(0, 0, 0))

        # Create viewer window
        cam_intr = create_camera(
            self._cfg.viewer_height,
            self._cfg.viewer_width,
            focal=self._cfg.viewer_width / 2.0,
        )
        top_down_extrinsic = self._compute_top_down_extrinsic()

        vis = create_interactive_vis(
            H=self._cfg.viewer_height,
            W=self._cfg.viewer_width,
            camera_intrinsic=cam_intr,
            camera_extrinsic=top_down_extrinsic,
            show_back_face=True,
            light_on=True,
            z_far=100.0,
        )
        register_basic_callbacks(vis)
        vis.add_geometry(mesh)

        # Mutable viewer state
        agent_sphere: Optional[o3d.geometry.TriangleMesh] = None
        agent_arrow: Optional[o3d.geometry.TriangleMesh] = None
        trajectory_geoms: List[o3d.geometry.TriangleMesh] = []
        trajectory_pts: List[np.ndarray] = []
        marker_geoms: List[o3d.geometry.TriangleMesh] = []

        # Create initial agent sphere + arrow
        agent_sphere, agent_arrow = self._create_agent_geoms(o3d)
        vis.add_geometry(agent_sphere)
        vis.add_geometry(agent_arrow)

        # Event loop
        poll_interval = 1.0 / max(self._cfg.viewer_fps, 1.0)

        while self._running:
            # Drain command queue
            while not self._cmd_queue.empty():
                try:
                    cmd = self._cmd_queue.get_nowait()
                except queue.Empty:
                    break

                name = cmd[0]

                if name == "shutdown":
                    self._running = False
                    break

                elif name == "update_agent":
                    pos, heading = cmd[1], cmd[2]
                    # Remove old agent geometries
                    vis.remove_geometry(agent_sphere, reset_bounding_box=False)
                    vis.remove_geometry(agent_arrow, reset_bounding_box=False)
                    # Recreate at new pose
                    agent_sphere, agent_arrow = self._create_agent_geoms(
                        o3d, position=pos, heading=heading,
                    )
                    vis.add_geometry(agent_sphere, reset_bounding_box=False)
                    vis.add_geometry(agent_arrow, reset_bounding_box=False)

                elif name == "add_trajectory_point":
                    pos = cmd[1]
                    trajectory_pts.append(pos)
                    if len(trajectory_pts) >= 2:
                        prev = trajectory_pts[-2]
                        curr = trajectory_pts[-1]
                        cyl = create_cylinder_between_points(
                            prev, curr,
                            radius=self._cfg.trajectory_line_radius,
                            color=(0.0, 1.0, 0.0),
                        )
                        if cyl is not None:
                            trajectory_geoms.append(cyl)
                            vis.add_geometry(cyl, reset_bounding_box=False)

                elif name == "reset":
                    for g in trajectory_geoms:
                        vis.remove_geometry(g, reset_bounding_box=False)
                    trajectory_geoms.clear()
                    trajectory_pts.clear()
                    for g in marker_geoms:
                        vis.remove_geometry(g, reset_bounding_box=False)
                    marker_geoms.clear()

                elif name == "add_markers":
                    # Clear old markers first
                    for g in marker_geoms:
                        vis.remove_geometry(g, reset_bounding_box=False)
                    marker_geoms.clear()
                    for pos, color, radius in cmd[1]:
                        sph = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
                        sph.paint_uniform_color(color)
                        sph.compute_vertex_normals()
                        sph.translate(pos)
                        marker_geoms.append(sph)
                        vis.add_geometry(sph, reset_bounding_box=False)

            # Poll Open3D events
            if not vis.poll_events():
                break
            vis.update_renderer()
            time.sleep(poll_interval)

        vis.destroy_window()

    # Geometry helpers
    def _create_agent_geoms(self, o3d, position=None, heading=0.0):
        """
        Create a sphere + heading arrow for the agent.

        If position is None, places them at the origin.
        """
        color = self._cfg.agent_color_normalized

        # Sphere
        sphere = o3d.geometry.TriangleMesh.create_sphere(
            radius=self._cfg.agent_sphere_radius,
        )
        sphere.paint_uniform_color(color)
        sphere.compute_vertex_normals()

        # Arrow (default orientation: along +Z)
        arrow = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=self._cfg.arrow_radius,
            cone_radius=self._cfg.arrow_radius * 2.0,
            cylinder_height=self._cfg.arrow_length * 0.7,
            cone_height=self._cfg.arrow_length * 0.3,
        )
        arrow.paint_uniform_color(color)
        arrow.compute_vertex_normals()

        # Rotate arrow to point along heading direction.
        # Default Open3D arrow is along +Z.
        # In Y-up Habitat: heading 0 = facing -Z, positive = CCW from above.
        # Rotate by (heading + pi) about Y to go from +Z to the desired dir.
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(
            np.array([0.0, heading + np.pi, 0.0])
        )
        arrow.rotate(R, center=(0, 0, 0))

        if position is not None:
            sphere.translate(position)
            arrow.translate(position)

        return sphere, arrow

    def _compute_top_down_extrinsic(self) -> np.ndarray:
        """
        Compute a 4x4 world-to-camera extrinsic matrix for a top-down view.

        The camera is placed above the scene center, looking down along -Y.
        """
        if self._scene_bounds is not None:
            lower, upper = self._scene_bounds
        else:
            lower = np.array([-5.0, -1.0, -5.0])
            upper = np.array([5.0, 3.0, 5.0])

        center = (lower + upper) / 2.0
        height = max(upper[0] - lower[0], upper[2] - lower[2]) * 1.2

        cam_pos = np.array([center[0], upper[1] + height, center[2]])

        # Camera-to-world rotation:
        #   camera right  = world +X
        #   camera down   = world +Z
        #   camera fwd    = world -Y  (looking down)
        R_cam2world = np.array([
            [1.0,  0.0,  0.0],
            [0.0,  0.0,  1.0],
            [0.0, -1.0,  0.0],
        ])

        T_cam2world = np.eye(4)
        T_cam2world[:3, :3] = R_cam2world
        T_cam2world[:3, 3] = cam_pos

        # Open3D extrinsic is world-to-camera
        return np.linalg.inv(T_cam2world)
