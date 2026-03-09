"""
Live Open3D viewer for semantic frontier exploration.

Provides Viewer (threaded Open3D window), NullViewer (headless no-op),
and collect_region_data (frontier coloring helper).
"""

from __future__ import annotations

import queue
import threading
import time
from pathlib import Path
from typing import Optional

import matplotlib.cm as cm
import numpy as np

# GL (OpenGL/Habitat) to CV (OpenCV) frame conversion
_GL_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])


def collect_region_data(region_map, text_emb: np.ndarray, arrow_len: float = 1.5):
    """
    Build colored sphere and arrow lists from a FrontierRegionMap.

    Returns (member_spheres, arrows, scored) where:
      member_spheres : list of (pos, rgb) -- one per frontier voxel
      arrows         : list of (start, end, rgb) -- one per scored region
      scored         : raw output of region_map.score_all()
    """
    scored = region_map.score_all(text_emb)

    if not scored:
        member_spheres = []
        for reg in region_map.active_regions:
            for pt in reg.member_pts:
                member_spheres.append((pt, [0.55, 0.55, 0.55]))
        return member_spheres, [], []

    scores = [s[2] for s in scored]
    s_min, s_max = min(scores), max(scores)
    spread = max(s_max - s_min, 1e-6)

    region_color = {}
    for centroid, direction, score, rid in scored:
        t = float((score - s_min) / spread)
        region_color[rid] = list(cm.turbo(t)[:3])

    scored_ids = {s[3] for s in scored}
    for reg in region_map.active_regions:
        if reg.id not in scored_ids:
            region_color[reg.id] = [0.55, 0.55, 0.55]

    member_spheres = []
    for reg in region_map.active_regions:
        col = region_color.get(reg.id, [0.55, 0.55, 0.55])
        for pt in reg.member_pts:
            member_spheres.append((pt, col))

    arrows = []
    for centroid, direction, score, rid in scored:
        col = region_color[rid]
        arrows.append((centroid.copy(), centroid + arrow_len * direction, col))

    return member_spheres, arrows, scored


class NullViewer:
    """No-op viewer for headless / batch evaluation runs."""

    def __init__(self, **kwargs):
        self.nav_queue: queue.Queue = queue.Queue()

    def start(self) -> None: pass
    def stop(self) -> None: pass
    def update(self, *args, **kwargs) -> None: pass


class Viewer:
    """Live Open3D window: occupancy map + frontier regions + exploration arrows."""

    def __init__(
        self,
        width: int = 1400,
        height: int = 900,
        intrinsics: np.ndarray = None,
        img_w: int = 640,
        img_h: int = 480,
        scene_path: str = "",
        nav_queue: queue.Queue = None,
        gt_goal_pos: Optional[np.ndarray] = None,
        gt_goal_positions: Optional[list] = None,
        gt_instance_positions: Optional[list] = None,
        overhead_cam_params: Optional[dict] = None,
        overhead_rgb: Optional[np.ndarray] = None,
    ) -> None:
        self._width = width
        self._height = height
        self._intrinsics = intrinsics if intrinsics is not None else np.eye(3)
        self._img_w = img_w
        self._img_h = img_h
        self._scene_path = scene_path
        self._nav_queue = nav_queue
        # Use object instance centroids for bbox visualization if available,
        # otherwise fall back to viewpoints / single goal pos
        if gt_instance_positions:
            self._gt_goal_positions = gt_instance_positions
        elif gt_goal_positions:
            self._gt_goal_positions = gt_goal_positions
        elif gt_goal_pos is not None:
            self._gt_goal_positions = [gt_goal_pos]
        else:
            self._gt_goal_positions = []
        self._overhead_cam = overhead_cam_params
        self._overhead_rgb = overhead_rgb
        self._cmd_queue: queue.Queue = queue.Queue()
        self._thread = None
        self._running = False

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="Viewer")
        self._thread.start()

    def stop(self) -> None:
        self._cmd_queue.put(("shutdown",))
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    def update(
        self,
        occ_pts,
        free_pts,
        member_spheres,
        arrows,
        camera_pose=None,
        goal_pos=None,
    ) -> None:
        self._cmd_queue.put((
            "update",
            occ_pts.copy() if occ_pts is not None else None,
            free_pts.copy() if free_pts is not None else None,
            list(member_spheres),
            list(arrows),
            camera_pose.copy() if camera_pose is not None else None,
            goal_pos.copy() if goal_pos is not None else None,
        ))

    def _make_frustum(self, o3d, pose_gl, scale: float = 1.5):
        fx = self._intrinsics[0, 0]
        fy = self._intrinsics[1, 1]
        cx = self._intrinsics[0, 2]
        cy = self._intrinsics[1, 2]
        W, H, d = self._img_w, self._img_h, scale
        corners_cv = np.array([
            [(0 - cx) / fx * d, (0 - cy) / fy * d, d],
            [(W - cx) / fx * d, (0 - cy) / fy * d, d],
            [(W - cx) / fx * d, (H - cy) / fy * d, d],
            [(0 - cx) / fx * d, (H - cy) / fy * d, d],
        ])
        pose_cv = pose_gl @ _GL_TO_CV
        R, t = pose_cv[:3, :3], pose_cv[:3, 3]
        corners_w = (R @ corners_cv.T).T + t
        pts = np.vstack([t[None, :], corners_w])
        lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]
        frustum = o3d.geometry.LineSet()
        frustum.points = o3d.utility.Vector3dVector(pts)
        frustum.lines = o3d.utility.Vector2iVector(lines)
        frustum.colors = o3d.utility.Vector3dVector([[1.0, 0.9, 0.0]] * len(lines))
        return frustum

    @staticmethod
    def _make_goal_bbox(o3d, center: np.ndarray, half: float = 0.6):
        x, y, z = center
        corners = [
            [x - half, y - half, z - half], [x + half, y - half, z - half],
            [x + half, y + half, z - half], [x - half, y + half, z - half],
            [x - half, y - half, z + half], [x + half, y - half, z + half],
            [x + half, y + half, z + half], [x - half, y + half, z + half],
        ]
        lines = [
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7],
        ]
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(corners)
        ls.lines = o3d.utility.Vector2iVector(lines)
        ls.colors = o3d.utility.Vector3dVector([[1.0, 0.0, 0.0]] * 12)
        return ls

    @staticmethod
    def _build_colored_sphere_mesh(o3d, member_spheres, radius: float = 0.10):
        combined = o3d.geometry.TriangleMesh()
        for pos, color in member_spheres:
            sph = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=5)
            sph.paint_uniform_color(color)
            sph.translate(pos)
            combined += sph
        if len(combined.vertices) == 0:
            return None
        combined.compute_vertex_normals()
        return combined

    @staticmethod
    def _build_arrow_mesh(o3d, arrows):
        if not arrows:
            return None
        z = np.array([0., 0., 1.])
        combined = o3d.geometry.TriangleMesh()
        for start, end, color in arrows:
            vec = end - start
            length = float(np.linalg.norm(vec))
            if length < 1e-6:
                continue
            direction = vec / length
            cone_h = min(0.35, length * 0.35)
            cyl_h  = max(length - cone_h, 0.05)
            mesh = o3d.geometry.TriangleMesh.create_arrow(
                cylinder_radius=0.07, cone_radius=0.16,
                cylinder_height=cyl_h, cone_height=cone_h, resolution=10,
            )
            mesh.paint_uniform_color(color)
            axis = np.cross(z, direction)
            axis_norm = float(np.linalg.norm(axis))
            if axis_norm > 1e-6:
                axis /= axis_norm
                angle = float(np.arccos(np.clip(np.dot(z, direction), -1., 1.)))
                K = np.array([[0, -axis[2], axis[1]],
                               [axis[2], 0, -axis[0]],
                               [-axis[1], axis[0], 0]])
                R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
                mesh.rotate(R, center=(0, 0, 0))
            elif np.dot(z, direction) < 0:
                mesh.rotate(np.diag([1., -1., -1.]), center=(0, 0, 0))
            mesh.translate(start)
            combined += mesh
        if len(combined.vertices) == 0:
            return None
        combined.compute_vertex_normals()
        return combined

    def _color_pcd_from_bev(self, o3d, pcd) -> None:
        """Project each point to the overhead BEV camera and sample its RGB color."""
        if self._overhead_cam is None or self._overhead_rgb is None:
            return
        cam = self._overhead_cam
        rgb = self._overhead_rgb  # [H, W, 3] uint8
        H, W = rgb.shape[:2]
        pts = np.asarray(pcd.points)  # [N, 3]

        dist_ground = cam["cam_y"] - cam["ground_y"]
        dist_point  = np.maximum(cam["cam_y"] - pts[:, 1], 0.01)
        scale = dist_ground / dist_point

        dx   = (pts[:, 0] - cam["center_x"]) * scale
        dz   = (pts[:, 2] - cam["center_z"]) * scale
        cols = np.clip(((dx + cam["extent_x"] / 2) / cam["extent_x"] * W).astype(int), 0, W - 1)
        rows = np.clip(((dz + cam["extent_z"] / 2) / cam["extent_z"] * H).astype(int), 0, H - 1)

        pcd.colors = o3d.utility.Vector3dVector(rgb[rows, cols].astype(np.float64) / 255.0)

    @staticmethod
    def _build_goal_sphere(o3d, pos: np.ndarray):
        sph = o3d.geometry.TriangleMesh.create_sphere(radius=0.35, resolution=8)
        sph.paint_uniform_color([0.0, 0.9, 0.9])
        sph.translate(pos)
        sph.compute_vertex_normals()
        return sph

    def _run(self) -> None:
        import open3d as o3d

        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window("Semantic Frontier Exploration", self._width, self._height)
        opt = vis.get_render_option()
        opt.point_size = 4.0
        opt.background_color = np.array([1.0, 1.0, 1.0])
        opt.light_on = True

        if self._nav_queue is not None:
            def _quit_cb(v):
                self._nav_queue.put("__quit__")
                return False
            vis.register_key_callback(ord('Q'), _quit_cb)

        occ_pcd = o3d.geometry.PointCloud()
        floor_pcd = o3d.geometry.PointCloud()
        use_colored_scene = self._overhead_cam is not None and self._overhead_rgb is not None

        coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        vis.add_geometry(coord)

        if self._scene_path and Path(self._scene_path).exists():
            glb_path = self._scene_path.replace(".basis.glb", ".glb")
            load_path = glb_path if Path(glb_path).exists() else self._scene_path
            mesh = o3d.io.read_triangle_mesh(load_path, enable_post_processing=True)
            R_fix = np.array([[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]])
            mesh.rotate(R_fix, center=(0, 0, 0))
            if use_colored_scene:
                scene_pcd = mesh.sample_points_uniformly(number_of_points=1_200_000)
                pts = np.asarray(scene_pcd.points)
                scene_pcd = scene_pcd.select_by_index(np.where(pts[:, 1] < 2.0)[0])
                self._color_pcd_from_bev(o3d, scene_pcd)
                vis.add_geometry(scene_pcd)
            else:
                mesh.paint_uniform_color([0.75, 0.72, 0.68])
                mesh.compute_vertex_normals()
                vis.add_geometry(mesh)

        for _gp in self._gt_goal_positions:
            vis.add_geometry(self._make_goal_bbox(o3d, _gp))

        frustum_geom = None
        region_mesh = None
        arrow_mesh = None
        goal_mesh = None
        base_added = False
        _bev_set = False

        while self._running:
            while True:
                try:
                    cmd = self._cmd_queue.get_nowait()
                except queue.Empty:
                    break

                if cmd[0] == "shutdown":
                    self._running = False
                    break

                elif cmd[0] == "update":
                    _, occ, free, member_spheres, arrows, pose, goal_pos = cmd

                    if not use_colored_scene:
                        if occ is not None and len(occ) > 0:
                            wall = occ[(occ[:, 1] >= 0.3) & (occ[:, 1] <= 2.5)]
                            step_s = max(1, len(wall) // 25_000)
                            occ_pcd.points = o3d.utility.Vector3dVector(wall[::step_s])
                            occ_pcd.colors = o3d.utility.Vector3dVector(
                                np.tile([0.32, 0.29, 0.26], (len(wall[::step_s]), 1))
                            )
                        if free is not None and len(free) > 0:
                            floor = free[(free[:, 1] >= -0.1) & (free[:, 1] <= 0.6)]
                            step_s = max(1, len(floor) // 25_000)
                            floor_pcd.points = o3d.utility.Vector3dVector(floor[::step_s])
                            floor_pcd.colors = o3d.utility.Vector3dVector(
                                np.tile([0.72, 0.70, 0.67], (len(floor[::step_s]), 1))
                            )

                    if region_mesh is not None:
                        vis.remove_geometry(region_mesh, reset_bounding_box=False)
                        region_mesh = None
                    if member_spheres:
                        region_mesh = self._build_colored_sphere_mesh(o3d, member_spheres)
                        if region_mesh is not None:
                            vis.add_geometry(region_mesh, reset_bounding_box=False)

                    if arrow_mesh is not None:
                        vis.remove_geometry(arrow_mesh, reset_bounding_box=False)
                        arrow_mesh = None
                    if arrows:
                        arrow_mesh = self._build_arrow_mesh(o3d, arrows)
                        if arrow_mesh is not None:
                            vis.add_geometry(arrow_mesh, reset_bounding_box=False)

                    if goal_mesh is not None:
                        vis.remove_geometry(goal_mesh, reset_bounding_box=False)
                        goal_mesh = None
                    if goal_pos is not None:
                        goal_mesh = self._build_goal_sphere(o3d, goal_pos)
                        vis.add_geometry(goal_mesh, reset_bounding_box=False)

                    if frustum_geom is not None:
                        vis.remove_geometry(frustum_geom, reset_bounding_box=False)
                        frustum_geom = None
                    if pose is not None:
                        frustum_geom = self._make_frustum(o3d, pose)
                        vis.add_geometry(frustum_geom, reset_bounding_box=False)

                    if not use_colored_scene:
                        if not base_added:
                            vis.add_geometry(occ_pcd, reset_bounding_box=False)
                            vis.add_geometry(floor_pcd, reset_bounding_box=False)
                            base_added = True
                        else:
                            vis.update_geometry(occ_pcd)
                            vis.update_geometry(floor_pcd)

            if not vis.poll_events():
                break
            vis.update_renderer()

            if not _bev_set:
                vis.reset_view_point(True)
                # Pitch down 90 degrees to reach top-down BEV.
                # rotate(x, y): angle = pi * y / window_height, so y = height/2 gives 90 deg.
                vis.get_view_control().rotate(0, self._height / 2)
                _bev_set = True

            time.sleep(1.0 / 30.0)

        vis.destroy_window()
