"""
Bird's-eye-view renderer for 2D overhead image overlays (Mode 1).

Takes the raw top-down image from the overhead habitat-sim sensor and
draws agent markers, trajectory lines, and (later) frontier/graph
annotations on top of it using OpenCV.

Coordinate mapping
------------------
The overhead camera looks down along -Y (Habitat Y-up).
In the resulting image:
    - image columns correspond to world X (left-to-right)
    - image rows correspond to world Z (top-to-bottom)
The world_to_pixel() method encodes this mapping using the scene bounds
stored in overhead_cam_params.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .configs import BEVConfig


# Helpers
def extract_heading_from_pose(pose: np.ndarray) -> float:
    """
    Extract the agent heading angle (rotation about the Y axis) from a
    4x4 camera-to-world pose matrix.

    In Habitat's Y-up convention the camera looks along -Z_camera.
    The forward direction in world frame is the negated third column of
    the rotation part of the pose.

    Returns heading in radians, where 0 = facing -Z_world and positive
    values rotate counter-clockwise when viewed from above.
    """
    forward_world = -pose[:3, 2]
    heading = np.arctan2(forward_world[0], -forward_world[2])
    return heading


# Overlay marker (generic, for future frontier / graph overlays)
@dataclass
class OverlayMarker:
    """
    A marker drawn on the overhead image.

    Fields
    ------
    position_world : (3,) array in Habitat world frame.
    color          : (R, G, B) 0-255.
    radius         : pixel radius.
    label          : optional text label drawn next to the marker.
    """
    position_world: np.ndarray
    color: Tuple[int, int, int] = (255, 255, 0)
    radius: int = 4
    label: str = ""


# BEVRenderer
class BEVRenderer:
    """
    Renders 2D BEV images with overlaid agent state.

    Usage
    -----
        renderer = BEVRenderer(bev_cfg, env.overhead_cam_params)

        # each step:
        bev_image = renderer.render(
            overhead_rgb=obs.overhead_rgb,
            agent_position=obs.pose[:3, 3],
            agent_heading_rad=extract_heading_from_pose(obs.pose),
        )
    """

    def __init__(
        self,
        cfg: BEVConfig,
        overhead_cam_params: dict,
    ) -> None:
        self._cfg = cfg
        self._cam = overhead_cam_params
        self._trajectory: List[np.ndarray] = []

    def render(
        self,
        overhead_rgb: np.ndarray,
        agent_position: np.ndarray,
        agent_heading_rad: float,
        extra_markers: Optional[List[OverlayMarker]] = None,
        agent_rgb: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Produce a BEV image with overlays.

        Parameters
        ----------
        overhead_rgb       : (H, W, 3) uint8 from the overhead sensor.
        agent_position     : (3,) world position [x, y, z].
        agent_heading_rad  : heading angle (see extract_heading_from_pose).
        extra_markers      : optional markers for frontiers / graph nodes.
        agent_rgb          : optional (H, W, 3) uint8 first-person RGB from
                             the agent's camera.  When provided, it is shown
                             as a panel on the right side of the BEV image.

        Returns
        -------
        (H_out, W_out, 3) uint8 annotated image (RGB).
        If agent_rgb is given, W_out = bev_width + scaled_agent_width.
        """
        canvas = overhead_rgb.copy()

        # Record trajectory
        self._trajectory.append(agent_position.copy())

        # Draw trajectory polyline
        self._draw_trajectory(canvas)

        # Draw agent marker with heading arrow
        self._draw_agent(canvas, agent_position, agent_heading_rad)

        # Draw extra markers (frontiers, graph nodes, etc.)
        if extra_markers:
            for marker in extra_markers:
                self._draw_marker(canvas, marker)

        # Compose agent first-person view on the right side
        if agent_rgb is not None:
            canvas = self._compose_agent_panel(canvas, agent_rgb)

        return canvas

    def reset(self) -> None:
        """Clear accumulated trajectory (call on env.reset())."""
        self._trajectory.clear()

    def world_to_pixel(self, world_pos: np.ndarray) -> Tuple[int, int]:
        """
        Convert a world (x, y, z) position to pixel (col, row) in the
        overhead image using perspective projection.

        The overhead camera is a perspective camera, so the Y coordinate
        of the point affects where it projects. Callers should pass the
        agent body position (floor level), NOT the sensor position.
        """
        cx = self._cam["center_x"]
        cz = self._cam["center_z"]
        ex = self._cam["extent_x"]
        ez = self._cam["extent_z"]
        w = self._cam["img_w"]
        h = self._cam["img_h"]
        cam_y = self._cam["cam_y"]
        ground_y = self._cam["ground_y"]

        # Distance from camera to ground plane (defines the extent scale)
        dist_ground = cam_y - ground_y
        # Distance from camera to this point's Y level
        dist_point = cam_y - world_pos[1]
        # Perspective scale
        scale = dist_ground / max(dist_point, 0.01)

        dx = (world_pos[0] - cx) * scale
        dz = (world_pos[2] - cz) * scale

        u = (dx + ex / 2.0) / ex  # X -> col
        v = (dz + ez / 2.0) / ez  # Z -> row

        col = int(np.clip(u * w, 0, w - 1))
        row = int(np.clip(v * h, 0, h - 1))
        return (col, row)

    # Drawing helpers

    def _draw_trajectory(self, canvas: np.ndarray) -> None:
        """Draw the agent trajectory as a polyline."""
        if len(self._trajectory) < 2:
            return

        pts = [self.world_to_pixel(p) for p in self._trajectory]
        pts_arr = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)

        color_bgr = self._cfg.trajectory_color[::-1]  # RGB -> BGR for cv2
        cv2.polylines(
            canvas, [pts_arr],
            isClosed=False,
            color=color_bgr,
            thickness=self._cfg.trajectory_thickness,
            lineType=cv2.LINE_AA,
        )

    def _draw_agent(
        self,
        canvas: np.ndarray,
        position: np.ndarray,
        heading_rad: float,
    ) -> None:
        """Draw the agent as a filled circle with a heading arrow."""
        col, row = self.world_to_pixel(position)
        r = self._cfg.agent_marker_radius
        color_bgr = self._cfg.agent_color[::-1]

        # Filled circle
        cv2.circle(canvas, (col, row), r, color_bgr, -1, cv2.LINE_AA)

        # Heading arrow.
        # heading_rad: 0 = facing -Z_world (upward in image = row-).
        # sin(heading) gives the X component, -cos(heading) the Z component.
        arrow_len = r * 3
        dx = np.sin(heading_rad) * arrow_len
        dz = -np.cos(heading_rad) * arrow_len
        end_col = int(col + dx)
        end_row = int(row + dz)

        cv2.arrowedLine(
            canvas, (col, row), (end_col, end_row),
            color_bgr, thickness=2, tipLength=0.3, line_type=cv2.LINE_AA,
        )

    def _draw_marker(self, canvas: np.ndarray, marker: OverlayMarker) -> None:
        """Draw a generic overlay marker."""
        col, row = self.world_to_pixel(marker.position_world)
        color_bgr = marker.color[::-1]
        cv2.circle(canvas, (col, row), marker.radius, color_bgr, -1, cv2.LINE_AA)

        if marker.label:
            cv2.putText(
                canvas, marker.label,
                (col + marker.radius + 2, row + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_bgr, 1, cv2.LINE_AA,
            )

    def draw_line(
        self,
        canvas: np.ndarray,
        pos_a: np.ndarray,
        pos_b: np.ndarray,
        color: Tuple[int, int, int],
        thickness: int = 1,
    ) -> None:
        """Draw a line between two world positions on the BEV canvas."""
        pa = self.world_to_pixel(pos_a)
        pb = self.world_to_pixel(pos_b)
        cv2.line(canvas, pa, pb, color[::-1], thickness, cv2.LINE_AA)

    # Panel composition
    @staticmethod
    def _compose_agent_panel(
        bev: np.ndarray,
        agent_rgb: np.ndarray,
        inset_fraction: float = 0.25,
    ) -> np.ndarray:
        """
        Overlay the agent's first-person RGB as a small inset in the
        bottom-left corner of the BEV image.
        """
        bev_h, bev_w = bev.shape[:2]
        ag_h, ag_w = agent_rgb.shape[:2]

        # Scale agent image to the desired inset width
        inset_w = int(bev_w * inset_fraction)
        scale = inset_w / ag_w
        inset_h = int(ag_h * scale)
        agent_resized = cv2.resize(
            agent_rgb, (inset_w, inset_h), interpolation=cv2.INTER_LINEAR,
        )

        # Bottom-left placement with a small margin
        margin = 8
        x0 = margin
        y0 = bev_h - inset_h - margin

        canvas = bev.copy()
        # Dark border (2px)
        canvas[y0 - 2 : y0 + inset_h + 2, x0 - 2 : x0 + inset_w + 2] = 30
        canvas[y0 : y0 + inset_h, x0 : x0 + inset_w] = agent_resized

        return canvas

    # Future extension helpers
    def make_frontier_markers(
        self,
        positions: List[np.ndarray],
        scores: Optional[List[float]] = None,
    ) -> List[OverlayMarker]:
        """
        Create OverlayMarker objects for frontier positions.

        Convenience method for Phase 8 integration. Pass the returned
        list as extra_markers to render().
        """
        markers = []
        for i, pos in enumerate(positions):
            score = scores[i] if scores else 0.0
            r_val = int(min(score, 1.0) * 255)
            g_val = int((1.0 - min(score, 1.0)) * 255)
            markers.append(OverlayMarker(
                position_world=pos,
                color=(r_val, g_val, 0),
                radius=5,
            ))
        return markers
