"""
Configuration dataclasses for the Habitat simulator wrapper.

These are plain dataclasses that mirror what will eventually become
Hydra structured configs.  When Hydra is wired in, each class gets a
``@dataclass`` + ``@hydra.main`` / ``cs.store`` counterpart — the fields
and defaults remain identical.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SensorConfig:
    """Per-sensor settings (shared by RGB and depth sensors)."""

    height: int = 480           # image height in pixels
    width: int = 640            # image width in pixels
    hfov: float = 90.0          # horizontal field of view in degrees
    sensor_height: float = 1.25  # sensor mount height above ground (metres)


@dataclass
class ActionConfig:
    """Discrete action parameters."""

    forward_amount: float = 0.25   # metres per move_forward
    turn_amount: float = 10.0      # degrees per turn_left / turn_right


@dataclass
class BEVConfig:
    """Bird's-eye-view visualization settings (developer-only, not agent knowledge)."""

    # Overhead sensor (Mode 1: rendered frames)
    enabled: bool = False               # master switch for overhead sensor
    overhead_height: int = 512          # overhead image height in pixels
    overhead_width: int = 512           # overhead image width in pixels
    overhead_hfov: float = 90.0         # base hfov (adjusted to fit scene)
    overhead_margin: float = 1.0        # extra margin around scene bounds (m)

    # Agent overlay on overhead image
    agent_marker_radius: int = 6        # radius of agent dot in pixels
    agent_color_r: int = 255            # agent marker color (RGB)
    agent_color_g: int = 0
    agent_color_b: int = 0
    trajectory_color_r: int = 0         # trajectory polyline color (RGB)
    trajectory_color_g: int = 255
    trajectory_color_b: int = 0
    trajectory_thickness: int = 2       # polyline thickness in pixels

    # Interactive Open3D viewer (Mode 2)
    viewer_enabled: bool = False        # launch Open3D viewer
    viewer_width: int = 1024
    viewer_height: int = 768
    viewer_fps: float = 30.0            # poll rate for Open3D event loop
    agent_sphere_radius: float = 0.15   # 3D agent marker size (m)
    arrow_length: float = 0.4           # heading arrow length (m)
    arrow_radius: float = 0.03          # heading arrow cylinder radius (m)
    trajectory_line_radius: float = 0.02  # 3D trajectory tube radius (m)

    @property
    def agent_color(self):
        return (self.agent_color_r, self.agent_color_g, self.agent_color_b)

    @property
    def trajectory_color(self):
        return (self.trajectory_color_r, self.trajectory_color_g, self.trajectory_color_b)

    @property
    def agent_color_normalized(self):
        return (self.agent_color_r / 255.0, self.agent_color_g / 255.0, self.agent_color_b / 255.0)


@dataclass
class SimulatorConfig:
    """Top-level simulator configuration."""

    # Scene
    scene_path: str = ""            # path to .glb scene file
    navmesh_path: str = ""          # path to .navmesh (auto-derived if empty)

    # Agent body
    agent_height: float = 1.25      # agent capsule height (metres)
    agent_radius: float = 0.1       # agent capsule radius (metres)

    # Sensors (one config shared by RGB and depth for now)
    sensor: SensorConfig = field(default_factory=SensorConfig)

    # Actions
    action: ActionConfig = field(default_factory=ActionConfig)

    # BEV visualization
    bev: BEVConfig = field(default_factory=BEVConfig)

    # Rendering
    gpu_device_id: int = 0          # CUDA device
    enable_physics: bool = False    # physics simulation (not needed for now)
    allow_sliding: bool = True      # slide along walls on collision
    random_seed: int = 42           # reproducibility

    # Depth clipping
    depth_clip_far: float = 10.0    # clip depth beyond this (metres)
