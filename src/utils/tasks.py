"""
Scene lookup table, task/episode loading, and simulator config construction.
"""

from __future__ import annotations

import ast
import csv
import gzip
import json
from pathlib import Path

from src.simulator.configs import SimulatorConfig, SensorConfig, ActionConfig, BEVConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

SCENE_LOOKUP: dict[str, str] = {
    "813": "data/hm3d/val/00813-svBbv1Pavdk/svBbv1Pavdk.basis.glb",
    "824": "data/hm3d/val/00824-Dd4bFSTQ8gi/Dd4bFSTQ8gi.basis.glb",
    "827": "data/hm3d/val/00827-BAbdmeyTvMZ/BAbdmeyTvMZ.basis.glb",
    "829": "data/hm3d/val/00829-QaLdnwvtxbs/QaLdnwvtxbs.basis.glb",
    "848": "data/hm3d/val/00848-ziup5kvtCCR/ziup5kvtCCR.basis.glb",
    "853": "data/hm3d/val/00853-5cdEh9F2hJL/5cdEh9F2hJL.basis.glb",
    "871": "data/hm3d/val/00871-VBzV5z6i1WS/VBzV5z6i1WS.basis.glb",
    "876": "data/hm3d/val/00876-mv2HUxq3B53/mv2HUxq3B53.basis.glb",
    "880": "data/hm3d/val/00880-Nfvxx8J5NCo/Nfvxx8J5NCo.basis.glb",
    "894": "data/hm3d/val/00894-HY1NcmCgn3n/HY1NcmCgn3n.basis.glb",
}

TASKS_VLN = PROJECT_ROOT / "data/hm3d/tasks-vln.csv"
TASKS_OBJ = PROJECT_ROOT / "data/hm3d/tasks-objnav.csv"

def _find_scene_path(scene_hash: str) -> str:
    """Resolve a scene hash to the .basis.glb path in data/hm3d/val/."""
    val_dir = PROJECT_ROOT / "data/hm3d/val"
    for d in val_dir.iterdir():
        if d.is_dir() and scene_hash in d.name:
            return str(d / f"{scene_hash}.basis.glb")
    raise ValueError(f"Scene '{scene_hash}' not found in {val_dir}")


def _scene_gz_path(scene_hash: str) -> Path:
    val_dir = PROJECT_ROOT / "data/hm3d/val"
    for d in val_dir.iterdir():
        if d.is_dir() and scene_hash in d.name:
            return d / f"{scene_hash}.json.gz"
    return val_dir / f"{scene_hash}.json.gz"  # fallback (will not exist)


def _load_viewpoints(
    scene_hash: str,
    object_category: str,
) -> list[list[float]]:
    """Load ALL viewpoints for a category from the objectnav_hm3d_v2 dataset.

    Returns viewpoint positions from all instances of the category in the scene.
    """
    gz_path = _scene_gz_path(scene_hash)
    if not gz_path.exists():
        return []
    with gzip.open(gz_path, "rt") as f:
        data = json.load(f)

    goals_key = f"{scene_hash}.basis.glb_{object_category}"
    goals = data.get("goals_by_category", {}).get(goals_key, [])
    if not goals:
        return []

    goal_positions: list[list[float]] = []
    for goal in goals:
        for vp in goal.get("view_points", []):
            pos = vp.get("agent_state", {}).get("position")
            if pos is not None:
                goal_positions.append(pos)
    return goal_positions


def _load_instance_positions(
    scene_hash: str,
    object_category: str,
) -> list[list[float]]:
    """Return the centroid position of each object instance for the category."""
    gz_path = _scene_gz_path(scene_hash)
    if not gz_path.exists():
        return []
    with gzip.open(gz_path, "rt") as f:
        data = json.load(f)

    goals_key = f"{scene_hash}.basis.glb_{object_category}"
    goals = data.get("goals_by_category", {}).get(goals_key, [])
    positions: list[list[float]] = []
    for goal in goals:
        pos = goal.get("position")
        if pos is not None:
            positions.append(pos)
    return positions




def load_task(task_id: int, vln: bool = False) -> dict:
    """Return task metadata for one CSV row.

    vln=True  -> tasks-vln.csv: single goal_position, instruction_text available.
    vln=False -> tasks-obj-30.csv: goal_positions list, no instruction_text.
    """
    csv_path = TASKS_VLN if vln else TASKS_OBJ
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            # ObjNav uses sequential task_id; VLN uses episode_id_dataset
            if vln:
                row_task_id = row.get("episode_id_dataset", row.get("episode_id"))
            else:
                row_task_id = row.get("task_id", row.get("episode_id_dataset"))
            if row_task_id is None:
                continue
            if int(row_task_id) == task_id:
                if vln:
                    scene_dir = row["scene"]  # e.g. "00853-5cdEh9F2hJL"
                    scene_hash = scene_dir.split("-", 1)[1]
                    rel_path = f"data/hm3d/val/{scene_dir}/{scene_hash}.basis.glb"
                    start_pos = ast.literal_eval(row["start_position"])
                    csv_goal = ast.literal_eval(row["goal_position"])
                    return {
                        "scene_path": rel_path,
                        "scene": scene_hash,
                        "object_category": row["object_category"],
                        "text": row["object_category"],
                        "instruction_text": row["instruction_text"],
                        "start_position": start_pos,
                        "goal_position": csv_goal,
                        "goal_positions": [csv_goal],
                    }
                else:
                    scene_hash = row["scene"]  # e.g. "5cdEh9F2hJL"
                    rel_path = _find_scene_path(scene_hash)
                    goal_positions = _load_viewpoints(
                        scene_hash,
                        row["object_category"],
                    )
                    instance_positions = _load_instance_positions(
                        scene_hash,
                        row["object_category"],
                    )
                    return {
                        "scene_path": rel_path,
                        "scene": scene_hash,
                        "object_category": row["object_category"],
                        "text": row["object_category"],
                        "instruction_text": None,
                        "start_position": ast.literal_eval(row["start_position"]),
                        "goal_position": goal_positions[0] if goal_positions else None,
                        "goal_positions": goal_positions,
                        "instance_positions": instance_positions,
                    }
    raise ValueError(f"Task {task_id} not found in {csv_path}")


def load_episode(scene_path: str, idx: int) -> dict:
    """Load episode idx from the .json.gz alongside scene_path."""
    gz = scene_path.replace(".basis.glb", ".json.gz")
    if not Path(gz).exists():
        raise FileNotFoundError(f"No episode file: {gz}")
    with gzip.open(gz, "rt") as f:
        data = json.load(f)
    eps = data["episodes"]
    if idx >= len(eps):
        raise ValueError(f"Episode {idx} out of range (scene has {len(eps)} episodes)")
    return eps[idx]


def make_sim_cfg(
    scene_key: str, seed: int = 42, depth_clip_far: float = 4.0,
    allow_sliding: bool = True, bev: bool = False,
) -> SimulatorConfig:
    """Build a SimulatorConfig from a scene key or direct .glb path."""
    scene_path = SCENE_LOOKUP.get(scene_key, scene_key)
    if not Path(scene_path).is_absolute():
        scene_path = str(PROJECT_ROOT / scene_path)
    navmesh_path = scene_path.replace(".basis.glb", ".navmesh")
    if not Path(navmesh_path).exists():
        navmesh_path = ""
    return SimulatorConfig(
        scene_path=scene_path,
        navmesh_path=navmesh_path,
        sensor=SensorConfig(width=640, height=480, hfov=90.0),
        action=ActionConfig(turn_amount=10.0, forward_amount=0.25),
        bev=BEVConfig(enabled=bev, overhead_height=800, overhead_width=800),
        random_seed=seed,
        depth_clip_far=depth_clip_far,
        allow_sliding=allow_sliding,
    )
