#!/usr/bin/env python3
"""
Semantic frontier exploration -- batch runner.

Usage:
    python run_tasks.py episodes=all
    python run_tasks.py episodes=1-10 resume=true
    python run_tasks.py episodes=0,5,18 dump=true
    python run_tasks.py episodes=5 vln=true
    python run_tasks.py episodes=all no_viewer=true max_steps=500
"""

from __future__ import annotations

import os
import sys

# Prevent matplotlib's _fix_ipython_backend2gui from crashing on circular IPython import
if "IPython" not in sys.modules:
    import types
    _mock_ipy = types.ModuleType("IPython")
    _mock_ipy.version_info = (0, 0)
    _mock_ipy.get_ipython = lambda: None
    sys.modules["IPython"] = _mock_ipy

import csv
import json
import logging
import queue
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import hydra
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"
os.environ.setdefault("MPLBACKEND", "Agg")
logging.getLogger("habitat_sim").setLevel(logging.ERROR)

from src.features import FeatureExtractor
from src.mapping import FrontierSemanticMapper
from src.navigation import Navigator
from src.nlp.nlp_pipeline import NLPPipeline
from src.policy import ExplorationPolicy
from src.simulator import HabitatEnv
from src.simulator.viewer import NullViewer, Viewer
from src.utils.tasks import TASKS_OBJ, TASKS_VLN, load_task, make_sim_cfg
from src.utils.torch_utils import seed_everything

RESULT_FIELDS = [
    "task_id", "scene", "object_category", "instruction",
    "found", "success",
    "d_xz", "gt_geo_start", "steps", "found_step", "path_length",
    "n_selected", "n_arrived", "n_stalled",
    "elapsed_s",
]


def _set_max_csv_field_size() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def _parse_task_ids(spec: str, vln: bool = False) -> list[int]:
    _set_max_csv_field_size()
    if spec == "all":
        csv_path = TASKS_VLN if vln else TASKS_OBJ
        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))
        ids: list[int] = []
        for r in rows:
            if vln:
                raw = r.get("episode_id_dataset", r.get("episode_id"))
            else:
                raw = r.get("task_id")
                if raw in (None, ""):
                    raw = r.get("episode_id", r.get("episode_id_dataset"))
            if raw is not None:
                ids.append(int(raw))
        return ids
    if "-" in spec and "," not in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in spec.split(",")]


def _group_by_scene(task_ids: list[int], vln: bool) -> list[tuple[str, list[int]]]:
    scene_to_tasks: dict[str, list[int]] = defaultdict(list)
    scene_order: list[str] = []
    for tid in task_ids:
        task = load_task(tid, vln=vln)
        sp = task["scene_path"]
        if sp not in scene_to_tasks:
            scene_order.append(sp)
        scene_to_tasks[sp].append(tid)
    return [(sp, scene_to_tasks[sp]) for sp in scene_order]


def _empty_result(task_id: int) -> dict:
    return {f: None for f in RESULT_FIELDS} | {"task_id": task_id, "found": False, "success": False}


def _snap_eval_goals(env: HabitatEnv, raw_goals: list) -> list[np.ndarray]:
    snapped: list[np.ndarray] = []
    for g in raw_goals:
        sg = np.asarray(env.snap_point(np.asarray(g, dtype=np.float32)), dtype=np.float32)
        if not np.isfinite(sg).all():
            continue
        if any(float(np.linalg.norm(sg[[0, 2]] - prev[[0, 2]])) < 0.05 for prev in snapped):
            continue
        snapped.append(sg)
    return snapped


def _run_one_task(
    task_id: int,
    env: HabitatEnv,
    extractor: Optional[FeatureExtractor],
    cfg: DictConfig,
    dump_root: Optional[Path],
):
    task = load_task(task_id, vln=cfg.vln)
    obs = env.reset()

    if task.get("start_position") is not None:
        obs = env.set_agent_pose(
            np.array(task["start_position"], dtype=np.float32),
            np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        )

    object_category = task.get("object_category", "")
    scene_name = task.get("scene", "")
    text = task["instruction_text"] if cfg.vln else task.get("text", "")

    raw = task.get("goal_positions") or (
        [task["goal_position"]] if task.get("goal_position") is not None else []
    )
    gt_goal_positions = _snap_eval_goals(env, [np.array(p, dtype=np.float32) for p in raw])
    gt_goal_pos: Optional[np.ndarray] = gt_goal_positions[0] if gt_goal_positions else None
    gt_instance_positions = [
        np.array(p, dtype=np.float32) for p in task.get("instance_positions", [])
    ]
    if gt_goal_positions:
        print(f"  {len(gt_goal_positions)} goal viewpoint(s), "
              f"{len(gt_instance_positions)} instance(s) for '{object_category}'")

    nlp_object_embs: Optional[dict] = None
    if extractor is None:
        text_emb = np.zeros(1, dtype=np.float32)
    else:
        if cfg.vln:
            target_phrase, text_emb, nlp_object_embs = NLPPipeline().build(text, extractor)
            text = target_phrase
            print(f"  [NLP] target     = '{target_phrase}'")
        else:
            text_emb = extractor.encode_text(text)
        print(f"  text='{text}'  norm={np.linalg.norm(text_emb):.4f}")

    map_cfg    = instantiate(cfg.mapping)
    front_cfg  = instantiate(cfg.frontier_detector)
    regions_cfg = instantiate(cfg.frontier_regions)

    sem_mapper = FrontierSemanticMapper(
        map_cfg, front_cfg, regions_cfg,
        obs.rgb.shape[1], obs.rgb.shape[0],
        env.intrinsics, extractor,
    )
    nav = Navigator(
        instantiate(cfg.navigator.controller),
        instantiate(cfg.navigator.config),
        find_path_fn=env.find_path,
    )
    nav_queue: queue.Queue = queue.Queue()
    if cfg.no_viewer:
        viewer = NullViewer()
    else:
        viewer = Viewer(
            width=1400, height=900,
            intrinsics=env.intrinsics,
            img_w=obs.rgb.shape[1],
            img_h=obs.rgb.shape[0],
            scene_path=str(task.get("scene_path", "")),
            nav_queue=nav_queue,
            gt_goal_pos=gt_goal_pos,
            gt_goal_positions=gt_goal_positions,
            gt_instance_positions=gt_instance_positions,
            overhead_cam_params=env.overhead_cam_params,
            overhead_rgb=getattr(obs, "overhead_rgb", None),
        )
    viewer.start()

    dump_dir: Optional[Path] = None
    if dump_root is not None:
        dump_dir = dump_root / f"task{task_id}"
        if dump_dir.exists():
            shutil.rmtree(dump_dir)
        dump_dir.mkdir(parents=True, exist_ok=True)
        print(f"Dumping frames -> {dump_dir}")

    policy = ExplorationPolicy(
        env=env,
        sem_mapper=sem_mapper,
        navigator=nav,
        viewer=viewer,
        nav_queue=nav_queue,
        text_emb=text_emb,
        extractor=extractor,
        text=text,
        gt_goal_pos=gt_goal_pos,
        dump_dir=dump_dir,
        map_every=cfg.map_every,
        task_id=task_id,
        object_category=object_category,
        scene_name=scene_name,
        nlp_object_embs=nlp_object_embs,
        gt_goal_positions=gt_goal_positions,
        gt_instance_positions=gt_instance_positions,
    )
    try:
        result = policy.run(obs, max_steps=cfg.max_steps)
    finally:
        viewer.stop()

    for field in RESULT_FIELDS:
        result.setdefault(field, None)
    return result


def _batch_run(cfg: DictConfig) -> None:
    """Batch run over multiple tasks, grouped by scene."""
    # Suppress verbose INFO from RADIO/OpenCLIP -- must be inside the hydra-decorated
    # function because Hydra resets logging configuration before calling it.
    logging.getLogger("root").setLevel(logging.WARNING)
    logging.getLogger("open_clip").setLevel(logging.WARNING)

    _set_max_csv_field_size()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = PROJECT_ROOT / "results" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    results_csv = out_dir / "results.csv"
    config_json = out_dir / "config.json"

    task_ids = _parse_task_ids(str(cfg.episodes), vln=cfg.vln)

    done_ids: set[int] = set()
    if cfg.resume and results_csv.exists():
        with open(results_csv, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("task_id") not in (None, ""):
                    done_ids.add(int(row["task_id"]))
        task_ids = [t for t in task_ids if t not in done_ids]
        print(f"Resuming: {len(done_ids)} tasks already done, {len(task_ids)} remaining")
    else:
        print(f"Running {len(task_ids)} tasks")

    print(f"Output   : {out_dir}\n")

    # Save full resolved config
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    config_dict["out_dir"] = str(out_dir)
    with open(config_json, "w") as f:
        json.dump(config_dict, f, indent=2)

    if not cfg.resume or not results_csv.exists():
        with open(results_csv, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=RESULT_FIELDS).writeheader()

    seed_everything(cfg.seed)

    print("Loading models...")
    extractor = instantiate(cfg.extractor)
    print(f"  device={extractor.device}  dim={extractor.feature_dim}")

    scene_groups = _group_by_scene(task_ids, vln=cfg.vln)
    print(f"  {len(scene_groups)} scenes  ({len(task_ids)} tasks total)\n")

    dump_root = out_dir / "dumps" if cfg.dump else None
    results = []

    for scene_path, tids in scene_groups:
        scene_name = Path(scene_path).stem.split(".")[0]
        print(f"\n=== Scene: {scene_name}  ({len(tids)} tasks) ===")
        sim_cfg = make_sim_cfg(
            scene_path,
            seed=cfg.seed,
            depth_clip_far=cfg.simulator.depth_clip_far,
            allow_sliding=cfg.simulator.allow_sliding,
            bev=not cfg.no_viewer,
        )
        print(f"Scene    : {sim_cfg.scene_path}")
        print(f"Steps    : {cfg.max_steps}  map-every={cfg.map_every}")
        env = HabitatEnv(sim_cfg)
        env.reset()
        try:
            for task_id in tids:
                seed_everything(cfg.seed)
                print(f"\n=== Task {task_id} ===")
                print(
                    f"  cmd: episodes={task_id} max_steps={cfg.max_steps} "
                    f"map_every={cfg.map_every}"
                    f"{' vln=true' if cfg.vln else ''}"
                    f"{' dump=true' if cfg.dump else ''}"
                )
                try:
                    row = _run_one_task(task_id, env, extractor, cfg, dump_root)
                except Exception as e:
                    import traceback
                    print(f"  [ERROR] task {task_id}: {e}")
                    traceback.print_exc()
                    row = _empty_result(task_id)
                results.append(row)
                print(
                    f"  found={row['found']}  success={row['success']}"
                    f"  d_xz={row['d_xz']}  steps={row['steps']}"
                    f"  path={row['path_length']}m"
                )
                with open(results_csv, "a", newline="") as f:
                    csv.DictWriter(f, fieldnames=RESULT_FIELDS, extrasaction="ignore").writerow(row)
        finally:
            env.close()

    n = len(results)
    n_found = sum(1 for r in results if r["found"])
    n_success = sum(1 for r in results if r["success"])
    steps_found = [r["steps"] for r in results if r["found"] and r["steps"] is not None]
    paths_found = [r["path_length"] for r in results if r["found"] and r["path_length"] is not None]
    d_xz_success = [r["d_xz"] for r in results if r["success"] and r["d_xz"] is not None]

    print(f"\nResults saved to {results_csv}")
    if n > 0:
        print(f"Found:   {n_found}/{n}  ({n_found/n:.1%})")
        print(f"Success: {n_success}/{n}  ({n_success/n:.1%})")
    if steps_found:
        print(f"Steps (found):  avg={sum(steps_found)/len(steps_found):.0f}  "
              f"min={min(steps_found)}  max={max(steps_found)}")
    if paths_found:
        print(f"Path  (found):  avg={sum(paths_found)/len(paths_found):.1f}m  "
              f"min={min(paths_found):.1f}m  max={max(paths_found):.1f}m")
    if d_xz_success:
        print(f"d_xz  (success): avg={sum(d_xz_success)/len(d_xz_success):.2f}m")


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    _batch_run(cfg)


if __name__ == "__main__":
    main()
