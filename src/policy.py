"""
Semantic frontier exploration policy.

ExplorationPolicy owns all runtime state (mapper, navigator, goal tracking,
counters) and drives the main exploration loop.

Usage
-----
    obs = env.reset()
    result = policy.run(obs, max_steps=1000)
    print(result)
"""

from __future__ import annotations

import math
import queue
import time
from pathlib import Path
from typing import Optional

import numpy as np

from src.mapping import FrontierSemanticMapper
from src.navigation import Navigator
from src.navigation.local_controller import LocalController, LocalControllerConfig
from src.rayfront.frontier_regions import FrontierRegionMap
from src.simulator.viewer import collect_region_data

# OpenGL camera (Habitat default) -> OpenCV camera (Z forward, Y down)
_GL_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])

MIN_ABS_SIM = 0.14
MIN_LANDMARK_SIM = 0.10   # lower threshold for landmark confirmation during scan
N_CONFIRM = 3         # consecutive frames above MIN_ABS_SIM to declare found
SUCCESS_RADIUS = 1.5   # m: XZ distance to GT goal for success
NAVMESH_SNAP_MAX = 2.0 # m: skip frontier regions whose centroid snaps farther than this
APPROACH_STOP_RADIUS = 0.5   # m: already close enough at detection time, skip approach nav
FINAL_APPROACH_EXTRA_STEPS = 200  # extra step budget reserved for the approach after FOUND
NO_SCORE_PATIENCE = 180         # give exploration a bit longer before early-stopping on weak scenes


def _reproject_max_patch(
    feat_map: np.ndarray,
    depth: np.ndarray,
    pose: np.ndarray,
    intrinsics: np.ndarray,
    text_emb: np.ndarray,
) -> Optional[np.ndarray]:
    """Return world-frame 3D position of the highest-similarity patch, or None."""
    ph, pw = feat_map.shape[:2]
    H, W = depth.shape
    sims = feat_map.reshape(-1, feat_map.shape[-1]) @ text_emb  # [ph*pw]
    best = int(np.argmax(sims))
    pi, pj = best // pw, best % pw
    # Map patch centre to image pixel
    vi = min(int((pi + 0.5) * H / ph), H - 1)
    ui = min(int((pj + 0.5) * W / pw), W - 1)
    d = float(depth[vi, ui])
    if d < 0.1:
        return None
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    cam_pos = pose[:3, 3]
    R_cv = (pose @ _GL_TO_CV)[:3, :3]  # OpenCV cam -> world
    p_cam = np.array([(ui - cx) / fx * d, (vi - cy) / fy * d, d])
    return (cam_pos + R_cv @ p_cam).astype(np.float32)


def _candidate_nav_positions(
    agent_pos: np.ndarray,
    target_pos: np.ndarray,
    body_y: float,
) -> list[np.ndarray]:
    """Grid of floor positions around target_pos, sorted closest to agent direction first.

    Samples concentric rings (radii 0.3 -> 2.5 m) at 12 angles each.
    The agent-facing direction comes first so the most natural approach is tried early.
    """
    agent_xz = np.asarray(agent_pos, dtype=np.float32)[[0, 2]]
    target_xz = np.asarray(target_pos, dtype=np.float32)[[0, 2]]
    vec = target_xz - agent_xz
    agent_dist = float(np.linalg.norm(vec))

    # Direction from target back toward agent -- the natural approach angle
    approach_angle = math.atan2(-vec[1], -vec[0]) if agent_dist > 1e-6 else 0.0

    radii = [0.3, 0.6, 1.0, 1.5, 2.0, 2.5]
    n_angles = 12
    seen: list[np.ndarray] = []
    scored: list[tuple[float, np.ndarray]] = []

    for r in radii:
        for i in range(n_angles):
            angle = approach_angle + i * (2 * math.pi / n_angles)
            xz = target_xz + np.array([math.cos(angle) * r, math.sin(angle) * r], dtype=np.float32)
            pos = np.array([float(xz[0]), float(body_y), float(xz[1])], dtype=np.float32)
            if any(float(np.linalg.norm(pos[[0, 2]] - s[[0, 2]])) < 0.05 for s in seen):
                continue
            seen.append(pos)
            angular_offset = abs(math.atan2(math.sin(angle - approach_angle),
                                             math.cos(angle - approach_angle)))
            scored.append((r + angular_offset * 0.3, pos))

    scored.sort(key=lambda x: x[0])
    return [p for _, p in scored]


def _save_found_bev(
    step: int,
    free_pts: Optional[np.ndarray],
    occ_pts: Optional[np.ndarray],
    agent_pos: np.ndarray,
    gt_instances: list,
    locked_target: np.ndarray,
    dump_dir: Path,
) -> None:
    """Save a top-down BEV at FOUND time for navmesh debugging."""
    import cv2 as _cv2

    CELL = 0.1   # grid cell size = wavemap voxel size
    MARGIN = 1.5
    SIZE = 768

    # Bounds include all gt instance positions so every object is on canvas
    bound_pts = [agent_pos[[0, 2]], locked_target[[0, 2]]]
    for g in gt_instances:
        bound_pts.append(np.asarray(g, dtype=np.float32)[[0, 2]])
    if free_pts is not None and len(free_pts):
        bound_pts.append(free_pts[:, [0, 2]].min(axis=0))
        bound_pts.append(free_pts[:, [0, 2]].max(axis=0))
    if occ_pts is not None and len(occ_pts):
        bound_pts.append(occ_pts[:, [0, 2]].min(axis=0))
        bound_pts.append(occ_pts[:, [0, 2]].max(axis=0))
    bound_arr = np.stack(bound_pts)
    x_min, z_min = bound_arr.min(axis=0) - MARGIN
    x_max, z_max = bound_arr.max(axis=0) + MARGIN

    # Rasterize free/occ pts into a 2D grid at CELL resolution
    grid_w = int((x_max - x_min) / CELL) + 2
    grid_h = int((z_max - z_min) / CELL) + 2
    free_grid = np.zeros((grid_h, grid_w), dtype=np.uint8)
    occ_grid  = np.zeros((grid_h, grid_w), dtype=np.uint8)

    if free_pts is not None and len(free_pts):
        cs = np.clip(((free_pts[:, 0] - x_min) / CELL).astype(int), 0, grid_w - 1)
        rs = np.clip(((free_pts[:, 2] - z_min) / CELL).astype(int), 0, grid_h - 1)
        free_grid[rs, cs] = 255

    if occ_pts is not None and len(occ_pts):
        cs = np.clip(((occ_pts[:, 0] - x_min) / CELL).astype(int), 0, grid_w - 1)
        rs = np.clip(((occ_pts[:, 2] - z_min) / CELL).astype(int), 0, grid_h - 1)
        occ_grid[rs, cs] = 255
    # Dilate occupied slightly to make walls visible at any zoom
    occ_grid = _cv2.dilate(occ_grid, np.ones((2, 2), np.uint8))

    # Scale grids to display canvas (aspect-preserving)
    span_x = x_max - x_min
    span_z = z_max - z_min
    scale = (SIZE - 1) / max(span_x, span_z)
    img_w = int(span_x * scale) + 1
    img_h = int(span_z * scale) + 1

    free_canvas = _cv2.resize(free_grid, (img_w, img_h), interpolation=_cv2.INTER_NEAREST)
    occ_canvas  = _cv2.resize(occ_grid,  (img_w, img_h), interpolation=_cv2.INTER_NEAREST)

    canvas = np.full((img_h, img_w, 3), 255, dtype=np.uint8)
    canvas[free_canvas > 0] = (210, 220, 225)  # free = light blue-gray
    canvas[occ_canvas  > 0] = (45,  45,  45)   # occupied = near-black walls

    def w2p(xz):
        c = int((xz[0] - x_min) * scale)
        r = int((xz[1] - z_min) * scale)
        return np.clip(c, 0, img_w - 1), np.clip(r, 0, img_h - 1)

    # GT object instances — green star, one per instance
    for g in gt_instances:
        c, r = w2p(np.asarray(g, dtype=np.float32)[[0, 2]])
        _cv2.drawMarker(canvas, (c, r), (0, 160, 0), _cv2.MARKER_STAR, 16, 2)

    # Locked target (found) — red cross
    c, r = w2p(locked_target[[0, 2]])
    _cv2.drawMarker(canvas, (c, r), (0, 0, 210), _cv2.MARKER_CROSS, 18, 2)

    # Agent — orange filled circle
    c, r = w2p(agent_pos[[0, 2]])
    _cv2.circle(canvas, (c, r), 8, (30, 110, 230), -1)
    _cv2.circle(canvas, (c, r), 8, (0, 0, 0), 1)

    # Legend (bottom-left)
    lx, ly = 8, img_h - 90
    for color, label in [
        ((210, 220, 225), "free"),
        ((45,  45,  45),  "occupied"),
        ((0,  160,   0),  "GT instance"),
        ((0,    0, 210),  "found target"),
        ((30, 110, 230),  "agent"),
    ]:
        _cv2.circle(canvas, (lx + 6, ly + 4), 5, color, -1)
        _cv2.putText(canvas, label, (lx + 16, ly + 9),
                     _cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1)
        ly += 16

    _cv2.putText(canvas, f"step={step}", (8, 18),
                 _cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    bar_px = int(scale)
    _cv2.line(canvas, (img_w - bar_px - 10, img_h - 12),
              (img_w - 10, img_h - 12), (0, 0, 0), 2)
    _cv2.putText(canvas, "1m", (img_w - bar_px - 10, img_h - 16),
                 _cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1)

    dump_dir.mkdir(parents=True, exist_ok=True)
    _cv2.imwrite(str(dump_dir / f"found_bev_{step:05d}.png"), canvas)


def _save_frame(
    step: int,
    rgb: np.ndarray,
    feat_map: np.ndarray,
    text_emb: np.ndarray,
    text: str,
    dump_dir: Path,
    suffix: str = "",
    landmark_embs: Optional[dict] = None,
    landmark_sims: Optional[dict] = None,
) -> None:
    from PIL import Image as _PILImage, ImageDraw as _PILImageDraw

    def _heatmap_rgb(arr: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
        t = (arr - vmin) / max(vmax - vmin, 1e-6)
        t = np.clip(t, 0.0, 1.0)
        r = np.clip(1.5 * t, 0.0, 1.0)
        g = np.clip(1.5 - 2.0 * np.abs(t - 0.5), 0.0, 1.0)
        b = np.clip(1.5 * (1.0 - t), 0.0, 1.0)
        return (np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)

    ph, pw = feat_map.shape[:2]
    H, W = rgb.shape[:2]
    sims = (feat_map.reshape(-1, feat_map.shape[-1]) @ text_emb).reshape(ph, pw)
    s_min, s_max = float(sims.min()), float(sims.max())
    sims_up = np.array(
        _PILImage.fromarray(sims.astype(np.float32), mode="F").resize((W, H), resample=_PILImage.BILINEAR)
    )
    heat = _heatmap_rgb(sims_up, s_min, s_max)
    gap = 10
    header_h = 26
    footer_h = 24 if landmark_sims else 0
    canvas = _PILImage.new("RGB", (W * 2 + gap, H + header_h + footer_h), (255, 255, 255))
    draw = _PILImageDraw.Draw(canvas)
    canvas.paste(_PILImage.fromarray(rgb.astype(np.uint8)), (0, header_h))
    canvas.paste(_PILImage.fromarray(heat), (W + gap, header_h))
    draw.text((6, 6), f"RGB  step={step}", fill=(0, 0, 0))
    draw.text((W + gap + 6, 6), f"sim '{text}' [{s_min:.4f}, {s_max:.4f}]", fill=(0, 0, 0))
    if landmark_sims:
        lines = []
        for lbl, sim in landmark_sims.items():
            marker = " OK" if sim >= MIN_ABS_SIM else ""
            lines.append(f"{lbl}: {sim:.4f}{marker}")
        draw.text((6, H + header_h + 4), " | ".join(lines), fill=(0, 0, 0))
    canvas.save(dump_dir / f"{step:05d}{suffix}.png")


def _save_frame_scan(
    step: int,
    rgb: np.ndarray,
    feat_map: np.ndarray,
    target_emb: np.ndarray,
    target_text: str,
    lm_emb: np.ndarray,
    lm_label: str,
    dump_dir: Path,
    suffix: str,
    landmark_sims: Optional[dict] = None,
) -> None:
    """Save a 3-panel figure: RGB | target heatmap | best-landmark heatmap."""
    from PIL import Image as _PILImage, ImageDraw as _PILImageDraw

    def _heatmap_rgb(arr: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
        t = (arr - vmin) / max(vmax - vmin, 1e-6)
        t = np.clip(t, 0.0, 1.0)
        r = np.clip(1.5 * t, 0.0, 1.0)
        g = np.clip(1.5 - 2.0 * np.abs(t - 0.5), 0.0, 1.0)
        b = np.clip(1.5 * (1.0 - t), 0.0, 1.0)
        return (np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)

    ph, pw = feat_map.shape[:2]
    H, W = rgb.shape[:2]
    patches = feat_map.reshape(-1, feat_map.shape[-1])

    def _upscale(emb):
        s = (patches @ emb).reshape(ph, pw)
        s_min, s_max = float(s.min()), float(s.max())
        sup = np.array(
            _PILImage.fromarray(s.astype(np.float32), mode="F").resize(
                (W, H), resample=_PILImage.BILINEAR)
        )
        return sup, s_min, s_max

    tgt_up, t_min, t_max = _upscale(target_emb)
    lm_up,  l_min, l_max = _upscale(lm_emb)
    tgt_heat = _heatmap_rgb(tgt_up, t_min, t_max)
    lm_heat = _heatmap_rgb(lm_up, l_min, l_max)
    gap = 10
    header_h = 26
    footer_h = 24 if landmark_sims else 0
    canvas = _PILImage.new("RGB", (W * 3 + gap * 2, H + header_h + footer_h), (255, 255, 255))
    draw = _PILImageDraw.Draw(canvas)
    canvas.paste(_PILImage.fromarray(rgb.astype(np.uint8)), (0, header_h))
    canvas.paste(_PILImage.fromarray(tgt_heat), (W + gap, header_h))
    canvas.paste(_PILImage.fromarray(lm_heat), (W * 2 + gap * 2, header_h))
    draw.text((6, 6), f"RGB  step={step}", fill=(0, 0, 0))
    draw.text((W + gap + 6, 6), f"target '{target_text}' [{t_min:.3f}, {t_max:.3f}]", fill=(0, 0, 0))
    draw.text((W * 2 + gap * 2 + 6, 6), f"landmark '{lm_label}' [{l_min:.3f}, {l_max:.3f}]", fill=(0, 0, 0))

    if landmark_sims:
        lines = []
        for lbl, sim in landmark_sims.items():
            marker = " OK" if sim >= MIN_ABS_SIM else ""
            lines.append(f"{lbl}: {sim:.4f}{marker}")
        draw.text((6, H + header_h + 4), " | ".join(lines), fill=(0, 0, 0))

    canvas.save(dump_dir / f"{step:05d}{suffix}.png")


class _NavGoal:
    """Minimal adapter to pass a region centroid to Navigator."""
    def __init__(self, pos3d: np.ndarray, rid: int) -> None:
        self.pos3d = pos3d
        self.id = rid


class ExplorationPolicy:
    """
    Semantic frontier exploration loop.

    Owns: sem_mapper, navigator, region_map, goal state, result counters.
    Does NOT own: env, viewer (injected at construction time).
    """

    def __init__(
        self,
        env,
        sem_mapper: FrontierSemanticMapper,
        navigator: Navigator,
        viewer,
        nav_queue: queue.Queue,
        text_emb: np.ndarray,
        extractor,
        text: str,
        gt_goal_pos: Optional[np.ndarray],
        dump_dir: Optional[Path],
        map_every: int,
        task_id: int = -1,
        object_category: str = "",
        scene_name: str = "",
        nlp_object_embs: Optional[dict] = None,
        gt_goal_positions: Optional[list] = None,
        gt_instance_positions: Optional[list] = None,
    ) -> None:
        self._env = env
        self._sem_mapper = sem_mapper
        self._region_map: FrontierRegionMap = sem_mapper.region_map
        self._nav = navigator
        self._viewer = viewer
        self._nav_queue = nav_queue
        self._text_emb = text_emb
        self._extractor = extractor
        self._text = text
        self._gt_goal_pos = gt_goal_pos
        # All instances of the target category; falls back to single gt_goal_pos if None
        self._gt_goal_positions: Optional[list] = gt_goal_positions
        # Points used for BEV GT overlay:
        # prefer object instance centroids, fallback to goal viewpoints (VLN),
        # and finally to the single goal if only that is available.
        if gt_instance_positions:
            self._gt_instance_positions = list(gt_instance_positions)
        elif gt_goal_positions:
            self._gt_instance_positions = list(gt_goal_positions)
        elif gt_goal_pos is not None:
            self._gt_instance_positions = [gt_goal_pos]
        else:
            self._gt_instance_positions = []
        self._dump_dir = dump_dir
        self._map_every = map_every
        self._task_id = task_id
        self._object_category = object_category
        self._scene_name = scene_name
        # dict label -> (D,) embedding for landmark objects extracted by NLP;
        # None means NLP mode is off → single-embedding detection as before
        self._nlp_object_embs: Optional[dict] = nlp_object_embs
        # (N_anchors, D) L2-normalised mean embeddings of each landmark, for frontier scoring bonus
        if nlp_object_embs:
            vecs = []
            for _, embs in nlp_object_embs.values():
                v = embs.mean(axis=0)
                v = v / (np.linalg.norm(v) + 1e-8)
                vecs.append(v)
            self._anchor_embs: Optional[np.ndarray] = np.stack(vecs)
        else:
            self._anchor_embs = None
        self._align_ctrl = LocalController(LocalControllerConfig())

    def _confirm_with_scan(self, obs, step: int, max_steps: int, candidate_rgb: np.ndarray):
        """
        Small ±20° yaw sweep to check whether landmark objects are visible.

        Sequence (each turn = 10°):
          current view → +20° left (2×L) → back (2×R) → −20° right (2×R) → back (2×L)
        Total: 8 steps, agent returns to original heading.

        Returns (obs, step, confirmed: bool).
        confirmed=True if NLP mode is off OR enough landmarks seen during scan.
        """
        if self._nlp_object_embs is None:
            return obs, step, True

        # Unpack (phrases, embs) tuples from nlp_object_embs
        label_names = list(self._nlp_object_embs.keys())
        landmark_phrases = [self._nlp_object_embs[k][0] for k in label_names]  # list[list[str]]
        landmark_embs    = [self._nlp_object_embs[k][1] for k in label_names]  # list[np.ndarray (N,D)]
        if not landmark_embs:
            return obs, step, True

        n_landmarks = len(landmark_embs)
        best_sim = np.zeros(n_landmarks, dtype=np.float32)

        def _current_lm_sims() -> dict:
            """Current per-landmark best sim so far (for annotation)."""
            return {lbl: float(best_sim[i]) for i, lbl in enumerate(label_names)}

        def _scan_frame():
            feat_map = self._sem_mapper.last_feat_map
            if feat_map is None:
                return
            patches = feat_map.reshape(-1, feat_map.shape[-1])
            for i, emb in enumerate(landmark_embs):
                # emb is (N_synonyms, D) — take max sim over all synonyms
                sim = float((patches @ emb.T).max())
                if sim > best_sim[i]:
                    best_sim[i] = sim

        def _best_landmark_emb() -> tuple:
            """Return (phrase, emb_1d) of the best-matching landmark synonym in current frame."""
            feat_map = self._sem_mapper.last_feat_map
            best_i = int(np.argmax(best_sim))
            emb = landmark_embs[best_i]     # (N_synonyms, D)
            phrases = landmark_phrases[best_i]
            if feat_map is None:
                return phrases[0], emb[0]
            # Pick the synonym with highest max-sim in the current frame
            patches = feat_map.reshape(-1, feat_map.shape[-1])
            syn_sims = (patches @ emb.T).max(axis=0)  # (N_synonyms,)
            best_syn = int(np.argmax(syn_sims))
            return phrases[best_syn], emb[best_syn]

        def _save_scan_frame(suffix: str):
            fm = self._sem_mapper.last_feat_map
            if fm is None or self._dump_dir is None:
                return
            lm_label, lm_emb = _best_landmark_emb()
            _save_frame_scan(
                step, obs.rgb, fm,
                self._text_emb, self._text,
                lm_emb, lm_label,
                self._dump_dir, suffix,
                landmark_sims=_current_lm_sims(),
            )

        def _do(action, n, suffix_prefix: str):
            nonlocal obs, step
            scan_idx = 0
            for _ in range(n):
                if step >= max_steps:
                    return
                obs = self._env.step(action)
                step += 1
                self._sem_mapper.step(obs.depth, obs.rgb, obs.pose, step_idx=step)
                _scan_frame()
                scan_idx += 1
                _save_scan_frame(suffix=f"_SCAN_{suffix_prefix}{scan_idx}")

        # Save the candidate (detection) frame before moving
        if self._dump_dir is not None and self._sem_mapper.last_feat_map is not None:
            _save_scan_frame(suffix="_SCAN_0")

        # Score current frame, then sweep ±20° to confirm landmark visibility
        _scan_frame()
        _do("turn_left",  2, "L")   # +20° left
        _do("turn_right", 2, "C")   # back to centre
        _do("turn_right", 2, "R")   # −20° right
        _do("turn_left",  2, "B")   # back to centre

        n_seen = int((best_sim >= MIN_LANDMARK_SIM).sum())
        min_required = max(1, math.ceil(n_landmarks * 0.6))  # at least 60% of landmarks
        confirmed = n_seen >= min_required

        print(f"  [SCAN] landmarks seen={n_seen}/{n_landmarks}  required={min_required}  confirmed={confirmed}")
        for i, (lbl, sim) in enumerate(zip(label_names, best_sim)):
            marker = " OK" if sim >= MIN_LANDMARK_SIM else ""
            n_phrases = landmark_embs[i].shape[0]
            phrases_str = ", ".join(landmark_phrases[i])
            print(f"    {lbl:20s}  sim={sim:.4f}{marker}  [{n_phrases} phrases: {phrases_str}]")

        return obs, step, confirmed

    def _goal_distance_xz(self, agent_pos: np.ndarray) -> float:
        """Min Euclidean XZ distance to any goal instance."""
        goals = self._gt_goal_positions if self._gt_goal_positions else [self._gt_goal_pos]
        return min(
            float(np.linalg.norm(agent_pos[[0, 2]] - np.asarray(g, dtype=np.float32)[[0, 2]]))
            for g in goals
        )

    def run(self, obs, max_steps: int) -> dict:
        """Execute the full exploration loop. Returns a result dict."""
        def _fmt_pt(p: Optional[np.ndarray]) -> str:
            if p is None:
                return "None"
            return np.array2string(np.asarray(p, dtype=np.float32), precision=2, suppress_small=True)

        def _goal_list() -> list[np.ndarray]:
            goals = self._gt_goal_positions if self._gt_goal_positions else [self._gt_goal_pos]
            return [np.asarray(g, dtype=np.float32) for g in goals if g is not None]

        def _min_xz_details(agent_pos: np.ndarray) -> tuple[Optional[float], Optional[np.ndarray]]:
            best_d: Optional[float] = None
            best_goal: Optional[np.ndarray] = None
            for g in _goal_list():
                d = float(np.linalg.norm(agent_pos[[0, 2]] - g[[0, 2]]))
                if best_d is None or d < best_d:
                    best_d = d
                    best_goal = g
            return best_d, best_goal

        def _min_geodesic(agent_pos: np.ndarray) -> Optional[float]:
            best_d: Optional[float] = None
            for g in _goal_list():
                path = self._env.find_path(agent_pos, g)
                if path is None or len(path) < 2:
                    continue
                d = float(sum(np.linalg.norm(path[i + 1] - path[i]) for i in range(len(path) - 1)))
                if best_d is None or d < best_d:
                    best_d = d
            return best_d

        t_start = time.time()
        step = 0
        goal_centroid: Optional[np.ndarray] = None
        goal_direction: Optional[np.ndarray] = None
        goal_rid: Optional[int] = None
        free_pts = occ_pts = None
        n_arrived = n_stalled = n_selected = 0
        n_found_consec = 0
        scan_cooldown = 0   # steps remaining before another scan is allowed
        found_step = -1
        final_d_xz: Optional[float] = None
        final_success: Optional[bool] = None
        path_length = 0.0
        prev_xz = obs.body_pose[[0, 2], 3].copy()
        gt_geo_start = _min_geodesic(obs.body_pose[:3, 3]) if self._gt_goal_pos is not None else None
        last_map_update_step = -1
        last_scored_step = 0     # last step at which score_all returned non-empty
        frontier_stall_counts: dict[int, int] = {}  # region_id -> number of stalls

        self._sem_mapper.step(obs.depth, obs.rgb, obs.pose)

        def _do_map_update():
            nonlocal free_pts, occ_pts, last_map_update_step
            self._sem_mapper.update(obs.body_pose[:3, 3], step_idx=step)
            free_pts = self._sem_mapper.free_pts
            occ_pts = self._sem_mapper.occ_pts
            self._region_map.record_visit(obs.body_pose[:3, 3])
            last_map_update_step = step

        def _viewer_update():
            member_spheres, arrows, _ = collect_region_data(self._region_map, self._text_emb)
            self._viewer.update(occ_pts, free_pts, member_spheres, arrows, obs.pose, goal_centroid)

        def _handle_candidate(candidate_feat_map: np.ndarray, max_sim: float, prefix: str = "  [CANDIDATE]") -> str:
            nonlocal obs, step, found_step, final_d_xz, final_success
            nonlocal goal_centroid, goal_direction, goal_rid, path_length, prev_xz, n_found_consec, scan_cooldown
            semantic_stop = False

            print(f"{prefix}  step={step}  max_sim={max_sim:.4f}  — starting landmark scan")
            candidate_rgb = obs.rgb.copy()
            candidate_depth = obs.depth.copy()
            candidate_pose = obs.pose.copy()

            obs, step, confirmed = self._confirm_with_scan(obs, step, max_steps, candidate_rgb)

            if not confirmed:
                print(f"  [FALSE POS]  step={step}  landmarks not confirmed — resuming")
                n_found_consec = 0
                scan_cooldown = 50
                self._nav.clear()
                goal_centroid = goal_direction = goal_rid = None
                return "continue"

            found_step = step
            agent_pos = obs.body_pose[:3, 3]
            if self._gt_goal_pos is not None:
                final_d_xz, best_xz_goal = _min_xz_details(agent_pos)
                final_success = final_d_xz < SUCCESS_RADIUS
                print(
                    f"  [FOUND]  step={step}  max_sim={max_sim:.4f}"
                    f"  d_xz={final_d_xz:.2f}m  success={final_success}"
                    f"  (radius={SUCCESS_RADIUS}m)"
                )
                print(f"  [GT-XZ]   best_goal={_fmt_pt(best_xz_goal)}")
            else:
                print(f"  [FOUND]  step={step}  max_sim={max_sim:.4f}  (no GT)")
            if self._dump_dir is not None:
                _save_frame(step, candidate_rgb, candidate_feat_map,
                            self._text_emb, self._text, self._dump_dir, suffix="_FOUND")

            locked_target = _reproject_max_patch(
                candidate_feat_map, candidate_depth, candidate_pose,
                self._sem_mapper._intrinsics, self._text_emb,
            )
            if locked_target is None:
                print("  [FOUND]  no valid depth patch — resuming exploration")
                n_found_consec = 0
                scan_cooldown = 50
                return "continue"

            locked_target[1] = float(obs.body_pose[1, 3])
            if self._dump_dir is not None:
                _save_found_bev(
                    step=step,
                    free_pts=self._sem_mapper.free_pts,
                    occ_pts=self._sem_mapper.occ_pts,
                    agent_pos=obs.body_pose[:3, 3],
                    gt_instances=self._gt_instance_positions,
                    locked_target=locked_target,
                    dump_dir=self._dump_dir,
                )

            _initial_dist = float(np.linalg.norm(
                locked_target[[0, 2]] - obs.body_pose[[0, 2], 3]
            ))
            if _initial_dist <= APPROACH_STOP_RADIUS:
                print(f"  [FOUND]  already within {APPROACH_STOP_RADIUS}m — stopping")
                if self._gt_goal_pos is not None:
                    _p = obs.body_pose[:3, 3]
                    final_d_xz = self._goal_distance_xz(_p)
                    final_success = final_d_xz < SUCCESS_RADIUS
                return "break"

            # Build candidate list: snap (closest navmesh point) first, then angular grid
            _body_y = float(obs.body_pose[1, 3])
            _snap = self._env.snap_point(locked_target)
            _snap_pos = None
            if np.isfinite(_snap).all():
                _snap_pos = _snap.copy()
                _snap_pos[1] = _body_y
            _grid = _candidate_nav_positions(obs.body_pose[:3, 3], locked_target, _body_y)
            _all_candidates = (
                [_snap_pos] + _grid if _snap_pos is not None else _grid
            )

            self._nav.clear()
            _approach_pos = None
            for _cand_idx, _cand in enumerate(_all_candidates, start=1):
                if _cand is None:
                    continue
                if self._nav.set_frontier(_NavGoal(_cand, -1), obs):
                    _approach_pos = _cand
                    print(
                        f"  [FOUND->NAV]  approach candidate #{_cand_idx}/{len(_all_candidates)} "
                        f"at {np.round(_cand, 2)}  target_dist={_initial_dist:.2f}m"
                    )
                    break

            if _approach_pos is None:
                print("  [FOUND->NAV]  no reachable approach position — resuming exploration")
                n_found_consec = 0
                scan_cooldown = 50
                return "continue"

            goal_centroid = _approach_pos
            _viewer_update()

            _step_limit = max_steps + FINAL_APPROACH_EXTRA_STEPS
            while step < _step_limit:
                _nav_result = self._nav.step(obs)
                if _nav_result.arrived or _nav_result.stalled:
                    break
                obs = self._env.step(_nav_result.action)
                step += 1
                new_xz = obs.body_pose[[0, 2], 3]
                path_length += float(np.linalg.norm(new_xz - prev_xz))
                prev_xz = new_xz.copy()
                self._sem_mapper.step(obs.depth, obs.rgb, obs.pose, step_idx=step)
                _viewer_update()

            self._nav.clear()
            goal_centroid = None
            _viewer_update()

            if self._gt_goal_pos is not None:
                _p = obs.body_pose[:3, 3]
                final_d_xz = self._goal_distance_xz(_p)
                final_success = final_d_xz < SUCCESS_RADIUS
            print(
                f"  [FOUND->NAV]  arrived at approach  "
                f"d_xz={final_d_xz:.2f}m  success={final_success}"
                if final_d_xz is not None else "  [FOUND->NAV]  arrived at approach"
            )
            return "break"

        try:
            while step < max_steps:
                try:
                    if self._nav_queue.get_nowait() == "__quit__":
                        print("\n[quit]")
                        break
                except queue.Empty:
                    pass

                if self._nav.current_frontier is None:
                    # Only re-sync regions when the agent has actually moved
                    if step != last_map_update_step:
                        _do_map_update()
                    self._sem_mapper.step(obs.depth, obs.rgb, obs.pose, step_idx=step)
                    scored = [
                        (c, d, s, r) for c, d, s, r in self._region_map.score_all(
                            self._text_emb, anchor_embs=self._anchor_embs)
                        if float(np.linalg.norm(self._env.snap_point(c)[[0, 2]] - c[[0, 2]])) <= NAVMESH_SNAP_MAX
                    ]

                    if scored:
                        last_scored_step = step
                        goal_centroid, goal_direction, goal_score, goal_rid = scored[0]
                    else:
                        if step - last_scored_step >= NO_SCORE_PATIENCE:
                            print(f"  [early-stop]  step={step}  no scored regions for {step - last_scored_step} steps")
                            break
                        # Apply same visited filter as score_all to avoid re-selecting
                        # regions the agent has already been to
                        novel_regions = [r for r in self._region_map.active_regions
                                         if self._region_map._is_novel(r.centroid)]
                        regions = novel_regions if novel_regions else self._region_map.active_regions
                        if not regions:
                            # No novel frontier: spin to reveal unexplored space
                            obs = self._env.step("turn_left")
                            step += 1
                            self._sem_mapper.step(obs.depth, obs.rgb, obs.pose, step_idx=step)
                            _viewer_update()
                            continue
                        agent_pos = obs.body_pose[:3, 3]
                        agent_xz = agent_pos[[0, 2]]
                        far = [r for r in regions
                               if float(np.linalg.norm(r.centroid[[0, 2]] - agent_xz)) > 1.0]
                        pool = far if far else regions
                        # When semantic scores are missing, prefer pushing into farther
                        # unexplored pockets instead of re-clearing the local neighborhood.
                        frontier = max(pool, key=lambda r: np.linalg.norm(r.centroid - agent_pos))
                        goal_centroid = frontier.centroid.copy()
                        goal_direction = None
                        goal_score = 0.0
                        goal_rid = frontier.id
                        _label = "far" if novel_regions else "fallback"
                        print(f"  step={step}  no features yet -- exploring {_label} region {goal_rid}")

                    nav_pos = goal_centroid.copy()
                    nav_pos[1] = float(obs.body_pose[1, 3])
                    nav_goal = _NavGoal(nav_pos, goal_rid)
                    ok = self._nav.set_frontier(nav_goal, obs)
                    if not ok:
                        self._region_map.invalidate_near(goal_centroid)
                        goal_centroid = None
                        goal_rid = None
                        continue

                    n_selected += 1
                    print(
                        f"  [select #{n_selected}]  step={step}  "
                        f"region {goal_rid}  score={goal_score:.3f}  "
                        f"pos={np.round(goal_centroid, 2)}"
                    )
                    _viewer_update()

                result = self._nav.step(obs)

                if result.arrived:
                    if goal_direction is not None:
                        look_dir = goal_direction.copy()
                        look_dir[1] = 0.0
                        ln = float(np.linalg.norm(look_dir))
                        if ln > 1e-6:
                            look_dir /= ln
                            look_target = obs.body_pose[:3, 3] + look_dir * 3.0
                            for _ in range(24):
                                action = self._align_ctrl.step(obs, look_target)
                                if action not in ("turn_left", "turn_right"):
                                    break
                                obs = self._env.step(action)
                                step += 1
                                self._sem_mapper.step(obs.depth, obs.rgb, obs.pose, step_idx=step)
                    print(f"  [arrived]  step={step}  region {goal_rid}")
                    self._region_map.invalidate_near(goal_centroid)
                    n_arrived += 1
                    goal_centroid = None
                    goal_direction = None
                    goal_rid = None
                    self._nav.clear()
                    _do_map_update()
                    _viewer_update()

                elif result.stalled:
                    print(f"  [stalled]  step={step}  region {goal_rid}")
                    n_stalled += 1
                    self._nav.clear()
                    # Lateral escape: sidestep to the right, then left, while keeping
                    # detection active. This is less brittle than a random turn.
                    _stall_detected = False

                    def _recovery_step(action: str) -> bool:
                        nonlocal obs, step, path_length, prev_xz, n_found_consec, scan_cooldown, _stall_detected
                        if step >= max_steps:
                            return False
                        obs = self._env.step(action)
                        step += 1
                        if action == "move_forward":
                            new_xz = obs.body_pose[[0, 2], 3]
                            path_length += float(np.linalg.norm(new_xz - prev_xz))
                            prev_xz = new_xz.copy()
                        self._sem_mapper.step(obs.depth, obs.rgb, obs.pose, step_idx=step)
                        _fm = self._sem_mapper.last_feat_map
                        if _fm is not None and self._extractor is not None:
                            _patches = _fm.reshape(-1, _fm.shape[-1])
                            _ms = float((_patches @ self._text_emb).max())
                            n_found_consec = n_found_consec + 1 if _ms >= MIN_ABS_SIM else 0
                            if scan_cooldown > 0:
                                scan_cooldown -= 1
                            if n_found_consec >= N_CONFIRM and scan_cooldown == 0:
                                print(f"  [stall-recovery CANDIDATE]  step={step}  max_sim={_ms:.4f}")
                                _stall_detected = True
                                return False
                        return True

                    _escaped = False
                    _backed_off = False
                    _backoff_start = obs.body_pose[[0, 2], 3].copy()
                    _backoff_ok = True
                    for _ in range(18):  # face away from the blockage
                        if not _recovery_step("turn_right"):
                            _backoff_ok = False
                            break
                    if not _stall_detected and _backoff_ok:
                        for _ in range(2):  # move away from the obstacle
                            if not _recovery_step("move_forward"):
                                _backoff_ok = False
                                break
                    if not _stall_detected:
                        for _ in range(18):  # restore original heading
                            if not _recovery_step("turn_right"):
                                _backoff_ok = False
                                break
                    if not _stall_detected:
                        _backoff_moved = float(np.linalg.norm(obs.body_pose[[0, 2], 3] - _backoff_start))
                        if _backoff_ok and _backoff_moved >= 0.2:
                            print(f"  [stalled]  backoff moved {_backoff_moved:.2f}m")
                            _backed_off = True

                    for _label, _turn_out, _turn_back in (
                        ("right", "turn_right", "turn_left"),
                        ("left", "turn_left", "turn_right"),
                    ):
                        if _backed_off:
                            break
                        _start_xz = obs.body_pose[[0, 2], 3].copy()
                        _ok = True
                        for _ in range(9):  # ~90 deg
                            if not _recovery_step(_turn_out):
                                _ok = False
                                break
                        if _stall_detected:
                            break
                        if _ok:
                            for _ in range(3):
                                if not _recovery_step("move_forward"):
                                    _ok = False
                                    break
                        if _stall_detected:
                            break
                        for _ in range(9):  # face original heading again
                            if not _recovery_step(_turn_back):
                                _ok = False
                                break
                        _moved = float(np.linalg.norm(obs.body_pose[[0, 2], 3] - _start_xz))
                        if _ok and _moved >= 0.45:
                            print(f"  [stalled]  lateral escape {_label} moved {_moved:.2f}m")
                            _escaped = True
                            break
                    if not _stall_detected and not _escaped and not _backed_off:
                        print("  [stalled]  recovery failed")
                    self._region_map.record_visit(obs.body_pose[:3, 3])
                    if _stall_detected:
                        # Let the main loop handle the candidate on next iteration
                        goal_centroid = goal_direction = goal_rid = None
                        continue
                    # Count stalls per frontier; invalidate after 2 stalls on the same region
                    if goal_rid is not None:
                        frontier_stall_counts[goal_rid] = frontier_stall_counts.get(goal_rid, 0) + 1
                    if goal_rid is not None and frontier_stall_counts[goal_rid] >= 2:
                        print(f"  [stalled] {frontier_stall_counts[goal_rid]} stalls on region {goal_rid} — invalidating")
                        self._region_map.invalidate_near(goal_centroid)
                        goal_centroid = None
                        goal_direction = None
                        goal_rid = None
                    else:
                        # Retry path to same frontier; if it fails, invalidate
                        _nav_goal = _NavGoal(goal_centroid, goal_rid)
                        if goal_centroid is not None and not self._nav.set_frontier(_nav_goal, obs):
                            print(f"  [stalled] retry failed — invalidating region {goal_rid}")
                            self._region_map.invalidate_near(goal_centroid)
                            goal_centroid = None
                            goal_direction = None
                            goal_rid = None

                else:
                    obs = self._env.step(result.action)
                    step += 1
                    new_xz = obs.body_pose[[0, 2], 3]
                    path_length += float(np.linalg.norm(new_xz - prev_xz))
                    prev_xz = new_xz.copy()
                    self._sem_mapper.step(obs.depth, obs.rgb, obs.pose, step_idx=step)

                    if (self._dump_dir is not None and step % 5 == 0
                            and self._sem_mapper.last_feat_map is not None
                            and self._extractor is not None):
                        _save_frame(step, obs.rgb, self._sem_mapper.last_feat_map,
                                    self._text_emb, self._text, self._dump_dir)

                    feat_map = self._sem_mapper.last_feat_map
                    if feat_map is not None and self._extractor is not None:
                        patches = feat_map.reshape(-1, feat_map.shape[-1])  # [N, D]
                        max_sim = float((patches @ self._text_emb).max())
                        n_found_consec = n_found_consec + 1 if max_sim >= MIN_ABS_SIM else 0
                        if scan_cooldown > 0:
                            scan_cooldown -= 1
                        if step % self._map_every == 0:
                            print(f"  [sim]  step={step}  max_sim={max_sim:.4f}  consec={n_found_consec}")
                        if n_found_consec >= N_CONFIRM and scan_cooldown == 0:
                            _candidate_outcome = _handle_candidate(feat_map, max_sim)
                            if _candidate_outcome == "break":
                                break
                            if _candidate_outcome == "continue":
                                continue

                    if step % self._map_every == 0:
                        _do_map_update()
                    self._region_map.invalidate_near(obs.body_pose[:3, 3])
                    _viewer_update()

        except KeyboardInterrupt:
            pass

        print(f"\nDone.  steps={step}  selected={n_selected}  arrived={n_arrived}  stalled={n_stalled}")
        print(self._region_map.summary())

        # Recompute distance metrics from the agent's actual final position
        if self._gt_goal_pos is not None:
            _final_pos = obs.body_pose[:3, 3]
            final_d_xz = self._goal_distance_xz(_final_pos)
            final_success = final_d_xz < SUCCESS_RADIUS

        return {
            "task_id": self._task_id,
            "scene": self._scene_name,
            "object_category": self._object_category,
            "instruction": self._text,
            "found": found_step >= 0,
            "success": final_success,
            "d_xz": round(final_d_xz, 3) if final_d_xz is not None else None,
            "gt_geo_start": round(gt_geo_start, 3) if gt_geo_start is not None else None,
            "steps": step,
            "found_step": found_step,
            "path_length": round(path_length, 2),
            "n_selected": n_selected,
            "n_arrived": n_arrived,
            "n_stalled": n_stalled,
            "elapsed_s": round(time.time() - t_start, 1),
        }
