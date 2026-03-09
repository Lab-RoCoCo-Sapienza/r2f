from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Observation:
    """Single-step sensor bundle returned by the simulator wrapper."""

    rgb: np.ndarray          # (H, W, 3) uint8
    depth: np.ndarray        # (H, W)   float32, metres
    pose: np.ndarray         # (4, 4)   camera-to-world SE(3)
    intrinsics: np.ndarray   # (3, 3)   camera intrinsic matrix K
    step_id: int = 0         # monotonically increasing counter
    body_pose: Optional[np.ndarray] = None  # (4, 4) agent body SE(3), Y at floor level
    overhead_rgb: Optional[np.ndarray] = field(  # (OH, OW, 3) uint8 if BEV enabled
        default=None, repr=False,
    )
