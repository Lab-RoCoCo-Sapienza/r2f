"""Utilities for computing aggregate metrics from task-level results CSV files."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def _parse_bool(value: Any) -> bool:
    """Parse common CSV boolean encodings; unknown values are treated as False."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "t", "yes", "y"}


def _parse_float(value: Any) -> float | None:
    """Parse numeric CSV values and return None for empty/invalid inputs."""
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def compute_results_csv_metrics(csv_path: str | Path) -> dict[str, float | int | None]:
    """Compute aggregate metrics from a results CSV.

    Metrics:
    - success_rate: mean of `success` over all tasks.
    - avg_elapsed_s_on_success: mean of `elapsed_s` over successful tasks.
    - success_weighted_by_path_length: for successful tasks only, mean of:
        gt_geo_start_i / max(path_length_i, gt_geo_start_i)
    """
    path = Path(csv_path)
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    n_tasks = len(rows)
    if n_tasks == 0:
        return {
            "n_tasks": 0,
            "success_rate": 0.0,
            "avg_elapsed_s_on_success": None,
            "spl": None,
        }

    success_count = 0
    elapsed_success: list[float] = []
    spl_terms: list[float] = []  # one entry per task (0 for failures)

    for row in rows:
        success = _parse_bool(row.get("success"))
        if success:
            success_count += 1
            elapsed = _parse_float(row.get("elapsed_s"))
            if elapsed is not None:
                elapsed_success.append(elapsed)

        gt_geo_start = _parse_float(row.get("gt_geo_start"))
        path_length = _parse_float(row.get("path_length"))
        if (
            success
            and gt_geo_start is not None
            and path_length is not None
            and gt_geo_start > 0.0
        ):
            spl_terms.append(gt_geo_start / max(path_length, gt_geo_start))
        else:
            # S_i = 0: contributes zero to the SPL sum, still counts in N
            spl_terms.append(0.0)

    avg_elapsed_s_on_success = (
        sum(elapsed_success) / len(elapsed_success) if elapsed_success else None
    )
    # SPL = (1/N) * sum_i( S_i * L*_i / max(L_i, L*_i) )
    success_weighted_by_path_length = sum(spl_terms) / n_tasks

    return {
        "n_tasks": n_tasks,
        "success_rate": round(success_count / n_tasks, 4),
        "avg_elapsed_s_on_success": avg_elapsed_s_on_success,
        "spl": round(success_weighted_by_path_length, 4),
    }
