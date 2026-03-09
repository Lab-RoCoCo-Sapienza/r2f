#!/usr/bin/env python3
"""CLI helper to evaluate task metrics from a results CSV.

Usage:
    python eval.py                                  # latest results run
    python eval.py csv_path=results/2026-03-06_*/   # specific run
    python eval.py json=true                        # machine-readable output
"""

from __future__ import annotations

import json
from pathlib import Path

import hydra
from omegaconf import DictConfig

from src.utils import compute_results_csv_metrics


def _default_results_csv(project_root: Path) -> Path | None:
    candidates = sorted((project_root / "results").glob("**/results.csv"))
    return candidates[-1] if candidates else None


@hydra.main(config_path="config", config_name="eval", version_base=None)
def main(cfg: DictConfig) -> None:
    project_root = Path(__file__).resolve().parent

    csv_path = Path(cfg.csv_path) if cfg.csv_path else _default_results_csv(project_root)
    if csv_path is not None and csv_path.is_dir():
        csv_path = csv_path / "results.csv"
    if csv_path is None:
        raise ValueError("No CSV path provided and no results/**/results.csv found.")
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    metrics = compute_results_csv_metrics(csv_path)

    if cfg.json:
        print(json.dumps(metrics, indent=2))
        return

    print(f"csv: {csv_path}")
    print(f"n_tasks: {metrics['n_tasks']}")
    print(f"success_rate: {metrics['success_rate']:.4f}")

    avg_elapsed = metrics["avg_elapsed_s_on_success"]
    spl = metrics["spl"]
    print(
        "avg_elapsed_s_on_success: "
        f"{avg_elapsed:.4f}" if avg_elapsed is not None else "avg_elapsed_s_on_success: None"
    )
    print(f"spl: {spl:.4f}")


if __name__ == "__main__":
    main()
