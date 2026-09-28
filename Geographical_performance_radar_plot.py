"""Create a radar plot from geographical-validation performance metrics."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


METRICS = ["AUROC", "AUPRC", "Accuracy", "Sensitivity", "Specificity"]
DPI = 600


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, default=Path("results/geographical_metrics.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    table = pd.read_csv(args.metrics)
    missing = [metric for metric in METRICS if metric not in table.columns]
    if missing or len(table) != 1:
        raise ValueError(f"Metric table must contain one row and columns {METRICS}.")
    values = [float(table.iloc[0][metric]) for metric in METRICS]
    if not all(0.0 <= value <= 1.0 for value in values):
        raise ValueError("All radar metrics must be in [0, 1].")

    angles = [2.0 * math.pi * index / len(METRICS) for index in range(len(METRICS))]
    closed_angles = angles + angles[:1]
    closed_values = values + values[:1]
    figure = plt.figure(figsize=(6.0, 6.2), dpi=DPI)
    axis = figure.add_subplot(111, polar=True)
    axis.set_theta_offset(math.pi / 2.0)
    axis.set_theta_direction(1)
    axis.set_ylim(0, 1)
    axis.set_xticks(angles, METRICS)
    axis.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    axis.plot(closed_angles, closed_values, color="#7B5DAA", linewidth=2.4)
    axis.fill(closed_angles, closed_values, color="#CFC7DB", alpha=0.55)
    for angle, value in zip(angles, values):
        axis.text(angle, min(value + 0.06, 1.04), f"{value:.3f}", ha="center", va="center")
    axis.grid(linestyle="--", alpha=0.6)
    figure.tight_layout()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / "Geographical_performance_radar_plot"
    figure.savefig(stem.with_suffix(".png"), dpi=DPI, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
