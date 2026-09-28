"""Create a cross-cohort comparison from performance estimates."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd


METRICS = ["AUROC", "AUPRC", "Accuracy"]
EXPECTED_COHORTS = [
    "Derivation nested cross-validation",
    "Temporal external validation",
    "Geographical external validation",
]
DPI = 600


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, default=Path("results/cross_cohort_metrics.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    table = pd.read_csv(args.metrics)
    required = ["cohort", *METRICS]
    missing = [column for column in required if column not in table.columns]
    if missing:
        raise ValueError(f"Cross-cohort table is missing columns: {missing}")
    if set(table["cohort"]) != set(EXPECTED_COHORTS):
        raise ValueError(f"cohort must contain exactly: {EXPECTED_COHORTS}")
    table = table.set_index("cohort").loc[EXPECTED_COHORTS]

    angles = [2.0 * math.pi * index / len(METRICS) for index in range(len(METRICS))]
    closed_angles = angles + angles[:1]
    colours = ["#5A9CC8", "#58BFAF", "#D98996"]
    figure = plt.figure(figsize=(6.1, 4.8), dpi=DPI)
    axis = figure.add_subplot(111, polar=True)
    axis.set_theta_offset(math.pi / 2.0)
    axis.set_theta_direction(-1)
    axis.set_ylim(0.5, 1.0)
    axis.set_xticks(angles, METRICS)
    axis.set_yticks([0.5, 0.7, 0.9, 1.0])
    axis.set_yticklabels(["0.5", "0.7", "0.9", "1.0"])
    handles = []
    for cohort, colour in zip(EXPECTED_COHORTS, colours):
        values = [float(table.loc[cohort, metric]) for metric in METRICS]
        if not all(0.0 <= value <= 1.0 for value in values):
            raise ValueError(f"Metrics outside [0, 1] for {cohort}.")
        axis.plot(closed_angles, values + values[:1], color=colour, linewidth=2.0)
        axis.fill(closed_angles, values + values[:1], color=colour, alpha=0.20)
        handles.append(Patch(facecolor=colour, edgecolor=colour, alpha=0.55, label=cohort))
    axis.grid(linestyle="--", alpha=0.6)
    axis.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.48, 1.12), frameon=False)
    figure.tight_layout()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / "Cross_cohort_performance_comparison"
    figure.savefig(stem.with_suffix(".png"), dpi=DPI, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
