"""Create individual classification profiles for external validation.

Predicted classes are derived only from the prespecified probability threshold
of 0.5. The script consumes de-identified outcome/probability tables and does
not require participant identifiers or source clinical files.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


THRESHOLD = 0.5
DPI = 600
COLOURS = ["#A8DCC4", "#F8C0A8"]


def read_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = [column for column in ["y_true", "y_prob"] if column not in frame.columns]
    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing}")
    frame = frame[["y_true", "y_prob"]].apply(pd.to_numeric, errors="raise")
    if set(frame["y_true"].astype(int).unique()) != {0, 1}:
        raise ValueError(f"{path.name}: y_true must contain both 0 and 1.")
    if not frame["y_prob"].between(0, 1).all():
        raise ValueError(f"{path.name}: y_prob must be in [0, 1].")
    return frame


def add_confusion_matrix(axis, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    inset = axis.inset_axes([0.67, 0.52, 0.27, 0.36])
    inset.set_xlim(0, 2)
    inset.set_ylim(0, 2)
    inset.invert_yaxis()
    inset.set_aspect("equal")
    for row in range(2):
        for column in range(2):
            inset.add_patch(
                Rectangle((column, row), 1, 1, facecolor=COLOURS[column], edgecolor="white", alpha=0.85)
            )
            inset.text(column + 0.5, row + 0.5, str(matrix[row, column]), ha="center", va="center", fontweight="bold")
    inset.set_xticks([0.5, 1.5], ["No event", "Event"])
    inset.set_yticks([0.5, 1.5], ["No event", "Event"])
    inset.set_xlabel("Predicted")
    inset.set_ylabel("Observed")
    inset.tick_params(length=0, labelsize=7)
    for spine in inset.spines.values():
        spine.set_visible(False)


def draw_profile(axis, predictions: pd.DataFrame, title: str) -> None:
    y_true = predictions["y_true"].astype(int).to_numpy()
    y_prob = predictions["y_prob"].to_numpy(dtype=float)
    y_pred = (y_prob >= THRESHOLD).astype(int)
    order = np.argsort(y_prob, kind="mergesort")
    sorted_true = y_true[order]
    sorted_pred = y_pred[order]
    sorted_probability = y_prob[order]

    class_map = ListedColormap(COLOURS)
    predicted_strip = axis.inset_axes([0.08, 0.88, 0.84, 0.055])
    observed_strip = axis.inset_axes([0.08, 0.80, 0.84, 0.055])
    for strip, values, label in (
        (predicted_strip, sorted_pred, "Predicted class"),
        (observed_strip, sorted_true, "Observed class"),
    ):
        strip.imshow(values[None, :], aspect="auto", cmap=class_map, vmin=0, vmax=1, interpolation="nearest")
        strip.set_xticks([])
        strip.set_yticks([])
        strip.set_ylabel(label, rotation=0, ha="right", va="center", labelpad=8, fontsize=8)
        for spine in strip.spines.values():
            spine.set_visible(False)

    participant_axis = axis.inset_axes([0.08, 0.08, 0.84, 0.62])
    x = np.arange(len(sorted_probability))
    participant_axis.scatter(x[sorted_true == 0], sorted_probability[sorted_true == 0], s=9, color=COLOURS[0], label="No event")
    participant_axis.scatter(x[sorted_true == 1], sorted_probability[sorted_true == 1], s=9, color=COLOURS[1], label="Event")
    participant_axis.axhline(THRESHOLD, linestyle="--", color="#777777", linewidth=1.0)
    participant_axis.set(xlabel="Participants ordered by predicted probability", ylabel="Predicted probability", ylim=(0, 1))
    participant_axis.legend(frameon=False, loc="upper left")
    add_confusion_matrix(participant_axis, y_true, y_pred)
    axis.set_title(title, loc="left", fontweight="bold")
    axis.set_axis_off()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--temporal", type=Path, default=Path("results/temporal_predictions.csv"))
    parser.add_argument("--geographical", type=Path, default=Path("results/geographical_predictions.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    temporal = read_predictions(args.temporal)
    geographical = read_predictions(args.geographical)
    figure, axes = plt.subplots(1, 2, figsize=(14.0, 5.4), dpi=DPI)
    draw_profile(axes[0], temporal, "Temporal external validation")
    draw_profile(axes[1], geographical, "Geographical external validation")
    figure.tight_layout()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / "Individual_classification_profiles"
    figure.savefig(stem.with_suffix(".png"), dpi=DPI, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
