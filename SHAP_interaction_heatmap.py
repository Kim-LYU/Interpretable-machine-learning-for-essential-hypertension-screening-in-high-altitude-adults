"""Create a heatmap from a precomputed SHAP-interaction artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


DPI = 600


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=Path("shap_artifacts.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return parser.parse_args()


def decode_names(values: np.ndarray) -> list[str]:
    names = []
    for value in values.tolist():
        names.append(value.decode("utf-8") if isinstance(value, bytes) else str(value))
    return names


def main() -> None:
    args = parse_arguments()
    artifact = np.load(args.artifact, allow_pickle=False)
    required = {"feature_names", "interaction_mean", "order"}
    missing = sorted(required.difference(artifact.files))
    if missing:
        raise KeyError(f"SHAP artifact is missing arrays: {missing}")

    feature_names = decode_names(artifact["feature_names"])
    interaction = np.asarray(artifact["interaction_mean"], dtype=float)
    order = np.asarray(artifact["order"], dtype=int)
    expected_shape = (len(feature_names), len(feature_names))
    if interaction.shape != expected_shape:
        raise ValueError(
            f"Interaction matrix shape {interaction.shape} does not match "
            f"{len(feature_names)} feature names."
        )
    if sorted(order.tolist()) != list(range(len(feature_names))):
        raise ValueError("The stored feature order is not a complete permutation.")

    matrix = interaction[np.ix_(order, order)]
    names = [feature_names[index] for index in order]
    colour_map = LinearSegmentedColormap.from_list(
        "interaction", ["#EEE7F4", "#B28DCC", "#7B4FA0", "#D81B60"]
    )

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axis = plt.subplots(figsize=(5.2, 5.0), dpi=DPI)
    image = axis.imshow(matrix, cmap=colour_map, aspect="equal", vmin=0.0)
    axis.set_xticks(range(len(names)), labels=names, rotation=45, ha="right")
    axis.set_yticks(range(len(names)), labels=names)

    threshold = float(np.nanmax(matrix)) * 0.57
    for row in range(len(names)):
        for column in range(len(names)):
            value = matrix[row, column]
            axis.text(
                column,
                row,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=7.3,
                color="white" if value >= threshold else "#30233F",
            )

    colour_bar = figure.colorbar(image, ax=axis, fraction=0.042, pad=0.03)
    colour_bar.set_label("Mean |SHAP interaction value|")
    figure.tight_layout()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / "SHAP_interaction_heatmap"
    figure.savefig(stem.with_suffix(".png"), dpi=DPI, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
