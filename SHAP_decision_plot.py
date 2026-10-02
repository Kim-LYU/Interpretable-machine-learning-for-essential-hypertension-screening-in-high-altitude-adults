"""Create a SHAP decision plot from a previously generated SHAP artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np


DPI = 600


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=Path("shap_artifacts.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return parser.parse_args()


def decode_names(values: np.ndarray) -> list[str]:
    return [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values.tolist()
    ]


def main() -> None:
    args = parse_arguments()
    artifact = np.load(args.artifact, allow_pickle=False)
    required = {"feature_names", "shap_values", "order", "expected_value"}
    missing = sorted(required.difference(artifact.files))
    if missing:
        raise KeyError(f"SHAP artifact is missing arrays: {missing}")

    feature_names = decode_names(artifact["feature_names"])
    shap_values = np.asarray(artifact["shap_values"], dtype=float)
    order = np.asarray(artifact["order"], dtype=int)
    expected_values = np.asarray(artifact["expected_value"], dtype=float).ravel()

    if shap_values.ndim != 2 or shap_values.shape[1] != len(feature_names):
        raise ValueError("SHAP values do not match the stored feature names.")
    if sorted(order.tolist()) != list(range(len(feature_names))):
        raise ValueError("The stored feature order is not a complete permutation.")
    if expected_values.size != 1:
        raise ValueError("The SHAP artifact must contain one expected value.")

    base_value = float(expected_values[0])
    ordered_values = shap_values[:, order]
    ranked_names = [feature_names[index] for index in order]

    cumulative = base_value + np.cumsum(ordered_values[:, ::-1], axis=1)
    cumulative = np.column_stack(
        [np.full(len(ordered_values), base_value), cumulative]
    )
    final_output = cumulative[:, -1]
    lower, upper = np.percentile(final_output, [2, 98])
    if np.isclose(lower, upper):
        lower, upper = float(final_output.min()), float(final_output.max())
    if np.isclose(lower, upper):
        lower, upper = lower - 0.5, upper + 0.5

    colour_map = LinearSegmentedColormap.from_list(
        "shap_decision", ["#0B84F3", "#5A56C5", "#A61FB8", "#FF0A5C"]
    )
    normalise = Normalize(vmin=lower, vmax=upper)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    figure, axis = plt.subplots(figsize=(7.0, 5.25), dpi=DPI)
    levels = np.arange(len(ranked_names) + 1)
    for row, value in zip(cumulative, final_output):
        axis.plot(
            row,
            levels,
            color=colour_map(normalise(value)),
            alpha=0.35,
            linewidth=0.65,
        )

    axis.axvline(base_value, color="#888888", linestyle="--", linewidth=0.8)
    axis.set_yticks(levels, labels=["Base value", *ranked_names[::-1]])
    axis.set_xlabel("Cumulative SHAP contribution")
    axis.set_ylabel("Predictors")
    axis.grid(True, color="#DDDDDD", linestyle="--", linewidth=0.5, alpha=0.7)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    colour_scale = plt.cm.ScalarMappable(cmap=colour_map, norm=normalise)
    colour_scale.set_array([])
    colour_bar = figure.colorbar(colour_scale, ax=axis, fraction=0.035, pad=0.02)
    colour_bar.set_label("Final model output")
    figure.tight_layout()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / "SHAP_decision_plot"
    figure.savefig(stem.with_suffix(".png"), dpi=DPI, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
