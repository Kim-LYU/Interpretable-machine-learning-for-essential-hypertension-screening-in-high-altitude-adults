"""Create a three-feature partial-dependence plot.

For every point on the three-feature grid, the selected predictors are replaced
for all derivation participants, the other predictors retain their observed
values, and predicted probabilities are averaged.  This is joint partial
dependence rather than prediction at a single median reference participant.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd


DPI = 600
GRID_POINTS_PER_AXIS = 18


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError("Derivation data must be a CSV or Excel file.")


def extract_model(loaded):
    if hasattr(loaded, "best_estimator_"):
        return loaded.best_estimator_
    if isinstance(loaded, dict):
        for key in ("best_estimator", "model", "estimator", "pipeline", "best_model"):
            if key in loaded:
                return loaded[key]
    if hasattr(loaded, "predict_proba"):
        return loaded
    raise TypeError("The model file does not contain a probabilistic estimator.")


def feature_names_from_model(model) -> list[str]:
    names = getattr(model, "feature_names_in_", None)
    if names is None and hasattr(model, "named_steps"):
        for step in model.named_steps.values():
            names = getattr(step, "feature_names_in_", None)
            if names is not None:
                break
    if names is None:
        raise AttributeError("The saved model does not expose feature_names_in_.")
    return [str(value) for value in names]


def positive_probability(model, X: pd.DataFrame) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(X), dtype=float)
    classes = getattr(model, "classes_", None)
    if classes is None and hasattr(model, "steps"):
        classes = getattr(model.steps[-1][1], "classes_", None)
    if classes is None:
        raise AttributeError("The model does not expose class labels.")
    positive = np.flatnonzero(np.asarray(classes) == 1)
    if len(positive) != 1:
        raise ValueError(f"Positive class 1 not found in {np.asarray(classes).tolist()}.")
    return probabilities[:, int(positive[0])]


def decode_names(values: np.ndarray) -> list[str]:
    return [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values.tolist()
    ]


def joint_partial_dependence(
    model,
    X: pd.DataFrame,
    selected: list[str],
    axes: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    grid_1, grid_2, grid_3 = np.meshgrid(*axes, indexing="ij")
    probabilities = np.empty(grid_1.size, dtype=float)
    base = X.copy()

    for index, values in enumerate(zip(grid_1.ravel(), grid_2.ravel(), grid_3.ravel())):
        modified = base.copy()
        for feature, value in zip(selected, values):
            modified.loc[:, feature] = value
        probabilities[index] = float(positive_probability(model, modified).mean())

    return grid_1, grid_2, grid_3, probabilities


def standardise(values: np.ndarray, reference: pd.Series) -> np.ndarray:
    scale = float(reference.std(ddof=0))
    if np.isclose(scale, 0.0):
        raise ValueError(f"Predictor has zero variance: {reference.name}")
    return (values - float(reference.mean())) / scale


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("derivation_data.csv"))
    parser.add_argument("--model", type=Path, default=Path("locked_model.joblib"))
    parser.add_argument("--artifact", type=Path, default=Path("shap_artifacts.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    model = extract_model(joblib.load(args.model))
    model_features = feature_names_from_model(model)
    data = read_table(args.data)
    missing = [feature for feature in model_features if feature not in data.columns]
    if missing:
        raise ValueError(f"Derivation data are missing model predictors: {missing}")
    X = data[model_features].apply(pd.to_numeric, errors="raise")
    if X.isna().any().any():
        raise ValueError("Derivation predictors contain missing values.")

    artifact = np.load(args.artifact, allow_pickle=False)
    required = {"feature_names", "order"}
    missing_arrays = sorted(required.difference(artifact.files))
    if missing_arrays:
        raise KeyError(f"SHAP artifact is missing arrays: {missing_arrays}")
    artifact_features = decode_names(artifact["feature_names"])
    if artifact_features != model_features:
        raise ValueError("SHAP artifact feature order does not match the locked model.")
    order = np.asarray(artifact["order"], dtype=int)
    selected = [artifact_features[index] for index in order[:3]]

    axes = [
        np.linspace(X[feature].quantile(0.05), X[feature].quantile(0.95), GRID_POINTS_PER_AXIS)
        for feature in selected
    ]
    grid_1, grid_2, grid_3, probability = joint_partial_dependence(
        model, X, selected, axes
    )

    output_table = pd.DataFrame(
        {
            selected[0]: grid_1.ravel(),
            selected[1]: grid_2.ravel(),
            selected[2]: grid_3.ravel(),
            "partial_dependence": probability,
        }
    )

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure = plt.figure(figsize=(6.4, 4.7), dpi=DPI)
    axis = figure.add_subplot(111, projection="3d")
    normalise = Normalize(vmin=float(probability.min()), vmax=float(probability.max()))
    scatter = axis.scatter(
        standardise(grid_1.ravel(), X[selected[0]]),
        standardise(grid_2.ravel(), X[selected[1]]),
        standardise(grid_3.ravel(), X[selected[2]]),
        c=probability,
        cmap="viridis",
        norm=normalise,
        s=15,
        alpha=0.95,
        linewidths=0,
        depthshade=False,
    )
    axis.set_xlabel(selected[0])
    axis.set_ylabel(selected[1])
    axis.set_zlabel(selected[2])
    axis.view_init(elev=22, azim=-58)
    colour_bar = figure.colorbar(scatter, ax=axis, fraction=0.045, pad=0.08)
    colour_bar.set_label("Predicted probability")
    figure.tight_layout()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / "3D_partial_dependence_plot"
    output_table.to_csv(stem.with_suffix(".csv"), index=False)
    figure.savefig(stem.with_suffix(".png"), dpi=DPI, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
