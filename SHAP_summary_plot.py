"""Calculate TreeSHAP values and create a global SHAP summary plot."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import shap


DPI = 600


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError("Input data must be a CSV or Excel file.")


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


def tree_estimator(model):
    if hasattr(model, "steps"):
        if len(model.steps) != 1:
            raise ValueError(
                "TreeSHAP requires the matrix entering the final tree estimator. "
                "Supply a model whose saved pipeline contains only the fitted tree step."
            )
        return model.steps[-1][1]
    return model


def positive_class_index(model) -> tuple[int, int]:
    classes = getattr(model, "classes_", None)
    if classes is None and hasattr(model, "steps"):
        classes = getattr(model.steps[-1][1], "classes_", None)
    if classes is None:
        raise AttributeError("The model does not expose class labels.")
    classes = np.asarray(classes)
    positive = np.flatnonzero(classes == 1)
    if len(positive) != 1:
        raise ValueError(f"Positive class 1 not found in {classes.tolist()}.")
    return int(positive[0]), int(len(classes))


def select_output(values, output_index: int, output_count: int, expected_ndim: int) -> np.ndarray:
    if isinstance(values, list):
        if len(values) != output_count:
            raise ValueError("Unexpected number of SHAP outputs.")
        array = np.asarray(values[output_index], dtype=float)
    else:
        array = np.asarray(values, dtype=float)
        if array.ndim == expected_ndim + 1:
            if array.shape[-1] == output_count:
                array = array[..., output_index]
            elif array.shape[0] == output_count:
                array = array[output_index]
            else:
                raise ValueError(f"Unsupported SHAP output shape: {array.shape}")
    if array.ndim != expected_ndim:
        raise ValueError(f"Unsupported SHAP output shape: {array.shape}")
    return array


def select_expected_value(value, output_index: int, output_count: int) -> float:
    values = np.asarray(value, dtype=float)
    if values.ndim == 0:
        return float(values)
    values = values.ravel()
    if values.size == output_count:
        return float(values[output_index])
    if values.size == 1:
        return float(values[0])
    raise ValueError(f"Unsupported expected-value shape: {values.shape}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("analysis_data.csv"))
    parser.add_argument("--model", type=Path, default=Path("locked_model.joblib"))
    parser.add_argument("--artifact", type=Path, default=Path("shap_artifacts.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return parser.parse_args()


def save_figure(figure: plt.Figure, stem: Path) -> None:
    figure.savefig(stem.with_suffix(".png"), dpi=DPI, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_arguments()
    model = extract_model(joblib.load(args.model))
    estimator = tree_estimator(model)
    feature_names = feature_names_from_model(model)

    frame = read_table(args.data)
    missing = [name for name in feature_names if name not in frame.columns]
    if missing:
        raise ValueError(f"Input data are missing model predictors: {missing}")
    X = frame.loc[:, feature_names].apply(pd.to_numeric, errors="raise")
    if X.isna().any().any():
        raise ValueError("Model predictors contain missing values.")

    output_index, output_count = positive_class_index(model)
    explainer = shap.TreeExplainer(estimator)
    shap_values = select_output(
        explainer.shap_values(X), output_index, output_count, expected_ndim=2
    )
    interaction_values = select_output(
        explainer.shap_interaction_values(X),
        output_index,
        output_count,
        expected_ndim=3,
    )
    if shap_values.shape != X.shape:
        raise ValueError(
            f"SHAP matrix shape {shap_values.shape} does not match input shape {X.shape}."
        )
    if interaction_values.shape != (len(X), X.shape[1], X.shape[1]):
        raise ValueError(
            "SHAP interaction values do not match the expected sample-by-feature matrix."
        )

    expected_value = select_expected_value(
        explainer.expected_value, output_index, output_count
    )
    importance = np.mean(np.abs(shap_values), axis=0)
    order = np.argsort(-importance, kind="stable")
    interaction_mean = np.mean(np.abs(interaction_values), axis=0)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.artifact,
        feature_names=np.asarray(feature_names, dtype="U"),
        feature_values=X.to_numpy(dtype=float),
        shap_values=shap_values,
        importance=importance,
        order=order,
        expected_value=np.asarray([expected_value], dtype=float),
        interaction_mean=interaction_mean,
    )
    pd.DataFrame(
        {
            "feature": np.asarray(feature_names)[order],
            "mean_absolute_shap": importance[order],
        }
    ).to_csv(args.output_dir / "SHAP_global_importance.csv", index=False)

    colour_map = LinearSegmentedColormap.from_list(
        "shap_summary", ["#0F7AE5", "#6E49B8", "#B61AA8", "#FF0E5E"]
    )
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    plt.figure(figsize=(7.0, 5.25), dpi=DPI)
    shap.summary_plot(
        shap_values,
        X,
        feature_names=feature_names,
        cmap=colour_map,
        max_display=len(feature_names),
        show=False,
    )
    figure = plt.gcf()
    axis = plt.gca()
    axis.set_xlabel("SHAP value (impact on model output)")
    axis.set_ylabel("Predictors")
    figure.tight_layout()
    save_figure(figure, args.output_dir / "SHAP_summary_plot")


if __name__ == "__main__":
    main()
