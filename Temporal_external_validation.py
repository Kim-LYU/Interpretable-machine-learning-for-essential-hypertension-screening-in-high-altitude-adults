"""Visualise temporal external-validation results from de-identified tables.

This script is figure-only: it does not refit the locked model. Confidence
intervals and subgroup O/E estimates must be supplied from the prespecified
evaluation workflow (including the reported 2,000-resample BCa procedure).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve


DPI = 600


def require_columns(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def read_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    require_columns(frame, ["y_true", "y_prob"], path.name)
    frame = frame[["y_true", "y_prob"]].apply(pd.to_numeric, errors="raise")
    if set(frame["y_true"].astype(int).unique()) != {0, 1}:
        raise ValueError(f"{path.name}: y_true must contain both 0 and 1.")
    if not frame["y_prob"].between(0, 1).all():
        raise ValueError(f"{path.name}: y_prob must be in [0, 1].")
    return frame


def metric_value(metrics: pd.DataFrame, metric: str, field: str = "estimate") -> float:
    row = metrics.loc[metrics["metric"] == metric]
    if len(row) != 1:
        raise ValueError(f"Expected one row for metric '{metric}'.")
    return float(row.iloc[0][field])


def panel_roc(axis, temporal: pd.DataFrame, derivation: pd.DataFrame, metrics: pd.DataFrame) -> None:
    for frame, label, colour in (
        (temporal, "Temporal external validation", "#D95F5F"),
        (derivation, "Derivation nested-CV OOF", "#4C78A8"),
    ):
        false_positive, true_positive, _ = roc_curve(frame["y_true"], frame["y_prob"])
        axis.plot(false_positive, true_positive, lw=1.8, color=colour, label=label)
    axis.plot([0, 1], [0, 1], "--", color="#999999", lw=1.0)
    estimate = metric_value(metrics, "AUROC")
    low = metric_value(metrics, "AUROC", "ci_low")
    high = metric_value(metrics, "AUROC", "ci_high")
    axis.text(0.98, 0.04, f"Temporal AUROC {estimate:.3f} ({low:.3f}-{high:.3f})", ha="right")
    axis.set(xlabel="1 - Specificity", ylabel="Sensitivity", xlim=(0, 1), ylim=(0, 1))
    axis.legend(frameon=False, loc="lower right")


def panel_pr(axis, temporal: pd.DataFrame, derivation: pd.DataFrame, metrics: pd.DataFrame) -> None:
    for frame, label, colour in (
        (temporal, "Temporal external validation", "#D95F5F"),
        (derivation, "Derivation nested-CV OOF", "#4C78A8"),
    ):
        precision, recall, _ = precision_recall_curve(frame["y_true"], frame["y_prob"])
        axis.plot(recall, precision, lw=1.8, color=colour, label=label)
    estimate = metric_value(metrics, "AUPRC")
    low = metric_value(metrics, "AUPRC", "ci_low")
    high = metric_value(metrics, "AUPRC", "ci_high")
    axis.text(0.02, 0.04, f"Temporal AUPRC {estimate:.3f} ({low:.3f}-{high:.3f})")
    axis.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1))
    axis.legend(frameon=False, loc="lower left")


def panel_dca(axis, dca: pd.DataFrame) -> None:
    require_columns(dca, ["threshold", "model", "treat_all", "treat_none"], "DCA table")
    axis.plot(dca["threshold"], dca["model"], label="Locked model", lw=1.8)
    axis.plot(dca["threshold"], dca["treat_all"], label="Treat all", lw=1.4)
    axis.plot(dca["threshold"], dca["treat_none"], label="Treat none", lw=1.2, color="#777777")
    axis.set(xlabel="Threshold probability", ylabel="Net benefit")
    axis.legend(frameon=False)


def panel_calibration(
    axis,
    calibration: pd.DataFrame,
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
) -> None:
    require_columns(calibration, ["mean_predicted_probability", "observed_fraction"], "Calibration table")
    axis.plot([0, 1], [0, 1], "--", color="#777777", lw=1.0, label="Ideal")
    axis.plot(
        calibration["mean_predicted_probability"],
        calibration["observed_fraction"],
        marker="o",
        lw=1.8,
        label="Temporal external validation",
    )
    histogram_axis = axis.twinx()
    histogram_axis.hist(
        predictions.loc[predictions["y_true"] == 0, "y_prob"],
        bins=20,
        range=(0, 1),
        alpha=0.25,
        color="#4C78A8",
        label="No event",
    )
    histogram_axis.hist(
        predictions.loc[predictions["y_true"] == 1, "y_prob"],
        bins=20,
        range=(0, 1),
        alpha=0.25,
        color="#E45756",
        label="Event",
    )
    histogram_axis.set_ylabel("Count")
    axis.text(0.03, 0.92, f"Brier score = {metric_value(metrics, 'Brier score'):.3f}")
    axis.set(xlabel="Predicted probability", ylabel="Observed proportion", xlim=(0, 1), ylim=(0, 1))
    axis.legend(frameon=False, loc="lower right")


def panel_subgroup_auroc(axis, subgroup: pd.DataFrame) -> None:
    require_columns(subgroup, ["subgroup", "n", "auroc", "auroc_low", "auroc_high"], "Subgroup table")
    plot = subgroup.iloc[::-1].reset_index(drop=True)
    positions = np.arange(len(plot))
    errors = np.vstack([plot["auroc"] - plot["auroc_low"], plot["auroc_high"] - plot["auroc"]])
    axis.errorbar(plot["auroc"], positions, xerr=errors, fmt="o", color="#4C78A8", capsize=3)
    overall = float(subgroup.loc[subgroup["subgroup"] == "Overall", "auroc"].iloc[0])
    axis.axvline(overall, linestyle="--", color="#999999", lw=1.0)
    axis.set_yticks(positions, [f"{name} (n={int(n)})" for name, n in zip(plot["subgroup"], plot["n"])])
    axis.set_xlabel("AUROC (95% CI)")
    axis.set_xlim(0.5, 1.0)


def panel_subgroup_oe(axis, subgroup: pd.DataFrame) -> None:
    require_columns(subgroup, ["subgroup", "n", "oe", "oe_low", "oe_high"], "Subgroup table")
    plot = subgroup.iloc[::-1].reset_index(drop=True)
    positions = np.arange(len(plot))
    errors = np.vstack([plot["oe"] - plot["oe_low"], plot["oe_high"] - plot["oe"]])
    axis.errorbar(plot["oe"], positions, xerr=errors, fmt="s", color="#7A5195", capsize=3)
    axis.axvline(1.0, linestyle="--", color="#777777", lw=1.0)
    axis.set_yticks(positions, [f"{name} (n={int(n)})" for name, n in zip(plot["subgroup"], plot["n"])])
    axis.set_xlabel("Observed-to-expected ratio (95% CI)")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    temporal = read_predictions(args.input_dir / "temporal_predictions.csv")
    derivation = read_predictions(args.input_dir / "derivation_oof_predictions.csv")
    metrics = pd.read_csv(args.input_dir / "temporal_metrics.csv")
    require_columns(metrics, ["metric", "estimate", "ci_low", "ci_high"], "Metrics table")
    dca = pd.read_csv(args.input_dir / "temporal_dca.csv")
    calibration = pd.read_csv(args.input_dir / "temporal_calibration.csv")
    subgroup = pd.read_csv(args.input_dir / "temporal_subgroup_metrics.csv")

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "pdf.fonttype": 42})
    figure, axes = plt.subplots(2, 3, figsize=(14.0, 8.8), dpi=DPI)
    panel_roc(axes[0, 0], temporal, derivation, metrics)
    panel_pr(axes[0, 1], temporal, derivation, metrics)
    panel_dca(axes[0, 2], dca)
    panel_calibration(axes[1, 0], calibration, temporal, metrics)
    panel_subgroup_auroc(axes[1, 1], subgroup)
    panel_subgroup_oe(axes[1, 2], subgroup)
    for label, axis in zip("abcdef", axes.ravel()):
        axis.text(-0.16, 1.06, label, transform=axis.transAxes, fontsize=15, fontweight="bold")
        axis.grid(alpha=0.2)
    figure.tight_layout()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / "Temporal_external_validation"
    figure.savefig(stem.with_suffix(".png"), dpi=DPI, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
