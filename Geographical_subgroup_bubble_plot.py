"""Create a subgroup bubble plot from geographical external predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score


THRESHOLD = 0.5
DPI = 600


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("results/geographical_subgroup_input.csv"))
    parser.add_argument("--female-code", default="0")
    parser.add_argument("--male-code", default="1")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return parser.parse_args()


def subgroup_metrics(data: pd.DataFrame, female_code: str, male_code: str) -> pd.DataFrame:
    sex = data["sex"].astype(str)
    definitions = [
        ("Female", sex == female_code),
        ("Male", sex == male_code),
        ("Age <60 years", data["age"] < 60),
        ("Age >=60 years", data["age"] >= 60),
        ("Overall", pd.Series(True, index=data.index)),
    ]
    records = []
    for name, mask in definitions:
        subset = data.loc[mask]
        if subset.empty or subset["y_true"].nunique() != 2:
            raise ValueError(f"Subgroup cannot support AUROC estimation: {name}")
        predicted = (subset["y_prob"] >= THRESHOLD).astype(int)
        records.append(
            {
                "subgroup": name,
                "n": len(subset),
                "auroc": roc_auc_score(subset["y_true"], subset["y_prob"]),
                "accuracy": accuracy_score(subset["y_true"], predicted),
            }
        )
    return pd.DataFrame(records)


def main() -> None:
    args = parse_arguments()
    data = pd.read_csv(args.input)
    required = ["y_true", "y_prob", "age", "sex"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Subgroup input is missing columns: {missing}")
    for column in ["y_true", "y_prob", "age"]:
        data[column] = pd.to_numeric(data[column], errors="raise")
    if set(data["y_true"].astype(int).unique()) != {0, 1}:
        raise ValueError("y_true must contain both 0 and 1.")
    metrics = subgroup_metrics(data, args.female_code, args.male_code)

    figure, axis = plt.subplots(figsize=(8.8, 5.8), dpi=DPI)
    positions = np.arange(len(metrics))[::-1]
    scatter = axis.scatter(
        metrics["auroc"],
        positions,
        s=metrics["n"] * 7.0,
        c=metrics["accuracy"],
        cmap="viridis",
        edgecolors="#555555",
        alpha=0.75,
    )
    for position, (_, row) in zip(positions, metrics.iterrows()):
        axis.text(row["auroc"], position, str(int(row["n"])), ha="center", va="center")
    axis.set_yticks(positions, metrics["subgroup"])
    axis.set_xlabel("AUROC")
    axis.grid(linestyle="--", alpha=0.4)
    colour_bar = figure.colorbar(scatter, ax=axis)
    colour_bar.set_label("Accuracy at threshold 0.5")
    figure.tight_layout()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_dir / "Geographical_subgroup_metrics.csv", index=False)
    stem = args.output_dir / "Geographical_subgroup_bubble_plot"
    figure.savefig(stem.with_suffix(".png"), dpi=DPI, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
