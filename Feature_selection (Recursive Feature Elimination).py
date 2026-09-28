"""Random-forest recursive feature elimination in the derivation cohort."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold


EXPECTED_PREDICTORS = 19
CV_FOLDS = 5
RANDOM_STATE = 42


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError("Input must be a CSV or Excel file.")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("derivation_19_predictors.csv"))
    parser.add_argument("--outcome", default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    data = read_table(args.input)
    outcome = args.outcome or str(data.columns[-1])
    if outcome not in data.columns:
        raise KeyError(f"Outcome column not found: {outcome}")

    predictors = [column for column in data.columns if column != outcome]
    if len(predictors) != EXPECTED_PREDICTORS:
        raise ValueError(f"Expected 19 predictors, found {len(predictors)}.")
    if data[predictors].isna().any().any():
        raise ValueError("Missing values remain in the feature-selection input.")

    X = data[predictors].astype(float)
    y = pd.to_numeric(data[outcome], errors="raise").astype(int)
    if set(y.unique()) != {0, 1}:
        raise ValueError("The outcome must be coded 0 and 1.")

    estimator = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    selector = RFECV(
        estimator=estimator,
        step=1,
        min_features_to_select=1,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
    )
    selector.fit(X, y)

    feature_table = pd.DataFrame(
        {
            "predictor": predictors,
            "selected": selector.support_,
            "ranking": selector.ranking_,
        }
    ).sort_values(["ranking", "predictor"], kind="mergesort")

    score_table = pd.DataFrame(
        {
            "number_of_selected_predictors": selector.cv_results_["n_features"],
            "mean_cv_auroc": selector.cv_results_["mean_test_score"],
            "sd_cv_auroc": selector.cv_results_["std_test_score"],
        }
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    feature_table.to_csv(args.output_dir / "rf_rfe_selected_features.csv", index=False)
    score_table.to_csv(args.output_dir / "rf_rfe_cross_validation.csv", index=False)


if __name__ == "__main__":
    main()
