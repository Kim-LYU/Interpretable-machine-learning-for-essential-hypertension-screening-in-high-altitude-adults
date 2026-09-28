"""LASSO feature selection in the derivation cohort using 10-fold CV.

The input is the 19-predictor, imputed derivation dataset produced after the
manuscript-specified Spearman correlation filter.  Scaling is fitted inside
each cross-validation training fold through a scikit-learn pipeline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


EXPECTED_PREDICTORS = 19
CV_FOLDS = 10
RANDOM_STATE = 42
C_VALUES = np.logspace(-4, 4, 100)


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError("Input must be a CSV or Excel file.")


def prepare_data(data: pd.DataFrame, outcome: str) -> tuple[pd.DataFrame, pd.Series]:
    if outcome not in data.columns:
        raise KeyError(f"Outcome column not found: {outcome}")
    predictors = [column for column in data.columns if column != outcome]
    if len(predictors) != EXPECTED_PREDICTORS:
        raise ValueError(f"Expected 19 predictors, found {len(predictors)}.")
    if data[predictors].isna().any().any():
        raise ValueError("Missing values remain in the feature-selection input.")
    y = pd.to_numeric(data[outcome], errors="raise").astype(int)
    if set(y.unique()) != {0, 1}:
        raise ValueError("The outcome must be coded 0 and 1.")
    return data[predictors].astype(float), y


def fit_lasso_cv(X: pd.DataFrame, y: pd.Series) -> tuple[GridSearchCV, pd.DataFrame]:
    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "lasso",
                LogisticRegression(
                    penalty="l1",
                    solver="liblinear",
                    max_iter=10000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        pipeline,
        param_grid={"lasso__C": C_VALUES},
        scoring="neg_log_loss",
        cv=cv,
        refit=True,
        n_jobs=-1,
        return_train_score=False,
    )
    search.fit(X, y)

    results = pd.DataFrame(search.cv_results_)
    results["C"] = results["param_lasso__C"].astype(float)
    results["lambda"] = 1.0 / results["C"]
    results["mean_log_loss"] = -results["mean_test_score"]
    results["se_log_loss"] = results["std_test_score"] / np.sqrt(CV_FOLDS)
    return search, results.sort_values("lambda", ascending=False).reset_index(drop=True)


def select_one_standard_error(results: pd.DataFrame) -> float:
    best_index = int(results["mean_log_loss"].idxmin())
    cutoff = float(
        results.loc[best_index, "mean_log_loss"]
        + results.loc[best_index, "se_log_loss"]
    )
    eligible = results.loc[results["mean_log_loss"] <= cutoff]
    return float(eligible["C"].min())  # strongest eligible L1 penalty


def fit_at_c(X: pd.DataFrame, y: pd.Series, c_value: float) -> Pipeline:
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "lasso",
                LogisticRegression(
                    penalty="l1",
                    solver="liblinear",
                    C=c_value,
                    max_iter=10000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    return model.fit(X, y)


def coefficient_table(
    model_min: Pipeline,
    model_1se: Pipeline,
    predictors: list[str],
) -> pd.DataFrame:
    coef_min = model_min.named_steps["lasso"].coef_.ravel()
    coef_1se = model_1se.named_steps["lasso"].coef_.ravel()
    return pd.DataFrame(
        {
            "predictor": predictors,
            "coefficient_lambda_min": coef_min,
            "selected_lambda_min": ~np.isclose(coef_min, 0.0),
            "coefficient_lambda_1se": coef_1se,
            "selected_lambda_1se": ~np.isclose(coef_1se, 0.0),
        }
    )


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
    X, y = prepare_data(data, outcome)
    search, cv_results = fit_lasso_cv(X, y)

    c_min = float(search.best_params_["lasso__C"])
    c_1se = select_one_standard_error(cv_results)
    model_min = fit_at_c(X, y, c_min)
    model_1se = fit_at_c(X, y, c_1se)
    coefficients = coefficient_table(model_min, model_1se, X.columns.tolist())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cv_results[["C", "lambda", "mean_log_loss", "se_log_loss"]].to_csv(
        args.output_dir / "lasso_cross_validation_curve.csv", index=False
    )
    coefficients.to_csv(args.output_dir / "lasso_selected_features.csv", index=False)
    metadata = {
        "cross_validation_folds": CV_FOLDS,
        "random_state": RANDOM_STATE,
        "C_lambda_min": c_min,
        "lambda_min": 1.0 / c_min,
        "C_lambda_1se": c_1se,
        "lambda_1se": 1.0 / c_1se,
    }
    (args.output_dir / "lasso_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
