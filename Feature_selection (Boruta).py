"""Boruta feature selection for the analysis-ready derivation dataset.

The input table must contain the 19 numeric predictors retained after the
prespecified derivation-cohort preprocessing and Spearman correlation filter,
followed by a binary outcome column coded as 0 and 1. Missing-value imputation
must be completed before this script is run.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier


DATA_FILE = Path("data.csv")
OUTPUT_DIR = Path("outputs")

RANDOM_STATE = 42
MAX_ITERATIONS = 500
SIGNIFICANCE_LEVEL = 0.05
EXPECTED_PREDICTOR_COUNT = 19


def load_analysis_data(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load numeric predictors and a binary outcome from the final column."""
    if path.suffix.lower() == ".csv":
        data = pd.read_csv(path)
    elif path.suffix.lower() in {".xlsx", ".xls"}:
        data = pd.read_excel(path)
    else:
        raise ValueError("The input file must be a CSV or Excel file.")

    if data.shape[1] < 2:
        raise ValueError("The input table must contain predictors and an outcome column.")

    predictors = data.iloc[:, :-1].copy()
    outcome = data.iloc[:, -1].copy()

    if predictors.shape[1] != EXPECTED_PREDICTOR_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_PREDICTOR_COUNT} predictors after correlation "
            f"filtering, but found {predictors.shape[1]}."
        )

    if predictors.columns.duplicated().any():
        raise ValueError("Predictor names must be unique.")

    non_numeric = predictors.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        raise ValueError(f"All predictors must be numeric. Non-numeric columns: {non_numeric}")

    if predictors.isna().any().any() or outcome.isna().any():
        raise ValueError(
            "Missing values were found. Apply the prespecified derivation-cohort "
            "imputation procedure before running feature selection."
        )

    outcome_values = set(pd.unique(outcome))
    if outcome_values != {0, 1}:
        raise ValueError("The outcome must contain both binary classes coded as 0 and 1.")

    return predictors, outcome.astype(int)


def run_boruta(predictors: pd.DataFrame, outcome: pd.Series) -> pd.DataFrame:
    """Run BorutaPy and return the feature decisions and ranks."""
    estimator = RandomForestClassifier(
        n_jobs=-1,
        max_depth=None,
        random_state=RANDOM_STATE,
    )

    selector = BorutaPy(
        estimator=estimator,
        n_estimators="auto",
        perc=100,
        alpha=SIGNIFICANCE_LEVEL,
        two_step=True,
        max_iter=MAX_ITERATIONS,
        random_state=RANDOM_STATE,
        verbose=2,
    )
    selector.fit(predictors.to_numpy(), outcome.to_numpy())

    decisions = np.select(
        [selector.support_, selector.support_weak_],
        ["Confirmed", "Tentative"],
        default="Rejected",
    )

    return (
        pd.DataFrame(
            {
                "feature": predictors.columns,
                "decision": decisions,
                "rank": selector.ranking_,
            }
        )
        .sort_values(["rank", "feature"])
        .reset_index(drop=True)
    )


def main() -> None:
    predictors, outcome = load_analysis_data(DATA_FILE)
    results = run_boruta(predictors, outcome)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / "boruta_feature_selection.csv"
    results.to_csv(output_file, index=False)

    print(results.to_string(index=False))
    print(f"Saved results to: {output_file}")


if __name__ == "__main__":
    main()
