"""Genetic-algorithm feature selection in the derivation cohort.

The chromosome contains one binary inclusion flag per predictor.  Fitness is
the sum of univariate mutual-information scores minus a fixed size penalty,
matching the analytical logic of the study code while removing study-specific
paths, file names, and predictor names.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif


EXPECTED_PREDICTORS = 19
RANDOM_STATE = 200


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError("Input must be a CSV or Excel file.")


def initialise_population(
    rng: np.random.Generator,
    population_size: int,
    chromosome_length: int,
) -> np.ndarray:
    population = rng.integers(0, 2, size=(population_size, chromosome_length), dtype=np.int8)
    empty = np.flatnonzero(population.sum(axis=1) == 0)
    for row in empty:
        population[row, rng.integers(0, chromosome_length)] = 1
    return population


def fitness(population: np.ndarray, information: np.ndarray, penalty: float) -> np.ndarray:
    return population @ information - penalty * population.sum(axis=1)


def tournament_select(
    rng: np.random.Generator,
    population: np.ndarray,
    scores: np.ndarray,
    tournament_size: int = 3,
) -> np.ndarray:
    selected = np.empty_like(population)
    for index in range(len(population)):
        contestants = rng.integers(0, len(population), size=tournament_size)
        winner = contestants[int(np.argmax(scores[contestants]))]
        selected[index] = population[winner]
    return selected


def evolve(
    information: np.ndarray,
    population_size: int,
    generations: int,
    crossover_probability: float,
    mutation_probability: float,
    bit_flip_probability: float,
    penalty: float,
) -> tuple[np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(RANDOM_STATE)
    population = initialise_population(rng, population_size, len(information))
    history: list[dict[str, float]] = []

    for generation in range(generations):
        scores = fitness(population, information, penalty)
        elite = population[int(np.argmax(scores))].copy()
        parents = tournament_select(rng, population, scores)
        offspring = parents.copy()

        for first in range(0, population_size - 1, 2):
            second = first + 1
            if rng.random() < crossover_probability and len(information) > 2:
                cut_points = np.sort(rng.choice(np.arange(1, len(information)), size=2, replace=False))
                left, right = int(cut_points[0]), int(cut_points[1])
                segment = offspring[first, left:right].copy()
                offspring[first, left:right] = offspring[second, left:right]
                offspring[second, left:right] = segment

        for row in range(population_size):
            if rng.random() < mutation_probability:
                flips = rng.random(len(information)) < bit_flip_probability
                offspring[row, flips] = 1 - offspring[row, flips]
            if offspring[row].sum() == 0:
                offspring[row, rng.integers(0, len(information))] = 1

        offspring_scores = fitness(offspring, information, penalty)
        offspring[int(np.argmin(offspring_scores))] = elite
        population = offspring
        current_scores = fitness(population, information, penalty)
        history.append(
            {
                "generation": generation + 1,
                "best_fitness": float(current_scores.max()),
                "mean_fitness": float(current_scores.mean()),
            }
        )

    final_scores = fitness(population, information, penalty)
    best = population[int(np.argmax(final_scores))]
    return best, pd.DataFrame(history)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("derivation_19_predictors.csv"))
    parser.add_argument("--outcome", default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--population-size", type=int, default=50)
    parser.add_argument("--generations", type=int, default=40)
    parser.add_argument("--crossover-probability", type=float, default=0.70)
    parser.add_argument("--mutation-probability", type=float, default=0.20)
    parser.add_argument("--bit-flip-probability", type=float, default=0.05)
    parser.add_argument("--size-penalty", type=float, default=0.03)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    data = read_table(args.input)
    outcome = args.outcome or str(data.columns[-1])
    predictors = [column for column in data.columns if column != outcome]
    if len(predictors) != EXPECTED_PREDICTORS:
        raise ValueError(f"Expected 19 predictors, found {len(predictors)}.")
    if data[predictors].isna().any().any():
        raise ValueError("Missing values remain in the feature-selection input.")

    predictor_frame = data[predictors]
    discrete_mask = np.array(
        [
            pd.api.types.is_integer_dtype(predictor_frame[column])
            and predictor_frame[column].nunique() <= 10
            for column in predictors
        ],
        dtype=bool,
    )
    X = predictor_frame.astype(float)
    y = pd.to_numeric(data[outcome], errors="raise").astype(int)
    if set(y.unique()) != {0, 1}:
        raise ValueError("The outcome must be coded 0 and 1.")

    information = mutual_info_classif(
        X,
        y,
        discrete_features=discrete_mask,
        random_state=RANDOM_STATE,
    )
    best, history = evolve(
        information,
        args.population_size,
        args.generations,
        args.crossover_probability,
        args.mutation_probability,
        args.bit_flip_probability,
        args.size_penalty,
    )

    feature_table = pd.DataFrame(
        {
            "predictor": predictors,
            "mutual_information": information,
            "selected": best.astype(bool),
        }
    ).sort_values(["selected", "mutual_information"], ascending=[False, False])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    feature_table.to_csv(args.output_dir / "ga_selected_features.csv", index=False)
    history.to_csv(args.output_dir / "ga_optimisation_history.csv", index=False)


if __name__ == "__main__":
    main()
