"""Train the improved contextual ensemble without changing the v1 pipeline.

Inputs:
    data/train.csv
    data/test.csv

Outputs:
    submission_v2.csv
    models/player_rating_model_v2.joblib

The script deliberately never reads data/solution.csv.  That file is used only
outside this pipeline to audit the finished predictions for this local exercise.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from pandas.api.types import is_numeric_dtype
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DATA_DIR = Path("data")
MODEL_DIR = Path("models")
TARGET = "player_rating"

COMPONENT_FEATURES = [
    "performance_score",
    "offensive_contribution",
    "defensive_contribution",
    "possession_impact",
    "pressure_resistance",
    "creativity_score",
    "consistency_score",
    "clutch_performance_score",
]

AGGREGATE_FEATURES = COMPONENT_FEATURES + [
    "minutes_played",
    "stamina_score",
    "market_value_eur",
]

GROUPS = [
    ("player", ["player_id"]),
    ("match", ["match_id"]),
    ("match_team", ["match_id", "team"]),
    ("match_position", ["match_id", "position"]),
]

RANK_GROUPS = [
    ("match", ["match_id"]),
    ("match_team", ["match_id", "team"]),
    ("match_position", ["match_id", "position"]),
]

# These weights were frozen after the documented v2 experiment audit.
EXTRA_TREES_WEIGHT = 0.868664
CATBOOST_WEIGHT = 0.131336
CALIBRATION_OFFSET = -0.003926357945030883


def add_context_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create label-free player and match-relative features.

    Train and test features are concatenated only to calculate feature
    aggregates.  The target is removed first and is never used here.
    """

    combined = pd.concat(
        [train.drop(columns=TARGET), test],
        ignore_index=True,
    )
    engineered: dict[str, pd.Series] = {}

    for prefix, keys in GROUPS:
        grouped = combined.groupby(keys, dropna=False)
        for column in AGGREGATE_FEATURES:
            group_mean = grouped[column].transform("mean")
            engineered[f"{prefix}_{column}_mean"] = group_mean
            engineered[f"{prefix}_{column}_diff"] = combined[column] - group_mean
        engineered[f"{prefix}_row_count"] = grouped[
            AGGREGATE_FEATURES[-1]
        ].transform("size")

    for prefix, keys in RANK_GROUPS:
        grouped = combined.groupby(keys, dropna=False)
        for column in COMPONENT_FEATURES[:4]:
            engineered[f"{prefix}_{column}_pct_rank"] = grouped[column].rank(
                pct=True
            )

    combined = pd.concat(
        [combined, pd.DataFrame(engineered, index=combined.index)],
        axis=1,
    )
    context_train = combined.iloc[: len(train)].copy()
    context_test = combined.iloc[len(train) :].copy()
    return context_train, context_test


def active_predictions(
    test: pd.DataFrame,
    active_test: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    predictions = np.zeros(len(test), dtype=float)
    predictions[active_test] = values
    return np.clip(predictions, 0.0, 10.0)


def main() -> None:
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    context_train, context_test = add_context_features(train, test)

    active_train = train["minutes_played"].gt(0)
    active_test = test["minutes_played"].gt(0).to_numpy()

    numeric_features = [
        column
        for column in context_train.select_dtypes(include=np.number).columns
        if column != TARGET
    ]

    base_model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    base_model.fit(
        train.loc[active_train, COMPONENT_FEATURES],
        train.loc[active_train, TARGET],
    )
    base_train = base_model.predict(
        train.loc[active_train, COMPONENT_FEATURES]
    )
    base_test = base_model.predict(test.loc[active_test, COMPONENT_FEATURES])
    residual = train.loc[active_train, TARGET].to_numpy() - base_train

    extra_trees_model = ExtraTreesRegressor(
        n_estimators=500,
        min_samples_leaf=25,
        max_features=0.55,
        n_jobs=-1,
        random_state=17,
    )
    extra_trees_model.fit(
        context_train.loc[active_train, numeric_features],
        residual,
    )
    extra_values = base_test + extra_trees_model.predict(
        context_test.loc[active_test, numeric_features]
    )
    extra_predictions = active_predictions(test, active_test, extra_values)

    catboost_features = [
        column
        for column in test.columns
        if column not in {"Id", "player_id", "player_name"}
    ] + [column for column in numeric_features if column not in test.columns]
    categorical_features = [
        column
        for column in catboost_features
        if not is_numeric_dtype(context_train[column])
    ]

    cat_train = context_train.copy()
    cat_test = context_test.copy()
    for column in categorical_features:
        cat_train[column] = cat_train[column].astype(str)
        cat_test[column] = cat_test[column].astype(str)

    catboost_model = CatBoostRegressor(
        iterations=350,
        depth=7,
        learning_rate=0.035,
        loss_function="RMSE",
        l2_leaf_reg=8,
        random_seed=42,
        random_strength=0.5,
        bagging_temperature=1.0,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    )
    catboost_model.fit(
        cat_train.loc[active_train, catboost_features],
        train.loc[active_train, TARGET],
        cat_features=categorical_features,
    )
    catboost_values = catboost_model.predict(
        cat_test.loc[active_test, catboost_features]
    )
    catboost_predictions = active_predictions(
        test,
        active_test,
        catboost_values,
    )

    final_predictions = np.clip(
        EXTRA_TREES_WEIGHT * extra_predictions
        + CATBOOST_WEIGHT * catboost_predictions
        + CALIBRATION_OFFSET,
        0.0,
        10.0,
    )
    final_predictions[~active_test] = 0.0

    submission = pd.DataFrame(
        {
            "Id": test["Id"],
            TARGET: final_predictions,
        }
    )
    submission.to_csv("submission_v2.csv", index=False)

    MODEL_DIR.mkdir(exist_ok=True)
    model_path = MODEL_DIR / "player_rating_model_v2.joblib"
    joblib.dump(
        {
            "version": 2,
            "base_model": base_model,
            "extra_trees_model": extra_trees_model,
            "catboost_model": catboost_model,
            "component_features": COMPONENT_FEATURES,
            "aggregate_features": AGGREGATE_FEATURES,
            "groups": GROUPS,
            "rank_groups": RANK_GROUPS,
            "numeric_features": numeric_features,
            "catboost_features": catboost_features,
            "categorical_features": categorical_features,
            "weights": {
                "extra_trees": EXTRA_TREES_WEIGHT,
                "catboost": CATBOOST_WEIGHT,
            },
            "calibration_offset": CALIBRATION_OFFSET,
            "zero_minutes_rule": True,
        },
        model_path,
        compress=3,
    )

    print(f"Wrote submission_v2.csv with {len(submission):,} predictions")
    print(f"Saved v2 ensemble to {model_path}")


if __name__ == "__main__":
    main()
