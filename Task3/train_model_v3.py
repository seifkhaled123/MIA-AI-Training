"""Train the leakage-free v3 player-rating ensemble.

This version removes ``performance_score`` immediately after loading the raw
CSV files.  It is therefore unavailable to feature engineering and to every
model.  Validation must hold out complete players because train and test have
disjoint player IDs.

Inputs:
    data/train.csv
    data/test.csv

Outputs:
    submission3.csv
    models/player_rating_model_v3.joblib

The script deliberately never reads data/solution.csv or experiment_oof.pkl.
The latter came from a notebook whose automatic feature lists accidentally
retained performance_score, so its predictions are not valid for v3.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from pandas.api.types import is_numeric_dtype
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DATA_DIR = Path("data")
MODEL_DIR = Path("models")
SUBMISSION_PATH = Path("submission3.csv")
MODEL_PATH = MODEL_DIR / "player_rating_model_v3.joblib"

TARGET = "player_rating"
PROHIBITED_FEATURES = ["performance_score"]

REMAINING_COMPONENTS = [
    "offensive_contribution",
    "defensive_contribution",
    "possession_impact",
    "pressure_resistance",
    "creativity_score",
    "consistency_score",
    "clutch_performance_score",
]

AGGREGATE_FEATURES = REMAINING_COMPONENTS + [
    "minutes_played",
    "stamina_score",
    "market_value_eur",
]

CONTEXT_GROUPS = [
    ("player", ["player_id"]),
    ("match", ["match_id"]),
    ("match_team", ["match_id", "team"]),
    ("match_position", ["match_id", "position"]),
]

RANK_GROUPS = CONTEXT_GROUPS[1:]

# Frozen after leakage-free GroupKFold validation with player_id as the group.
ENSEMBLE_WEIGHTS = {
    "catboost": 0.60,
    "lightgbm": 0.20,
    "hist_residual": 0.20,
}


def remove_prohibited_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with every explicitly prohibited source column removed."""

    present = [column for column in PROHIBITED_FEATURES if column in frame]
    return frame.drop(columns=present).copy()


def add_context_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create target-free player and match-relative context features.

    Combining train and test here is label-free and makes the transformation
    identical for unseen test players.  The target and performance_score have
    already been removed before this function is called.
    """

    combined = pd.concat(
        [train.drop(columns=TARGET), test],
        ignore_index=True,
    )
    engineered: dict[str, pd.Series] = {}

    for prefix, keys in CONTEXT_GROUPS:
        grouped = combined.groupby(keys, dropna=False)
        for column in AGGREGATE_FEATURES:
            group_mean = grouped[column].transform("mean")
            engineered[f"{prefix}_{column}_mean"] = group_mean
            engineered[f"{prefix}_{column}_diff"] = (
                combined[column] - group_mean
            )
        engineered[f"{prefix}_row_count"] = grouped[
            AGGREGATE_FEATURES[-1]
        ].transform("size")

    for prefix, keys in RANK_GROUPS:
        grouped = combined.groupby(keys, dropna=False)
        for column in REMAINING_COMPONENTS[:4]:
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


def assert_leakage_free(columns: list[str], label: str) -> None:
    """Fail loudly if a prohibited raw feature reaches a model."""

    leaked = sorted(set(columns).intersection(PROHIBITED_FEATURES))
    if leaked:
        raise RuntimeError(f"{label} contains prohibited features: {leaked}")


def active_predictions(
    row_count: int,
    active_mask: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    predictions = np.zeros(row_count, dtype=float)
    predictions[active_mask] = values
    return np.clip(predictions, 0.0, 10.0)


def main() -> None:
    raw_train = pd.read_csv(DATA_DIR / "train.csv")
    raw_test = pd.read_csv(DATA_DIR / "test.csv")

    # This is intentionally the first transformation applied to model inputs.
    train = remove_prohibited_features(raw_train)
    test = remove_prohibited_features(raw_test)
    if any(column in train or column in test for column in PROHIBITED_FEATURES):
        raise RuntimeError("A prohibited feature survived input sanitization")

    context_train, context_test = add_context_features(train, test)
    context_train[TARGET] = train[TARGET].to_numpy()

    active_train = train["minutes_played"].gt(0).to_numpy()
    active_test = test["minutes_played"].gt(0).to_numpy()
    active_target = train.loc[active_train, TARGET].to_numpy()

    inactive_targets = train.loc[~active_train, TARGET]
    if not inactive_targets.eq(0.0).all():
        raise RuntimeError("The zero-minute/zero-rating rule no longer holds")

    numeric_features = [
        column
        for column in context_train.select_dtypes(include=np.number).columns
        if column not in {TARGET, "Id", "player_id", "jersey_number"}
    ]
    assert_leakage_free(numeric_features, "numeric_features")

    # Model 1: strongest leakage-free validation candidate.
    catboost_features = [
        column
        for column in context_train.columns
        if column not in {TARGET, "Id", "player_id", "player_name"}
    ]
    categorical_features = [
        column
        for column in catboost_features
        if not is_numeric_dtype(context_train[column])
    ]
    assert_leakage_free(catboost_features, "catboost_features")

    cat_train = context_train.copy()
    cat_test = context_test.copy()
    for column in categorical_features:
        cat_train[column] = cat_train[column].astype(str)
        cat_test[column] = cat_test[column].astype(str)

    catboost_model = CatBoostRegressor(
        iterations=700,
        depth=7,
        learning_rate=0.03,
        loss_function="RMSE",
        l2_leaf_reg=10,
        random_seed=42,
        random_strength=0.5,
        bagging_temperature=1.0,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    )
    catboost_model.fit(
        cat_train.loc[active_train, catboost_features],
        active_target,
        cat_features=categorical_features,
    )
    catboost_values = catboost_model.predict(
        cat_test.loc[active_test, catboost_features]
    )

    # Model 2: a regularized numeric boosting model.
    lightgbm_model = LGBMRegressor(
        objective="regression",
        n_estimators=900,
        learning_rate=0.02,
        num_leaves=15,
        min_child_samples=50,
        reg_lambda=15.0,
        reg_alpha=1.0,
        subsample=0.9,
        colsample_bytree=0.8,
        random_state=2,
        n_jobs=-1,
        verbosity=-1,
    )
    lightgbm_model.fit(
        context_train.loc[active_train, numeric_features],
        active_target,
    )
    lightgbm_values = lightgbm_model.predict(
        context_test.loc[active_test, numeric_features]
    )

    # Model 3: model the smaller nonlinear error left by a stable linear base.
    base_model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    base_model.fit(
        train.loc[active_train, REMAINING_COMPONENTS],
        active_target,
    )
    train_residual = active_target - base_model.predict(
        train.loc[active_train, REMAINING_COMPONENTS]
    )

    hist_residual_model = HistGradientBoostingRegressor(
        max_iter=500,
        learning_rate=0.03,
        max_leaf_nodes=31,
        min_samples_leaf=50,
        l2_regularization=10.0,
        random_state=1,
    )
    hist_residual_model.fit(
        context_train.loc[active_train, numeric_features],
        train_residual,
    )
    hist_values = base_model.predict(
        test.loc[active_test, REMAINING_COMPONENTS]
    ) + hist_residual_model.predict(
        context_test.loc[active_test, numeric_features]
    )

    active_values = (
        ENSEMBLE_WEIGHTS["catboost"] * catboost_values
        + ENSEMBLE_WEIGHTS["lightgbm"] * lightgbm_values
        + ENSEMBLE_WEIGHTS["hist_residual"] * hist_values
    )
    final_predictions = active_predictions(
        len(test),
        active_test,
        active_values,
    )

    submission = pd.DataFrame(
        {
            "Id": raw_test["Id"],
            TARGET: final_predictions,
        }
    )
    submission.to_csv(SUBMISSION_PATH, index=False)

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(
        {
            "version": 3,
            "target": TARGET,
            "prohibited_features": PROHIBITED_FEATURES,
            "catboost_model": catboost_model,
            "lightgbm_model": lightgbm_model,
            "base_model": base_model,
            "hist_residual_model": hist_residual_model,
            "remaining_components": REMAINING_COMPONENTS,
            "aggregate_features": AGGREGATE_FEATURES,
            "context_groups": CONTEXT_GROUPS,
            "rank_groups": RANK_GROUPS,
            "numeric_features": numeric_features,
            "catboost_features": catboost_features,
            "categorical_features": categorical_features,
            "weights": ENSEMBLE_WEIGHTS,
            "zero_minutes_rule": True,
            "active_target_range": (
                float(active_target.min()),
                float(active_target.max()),
            ),
            "training_rows": len(train),
            "active_training_rows": int(active_train.sum()),
        },
        MODEL_PATH,
        compress=3,
    )

    print(
        f"Training rows: {len(train):,} "
        f"({active_train.sum():,} active, {(~active_train).sum():,} zero-minute)"
    )
    print(f"Model features: {len(numeric_features)} numeric, "
          f"{len(catboost_features)} CatBoost")
    print(f"Prohibited features used: {set(PROHIBITED_FEATURES) & set(catboost_features + numeric_features)}")
    print(f"Wrote {SUBMISSION_PATH} with {len(submission):,} predictions")
    print(f"Saved v3 ensemble to {MODEL_PATH}")


if __name__ == "__main__":
    main()
