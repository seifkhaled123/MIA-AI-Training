from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from pandas.api.types import is_numeric_dtype
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
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

HIST_CONFIGS = [
    (7, 30, 2.0, 0.04, 300),
    (7, 60, 5.0, 0.04, 300),
    (15, 30, 2.0, 0.04, 300),
    (15, 50, 5.0, 0.04, 300),
    (15, 100, 10.0, 0.04, 300),
    (31, 50, 10.0, 0.03, 400),
    (31, 100, 20.0, 0.03, 400),
]

LIGHTGBM_CONFIGS = [
    (7, 50, 5.0),
    (7, 100, 10.0),
    (15, 50, 10.0),
    (15, 100, 20.0),
    (31, 100, 30.0),
]

CATBOOST_CONFIGS = [
    (8, 300, 42, 5.0),
    (6, 350, 17, 8.0),
    (7, 320, 73, 6.0),
]

RANDOM_FOREST_CONFIGS = [
    (500, 15, 0.7, 42),
    (300, 8, 0.5, 8),
]

MLP_SEEDS = [42, 7, 99, 123]
TREE_CORE_WEIGHT = 0.4
RANDOM_FOREST_WEIGHT = 0.6
MLP_WEIGHT = 0.25


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

    active_train = train["minutes_played"].gt(0)
    active_test = test["minutes_played"].gt(0).to_numpy()
    numeric_features = [
        column
        for column in train.select_dtypes(include=np.number).columns
        if column != TARGET
    ]

    base_model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    base_model.fit(
        train.loc[active_train, COMPONENT_FEATURES],
        train.loc[active_train, TARGET],
    )
    base_train = base_model.predict(train.loc[active_train, COMPONENT_FEATURES])
    base_test = base_model.predict(test.loc[active_test, COMPONENT_FEATURES])
    residual = train.loc[active_train, TARGET].to_numpy() - base_train

    hist_models = []
    hist_predictions = []
    for seed, (leaves, min_leaf, l2, learning_rate, iterations) in enumerate(
        HIST_CONFIGS,
        start=1,
    ):
        model = HistGradientBoostingRegressor(
            max_iter=iterations,
            learning_rate=learning_rate,
            max_leaf_nodes=leaves,
            min_samples_leaf=min_leaf,
            l2_regularization=l2,
            random_state=seed,
        )
        model.fit(train.loc[active_train, numeric_features], residual)
        predictions = base_test + model.predict(
            test.loc[active_test, numeric_features]
        )
        hist_predictions.append(active_predictions(test, active_test, predictions))
        hist_models.append(model)
    hist_average = np.mean(hist_predictions, axis=0)

    lightgbm_models = []
    lightgbm_predictions = []
    for seed, (leaves, min_leaf, l2) in enumerate(
        LIGHTGBM_CONFIGS,
        start=11,
    ):
        model = LGBMRegressor(
            objective="regression",
            n_estimators=500,
            learning_rate=0.02,
            num_leaves=leaves,
            min_child_samples=min_leaf,
            reg_lambda=l2,
            reg_alpha=1.0,
            subsample=0.9,
            colsample_bytree=0.8,
            random_state=seed,
            n_jobs=-1,
            verbosity=-1,
        )
        model.fit(train.loc[active_train, numeric_features], residual)
        predictions = base_test + model.predict(
            test.loc[active_test, numeric_features]
        )
        lightgbm_predictions.append(
            active_predictions(test, active_test, predictions)
        )
        lightgbm_models.append(model)
    lightgbm_average = np.mean(lightgbm_predictions, axis=0)

    catboost_features = [
        column
        for column in test.columns
        if column not in {"Id", "player_id", "player_name"}
    ]
    categorical_features = [
        column
        for column in catboost_features
        if not is_numeric_dtype(train[column])
    ]
    cat_train = train.copy()
    cat_test = test.copy()
    for column in categorical_features:
        cat_train[column] = cat_train[column].astype(str)
        cat_test[column] = cat_test[column].astype(str)

    catboost_models = []
    catboost_predictions = []
    for depth, iterations, seed, l2 in CATBOOST_CONFIGS:
        model = CatBoostRegressor(
            iterations=iterations,
            depth=depth,
            learning_rate=0.04,
            loss_function="RMSE",
            l2_leaf_reg=l2,
            random_seed=seed,
            random_strength=1.0,
            verbose=False,
            allow_writing_files=False,
            thread_count=-1,
        )
        model.fit(
            cat_train.loc[active_train, catboost_features],
            train.loc[active_train, TARGET],
            cat_features=categorical_features,
        )
        predictions = model.predict(
            cat_test.loc[active_test, catboost_features]
        )
        catboost_predictions.append(
            active_predictions(test, active_test, predictions)
        )
        catboost_models.append(model)
    catboost_average = np.mean(catboost_predictions, axis=0)

    tree_core = np.mean(
        [hist_average, lightgbm_average, catboost_average],
        axis=0,
    )

    random_forest_models = []
    random_forest_predictions = []
    for estimators, min_leaf, max_features, seed in RANDOM_FOREST_CONFIGS:
        model = RandomForestRegressor(
            n_estimators=estimators,
            min_samples_leaf=min_leaf,
            max_features=max_features,
            n_jobs=-1,
            random_state=seed,
        )
        model.fit(train.loc[active_train, numeric_features], residual)
        predictions = base_test + model.predict(
            test.loc[active_test, numeric_features]
        )
        random_forest_predictions.append(
            active_predictions(test, active_test, predictions)
        )
        random_forest_models.append(model)
    random_forest_average = np.mean(random_forest_predictions, axis=0)

    tree_ensemble = (
        RANDOM_FOREST_WEIGHT * random_forest_average
        + TREE_CORE_WEIGHT * tree_core
    )

    mlp_models = []
    mlp_predictions = []
    for seed in MLP_SEEDS:
        model = make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                solver="adam",
                alpha=1.0,
                batch_size=256,
                learning_rate_init=0.001,
                max_iter=200,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=15,
                random_state=seed,
            ),
        )
        model.fit(train.loc[active_train, numeric_features], residual)
        predictions = base_test + model.predict(
            test.loc[active_test, numeric_features]
        )
        mlp_predictions.append(active_predictions(test, active_test, predictions))
        mlp_models.append(model)
    mlp_average = np.mean(mlp_predictions, axis=0)

    final_predictions = np.clip(
        (1.0 - MLP_WEIGHT) * tree_ensemble + MLP_WEIGHT * mlp_average,
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
    submission.to_csv("submission.csv", index=False)

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(
        {
            "base_model": base_model,
            "hist_models": hist_models,
            "lightgbm_models": lightgbm_models,
            "catboost_models": catboost_models,
            "random_forest_models": random_forest_models,
            "mlp_models": mlp_models,
            "component_features": COMPONENT_FEATURES,
            "numeric_features": numeric_features,
            "catboost_features": catboost_features,
            "categorical_features": categorical_features,
            "weights": {
                "tree_core": TREE_CORE_WEIGHT,
                "random_forest": RANDOM_FOREST_WEIGHT,
                "mlp": MLP_WEIGHT,
            },
            "zero_minutes_rule": True,
        },
        MODEL_DIR / "player_rating_model.joblib",
        compress=3,
    )
    print(f"Wrote submission.csv with {len(submission)} predictions")
    print(f"Saved ensemble to {MODEL_DIR / 'player_rating_model.joblib'}")


if __name__ == "__main__":
    main()
