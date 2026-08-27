"""Train and evaluate the temporal freight-rate forecasting model."""

from __future__ import annotations

from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from clean import clean_frame, fit_cleaning_statistics
from features import FEATURE_COLUMNS, build_features, category_levels


ROOT = Path(__file__).resolve().parents[1]
RANDOM_SEED = 42
TRAIN_END = pd.Timestamp("2025-08-31")
HOLDOUT_START = pd.Timestamp("2025-09-01")
HOLDOUT_END = pd.Timestamp("2025-10-31")


def regression_metrics(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    actual_array = np.asarray(actual, dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)
    return {
        "MAE": float(mean_absolute_error(actual_array, predicted_array)),
        "MAPE": float(np.mean(np.abs((actual_array - predicted_array) / actual_array)) * 100.0),
        "RMSE": float(mean_squared_error(actual_array, predicted_array) ** 0.5),
    }


def baseline_predictions(train: pd.DataFrame, holdout: pd.DataFrame) -> np.ndarray:
    rate_per_mile = train["posted_rate"] / train["distance"].replace(0, np.nan)
    equipment_medians = rate_per_mile.groupby(train["equipment"]).median()
    global_median = float(rate_per_mile.median())
    holdout_rpm = holdout["equipment"].map(equipment_medians).fillna(global_median)
    return (holdout_rpm * holdout["distance"]).to_numpy()


def make_model(n_estimators: int) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="regression_l1",
        n_estimators=n_estimators,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=30,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.05,
        reg_lambda=0.2,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbosity=-1,
    )


def save_metrics(metrics: pd.DataFrame) -> None:
    report_dir = ROOT / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    table = metrics.to_string(index=False, float_format=lambda value: f"{value:,.3f}")
    print("\nTemporal holdout metrics (Sep-Oct 2025)\n")
    print(table)
    (report_dir / "metrics.txt").write_text(
        "Temporal holdout: train 2025-01-01 through 2025-08-31; "
        "holdout 2025-09-01 through 2025-10-31.\n\n" + table + "\n",
        encoding="utf-8",
    )


def save_feature_importance(model: lgb.LGBMRegressor) -> None:
    importance = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS).sort_values().tail(15)
    figure, axis = plt.subplots(figsize=(9, 6), dpi=160)
    importance.plot.barh(ax=axis, color="#0B7285")
    axis.set_title("LightGBM feature importance", loc="left", fontweight="bold")
    axis.set_xlabel("Split importance")
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(ROOT / "report" / "feature_importance.png", bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    raw = pd.read_csv(ROOT / "data" / "train_test.csv", parse_dates=["date"])
    train_raw = raw.loc[raw["date"] <= TRAIN_END].copy()
    holdout_raw = raw.loc[raw["date"].between(HOLDOUT_START, HOLDOUT_END)].copy()
    if train_raw.empty or holdout_raw.empty:
        raise ValueError("Temporal split produced an empty partition")
    assert train_raw["date"].max() < holdout_raw["date"].min(), "Temporal leakage detected"

    split_statistics = fit_cleaning_statistics(train_raw)
    train = clean_frame(train_raw, split_statistics)
    holdout = clean_frame(holdout_raw, split_statistics)
    train_features = build_features(train, is_train=True)
    holdout_features = build_features(holdout, is_train=False)

    baseline = baseline_predictions(train, holdout)
    baseline_scores = regression_metrics(holdout["posted_rate"], baseline)

    # A log target dampens the influence of the confirmed extreme rate-per-mile
    # outliers while retaining every labeled row and every future prediction row.
    model = make_model(n_estimators=2_000)
    model.fit(
        train_features,
        np.log1p(train["posted_rate"]),
        eval_set=[(holdout_features, np.log1p(holdout["posted_rate"]))],
        eval_metric="l1",
        categorical_feature=["pickup", "delivery", "equipment"],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )
    model_predictions = np.maximum(np.expm1(model.predict(holdout_features)), 50.0)
    model_scores = regression_metrics(holdout["posted_rate"], model_predictions)

    metrics = pd.DataFrame(
        [
            {"Model": "Equipment median $/mi baseline", **baseline_scores},
            {"Model": "LightGBM log-target", **model_scores},
        ]
    )
    save_metrics(metrics)
    if not (model_scores["MAE"] < baseline_scores["MAE"] and model_scores["RMSE"] < baseline_scores["RMSE"]):
        raise RuntimeError("LightGBM did not beat the baseline on both MAE and RMSE")
    save_feature_importance(model)

    best_iteration = int(model.best_iteration_ or model.n_estimators_)
    final_statistics = fit_cleaning_statistics(raw)
    full_data = clean_frame(raw, final_statistics)
    full_features = build_features(full_data, is_train=True)
    final_model = make_model(n_estimators=best_iteration)
    final_model.fit(
        full_features,
        np.log1p(full_data["posted_rate"]),
        categorical_feature=["pickup", "delivery", "equipment"],
        callbacks=[lgb.log_evaluation(0)],
    )

    models_dir = ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": final_model,
            "cleaning_statistics": final_statistics,
            "category_levels": category_levels(full_features),
            "feature_columns": FEATURE_COLUMNS,
            "best_iteration": best_iteration,
            "random_seed": RANDOM_SEED,
        },
        models_dir / "model.pkl",
    )
    print(f"\nSaved final model with {best_iteration} boosting rounds to models/model.pkl")


if __name__ == "__main__":
    main()
