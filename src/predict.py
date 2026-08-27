"""Create validation and fixed-lane December predictions."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from clean import clean_frame
from features import align_categories, build_features


ROOT = Path(__file__).resolve().parents[1]
DECEMBER_COLUMNS = ["pickup", "delivery", "distance", "equipment", "weight", "date", "predicted_rate"]


def model_predict(frame: pd.DataFrame, artifact: dict) -> np.ndarray:
    cleaned = clean_frame(frame, artifact["cleaning_statistics"])
    features = build_features(cleaned, is_train=False)
    features = align_categories(features, artifact["category_levels"])
    predictions = np.expm1(artifact["model"].predict(features))
    return np.maximum(np.asarray(predictions, dtype=float), 50.0)


def predict_validation(artifact: dict) -> None:
    validation = pd.read_csv(ROOT / "data" / "validation.csv", parse_dates=["date"])
    rates = model_predict(validation, artifact)
    keyed_predictions = pd.DataFrame(
        {"load_id": validation["load_id"].astype(str), "predicted_rate": rates}
    )

    template = pd.read_csv(ROOT / "data" / "validation_predictions_template.csv")
    if list(template.columns) != ["load_id", "predicted_rate"]:
        raise ValueError("Validation template schema has changed")
    output = template[["load_id"]].merge(
        keyed_predictions,
        on="load_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if output["predicted_rate"].isna().any():
        raise ValueError("Some template load_id values had no matching prediction")
    output.to_csv(ROOT / "validation_predictions.csv", index=False)
    print(f"Saved {len(output):,} keyed validation predictions to validation_predictions.csv")


def _city_coordinates(train: pd.DataFrame, city: str, side: str) -> tuple[float, float]:
    rows = train.loc[train[side].eq(city)]
    if rows.empty:
        raise ValueError(f"Could not find {city!r} in training {side} values")
    return float(rows[f"{side}_lat"].median()), float(rows[f"{side}_lon"].median())


def predict_december(artifact: dict) -> None:
    path = ROOT / "data" / "december_chart_inputs.csv"
    original = pd.read_csv(path)
    if list(original.columns) != DECEMBER_COLUMNS:
        raise ValueError("December input schema has changed")

    train = pd.read_csv(ROOT / "data" / "train_test.csv", parse_dates=["date"])
    december = original.drop(columns=["predicted_rate"]).copy()
    pickup_lat, pickup_lon = _city_coordinates(train, "Lexington", "pickup")
    delivery_lat, delivery_lon = _city_coordinates(train, "Fort Wayne", "delivery")
    december["pickup_lat"] = pickup_lat
    december["pickup_lon"] = pickup_lon
    december["delivery_lat"] = delivery_lat
    december["delivery_lon"] = delivery_lon

    # December is beyond the labeled range. Use the median Jan-Feb winter index
    # as the documented seasonal proxy instead of extrapolating a short trend.
    winter = train.loc[train["date"].dt.month.isin([1, 2]), "market_index"]
    december["market_index"] = float(winter.median())

    similar_loads = train.loc[
        train["equipment"].eq("Dry Van") & train["distance"].between(330, 390)
    ].copy()
    if similar_loads.empty:
        similar_loads = train.loc[train["equipment"].eq("Dry Van")].copy()
    december["quote_signal"] = float(similar_loads["quote_signal"].median())

    rates = model_predict(december, artifact)
    # The tree model has little resolution beyond its Oct training boundary. A
    # small training-only weekday calibration supplies the requested daily
    # variation using comparable-lane behavior, capped at +/-3% for stability.
    similar_loads["weekday"] = similar_loads["date"].dt.dayofweek
    similar_loads["rate_per_mile"] = similar_loads["posted_rate"] / similar_loads["distance"]
    weekday_rate = similar_loads.groupby("weekday")["rate_per_mile"].median()
    weekday_factor = weekday_rate / weekday_rate.mean()
    december_weekday = pd.to_datetime(december["date"]).dt.dayofweek
    calibration = december_weekday.map(weekday_factor).fillna(1.0).clip(0.97, 1.03)
    rates = rates * calibration.to_numpy()
    completed = original.copy()
    completed["predicted_rate"] = rates
    completed = completed[DECEMBER_COLUMNS]
    completed.to_csv(path, index=False)
    print(f"Filled {len(completed)} December predictions in data/december_chart_inputs.csv")


def main() -> None:
    artifact_path = ROOT / "models" / "model.pkl"
    if not artifact_path.is_file():
        raise FileNotFoundError("models/model.pkl is missing; run python src/train.py first")
    artifact = joblib.load(artifact_path)
    predict_validation(artifact)
    predict_december(artifact)


if __name__ == "__main__":
    main()
