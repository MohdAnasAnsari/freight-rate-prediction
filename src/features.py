"""Feature engineering for the freight-rate model."""

from __future__ import annotations

import numpy as np
import pandas as pd


CATEGORICAL_FEATURES = ["pickup", "delivery", "equipment"]
NUMERIC_FEATURES = [
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "distance",
    "weight",
    "market_index",
    "quote_signal",
    "month",
    "day_of_week",
    "day_of_year",
    "week_of_year",
    "hav_distance",
    "weight_per_mile",
]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _haversine_distance(frame: pd.DataFrame) -> pd.Series:
    """Return great-circle miles, falling back to supplied route distance."""
    lat1 = np.radians(_numeric(frame, "pickup_lat"))
    lon1 = np.radians(_numeric(frame, "pickup_lon"))
    lat2 = np.radians(_numeric(frame, "delivery_lat"))
    lon2 = np.radians(_numeric(frame, "delivery_lon"))

    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1
    hav = np.sin(delta_lat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon / 2.0) ** 2
    miles = 2.0 * 3958.7613 * np.arcsin(np.sqrt(np.clip(hav, 0.0, 1.0)))
    miles = pd.Series(miles, index=frame.index)
    return miles.fillna(_numeric(frame, "distance"))


def build_features(df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
    """Build the model matrix while preserving LightGBM categorical dtypes."""
    if is_train and "posted_rate" not in df:
        raise ValueError("Training data must include posted_rate")

    data = df.copy()
    date = pd.to_datetime(data["date"], errors="raise")
    data["month"] = date.dt.month
    data["day_of_week"] = date.dt.dayofweek
    data["day_of_year"] = date.dt.dayofyear
    data["week_of_year"] = date.dt.isocalendar().week.astype(int)
    data["hav_distance"] = _haversine_distance(data)

    distance = _numeric(data, "distance").replace(0, np.nan)
    data["weight_per_mile"] = _numeric(data, "weight") / distance

    for column in CATEGORICAL_FEATURES:
        if column not in data:
            data[column] = "Unknown"
        data[column] = data[column].fillna("Unknown").astype("category")
    for column in NUMERIC_FEATURES:
        data[column] = _numeric(data, column).astype(float)

    return data[FEATURE_COLUMNS]


def category_levels(features: pd.DataFrame) -> dict[str, list[str]]:
    """Capture training category order for deterministic inference."""
    return {
        column: [str(value) for value in features[column].cat.categories]
        for column in CATEGORICAL_FEATURES
    }


def align_categories(features: pd.DataFrame, levels: dict[str, list[str]]) -> pd.DataFrame:
    """Align inference categoricals to the categories learned during training."""
    result = features.copy()
    for column in CATEGORICAL_FEATURES:
        result[column] = result[column].astype(str).astype(
            pd.CategoricalDtype(categories=levels[column])
        )
    return result
