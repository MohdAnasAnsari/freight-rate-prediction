"""Cleaning utilities shared by training, inference, and EDA."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class CleaningStatistics:
    """Training-derived fallback values used without target leakage."""

    equipment_weight_medians: dict[str, float]
    global_weight_median: float
    global_market_median: float


def fit_cleaning_statistics(frame: pd.DataFrame) -> CleaningStatistics:
    """Fit reusable imputation statistics on a training frame only."""
    data = frame.copy()
    weight = pd.to_numeric(data.get("weight"), errors="coerce").abs()
    equipment = data.get("equipment", pd.Series("Unknown", index=data.index)).astype(str)
    equipment_medians = weight.groupby(equipment).median().dropna().to_dict()
    global_weight = float(weight.median())

    market = pd.to_numeric(data.get("market_index"), errors="coerce")
    global_market = float(market.median())
    if not np.isfinite(global_market):
        global_market = 1.0

    return CleaningStatistics(
        equipment_weight_medians={str(key): float(value) for key, value in equipment_medians.items()},
        global_weight_median=global_weight,
        global_market_median=global_market,
    )


def clean_frame(frame: pd.DataFrame, statistics: CleaningStatistics) -> pd.DataFrame:
    """Clean a frame without dropping rows.

    Weight fallbacks are learned only from training data. Market-index gaps first
    use other observed values on the same date, then the same month, as required
    by the assessment. These are feature-only aggregates and never use the target;
    the final fallback is the training-derived global median.
    """
    data = frame.copy()
    data["date"] = pd.to_datetime(data["date"], errors="raise")

    data["weight"] = pd.to_numeric(data.get("weight"), errors="coerce").abs()
    equipment_fallback = data["equipment"].astype(str).map(statistics.equipment_weight_medians)
    data["weight"] = data["weight"].fillna(equipment_fallback).fillna(statistics.global_weight_median)

    if "market_index" not in data:
        data["market_index"] = np.nan
    data["market_index"] = pd.to_numeric(data["market_index"], errors="coerce")
    date_medians = data.groupby("date", observed=True)["market_index"].transform("median")
    month_medians = data.groupby(data["date"].dt.month, observed=True)["market_index"].transform("median")
    data["market_index"] = (
        data["market_index"]
        .fillna(date_medians)
        .fillna(month_medians)
        .fillna(statistics.global_market_median)
    )

    return data


def load_and_clean(path: str | Path) -> pd.DataFrame:
    """Load a CSV, parse dates, fix weights, and impute missing values."""
    frame = pd.read_csv(path)
    statistics = fit_cleaning_statistics(frame)
    return clean_frame(frame, statistics)
