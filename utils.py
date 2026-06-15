"""
utils.py
--------
Shared helpers for the Walmart sales-forecasting project:
  - robust loader for the Kaggle "Walmart.csv" dataset
  - time-series feature engineering (date parts, lags, rolling means)
  - a from-scratch Holt-Winters (triple exponential smoothing) forecaster
  - a currency formatter

The loader prefers the REAL Kaggle file (data/Walmart.csv) and falls back to
the demo file. It handles the dd-mm-yyyy date format used in the real data.
"""

import os
import numpy as np
import pandas as pd

CANDIDATES = [
    "data/Walmart.csv",
    "data/walmart.csv",
    "data/sample_demo_data.csv",
]


def find_data_file():
    for path in CANDIDATES:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "No dataset in data/. Run `python make_demo_data.py` or download "
        "'Walmart.csv' from Kaggle into data/."
    )


def _parse_dates(series):
    """Walmart dates are dd-mm-yyyy; fall back to generic parsing."""
    d = pd.to_datetime(series, format="%d-%m-%Y", errors="coerce")
    if d.isna().mean() > 0.3:
        d = pd.to_datetime(series, errors="coerce", dayfirst=True)
    return d


def load_walmart():
    """Load + clean the Walmart data; returns (DataFrame, source_filename)."""
    path = find_data_file()
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df["Date"] = _parse_dates(df["Date"])
    df = df.dropna(subset=["Date"]).sort_values(["Store", "Date"]).reset_index(drop=True)

    # numeric coercion + basic fills
    for c in ["Weekly_Sales", "Temperature", "Fuel_Price", "CPI", "Unemployment"]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["Holiday_Flag"] = pd.to_numeric(df.get("Holiday_Flag", 0), errors="coerce").fillna(0).astype(int)
    df = df.drop_duplicates().reset_index(drop=True)
    return df, os.path.basename(path)


def add_time_features(df):
    """Add calendar features + per-store lag and rolling-mean features."""
    df = df.copy()
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Week"] = df["Date"].dt.isocalendar().week.astype(int)
    df["DayOfYear"] = df["Date"].dt.dayofyear
    df = df.sort_values(["Store", "Date"])
    g = df.groupby("Store")["Weekly_Sales"]
    df["lag1"] = g.shift(1)
    df["lag2"] = g.shift(2)
    df["lag52"] = g.shift(52)                       # same week last year
    df["roll4"] = g.shift(1).rolling(4).mean().reset_index(level=0, drop=True)
    df["roll12"] = g.shift(1).rolling(12).mean().reset_index(level=0, drop=True)
    return df


def holt_winters_additive(series, season_len=52, alpha=0.3, beta=0.05,
                          gamma=0.3, n_forecast=12):
    """
    From-scratch additive Holt-Winters (triple exponential smoothing).
    Returns the forecast list for the next `n_forecast` steps.
    Falls back to a shorter season if the series is too short.
    """
    y = list(series)
    n = len(y)
    if n < 2 * season_len:
        season_len = max(2, min(season_len, n // 2))
    if season_len < 2:
        return [y[-1]] * n_forecast

    # initial level, trend, seasonals
    level = np.mean(y[:season_len])
    trend = (np.mean(y[season_len:2 * season_len]) - np.mean(y[:season_len])) / season_len
    seasonals = [y[i] - level for i in range(season_len)]

    for i in range(season_len, n):
        val = y[i]
        last_level = level
        s = seasonals[i % season_len]
        level = alpha * (val - s) + (1 - alpha) * (level + trend)
        trend = beta * (level - last_level) + (1 - beta) * trend
        seasonals[i % season_len] = gamma * (val - level) + (1 - gamma) * s

    forecast = []
    for m in range(1, n_forecast + 1):
        forecast.append(level + m * trend + seasonals[(n + m - 1) % season_len])
    return forecast


def money(x):
    if abs(x) >= 1e9:
        return f"${x/1e9:.2f}B"
    if abs(x) >= 1e6:
        return f"${x/1e6:.2f}M"
    if abs(x) >= 1e3:
        return f"${x/1e3:.1f}K"
    return f"${x:,.0f}"
