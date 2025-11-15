# btc_engine/prophet_model.py
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


def load_prophet_model(model_path: str | Path) -> Any:
    """
    Load a saved Prophet model from disk.

    The model is assumed to have been trained on:
        y = log(Typical_Price)

    Parameters
    ----------
    model_path:
        Path to the joblib file.

    Returns
    -------
    Any
        A fitted Prophet model instance.
    """
    model_path = Path(model_path)
    model = joblib.load(model_path)
    return model


def forecast_price_path(
    model: Any,
    periods: int,
    freq: str = "D",
) -> pd.DataFrame:
    """
    Build a day-ahead forecast path from a fitted Prophet model.

    The model is assumed to predict log-price (y = log(price)).
    This function converts predictions back to price space and derives
    predicted returns, plus a simple confidence band based on Prophet
    intervals.

    Parameters
    ----------
    model:
        Fitted Prophet model (with .history and .predict).
    periods:
        Number of days to forecast ahead.
    freq:
        Frequency for the future dataframe (default "D").

    Returns
    -------
    pd.DataFrame
        Columns:
        - ds
        - yhat, yhat_lower, yhat_upper (log-price)
        - price_pred, price_lower, price_upper
        - predicted_return, predicted_return_lower, predicted_return_upper
    """
    if periods <= 0:
        raise ValueError("`periods` must be a positive integer.")

    # Build future dates and run Prophet
    future = model.make_future_dataframe(periods=periods, freq=freq, include_history=False)
    forecast = model.predict(future)[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()

    # Last observed log-price from training history
    last_log_price = float(model.history["y"].iloc[-1])
    last_price = float(np.exp(last_log_price))

    # Convert predicted log-prices to price space
    forecast["price_pred"] = np.exp(forecast["yhat"])
    forecast["price_lower"] = np.exp(forecast["yhat_lower"])
    forecast["price_upper"] = np.exp(forecast["yhat_upper"])

    # Build sequential returns vs last observed / previous predicted price
    prices = np.concatenate([[last_price], forecast["price_pred"].values])
    lower_prices = np.concatenate([[last_price], forecast["price_lower"].values])
    upper_prices = np.concatenate([[last_price], forecast["price_upper"].values])

    predicted_returns = prices[1:] / prices[:-1] - 1.0
    predicted_returns_lower = lower_prices[1:] / lower_prices[:-1] - 1.0
    predicted_returns_upper = upper_prices[1:] / upper_prices[:-1] - 1.0

    forecast["predicted_return"] = predicted_returns
    forecast["predicted_return_lower"] = predicted_returns_lower
    forecast["predicted_return_upper"] = predicted_returns_upper

    return forecast
