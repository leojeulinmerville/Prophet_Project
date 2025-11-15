# btc_engine/data_loader.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from .config import DATA_PATH, MODEL_PATH


REQUIRED_BT_COLUMNS = {
    "date_t",
    "date_t_plus_1",
    "price_t",
    "price_t_plus_1",
    "actual_return",
    "yhat_log",
    "yhat_lower_log",
    "yhat_upper_log",
    "yhat_price_next",
    "predicted_return",
}


def load_price_history(path: Optional[str | Path] = None) -> pd.DataFrame:
    """
    Load and clean the raw BTC-USD CSV (Yahoo-style).

    Expected structure (like your screenshot):
        Price,Close,High,Low,Open,Volume
        Ticker,BTC-USD,BTC-USD,BTC-USD,BTC-USD,BTC-USD
        Date,,,,,
        2020-01-01, 7200..., 7254..., 7174..., ...

    We:
      - rename 'Price' -> 'Date'
      - drop the 'Ticker' / 'Date' metadata rows
      - keep only rows where Date looks like YYYY-MM-DD
      - compute Typical_Price = (High + Low + Close) / 3

    Returns
    -------
    pd.DataFrame
        Columns: ['ds', 'Typical_Price']
        sorted by ds (ascending).
    """
    if path is None:
        path = DATA_PATH

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"BTC-USD CSV not found at {path}")

    df = pd.read_csv(path)

    # Clean column names
    df.columns = [c.strip() for c in df.columns]

    # Your file uses 'Price' as the first column header -> actual Date values underneath.
    if "Price" in df.columns and "Date" not in df.columns:
        df = df.rename(columns={"Price": "Date"})

    if "Date" not in df.columns:
        raise ValueError(
            "Could not find a 'Date' column in BTC-USD.csv after cleaning. "
            "Make sure the first column is the date."
        )

    # Keep only rows where Date looks like 'YYYY-MM-DD'
    df["Date"] = df["Date"].astype(str).str.strip()
    mask = df["Date"].str.match(r"^\d{4}-\d{2}-\d{2}$")
    df = df[mask].copy()

    if df.empty:
        raise ValueError(
            "No valid date rows found in BTC-USD.csv. "
            "Check that the file has rows like '2020-01-01,...'."
        )

    df["Date"] = pd.to_datetime(df["Date"])

    # Convert price columns to numeric where present
    for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # We need at least Close, High, Low for Typical_Price
    for col in ["Close", "High", "Low"]:
        if col not in df.columns:
            raise ValueError(
                "BTC-USD.csv must contain 'Close', 'High' and 'Low' columns "
                f"after cleaning, but got columns: {list(df.columns)}"
            )

    df = df.sort_values("Date").reset_index(drop=True)
    df["Typical_Price"] = (df["High"] + df["Low"] + df["Close"]) / 3.0

    price_df = df[["Date", "Typical_Price"]].rename(columns={"Date": "ds"})
    return price_df


def load_backtest_dataframe(
    model: Any,
    data_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """
    Build the backtest dataframe `bt_df` from raw BTC-USD data and a fitted Prophet model.

    We do NOT retrain the model here; we:
      - load cleaned BTC price history
      - ask the saved Prophet model to predict log-price yhat, yhat_lower, yhat_upper
        for each historical date
      - build a day-ahead backtest:
            t -> t+1 for
            price, actual_return, prophet-forecasted next-day log price & return

    This is in-sample (since the model was trained on the full history),
    but it's good enough for an educational, non-production engine.

    Returns
    -------
    pd.DataFrame
        Columns:
        - date_t, date_t_plus_1
        - price_t, price_t_plus_1
        - actual_return
        - yhat_log, yhat_lower_log, yhat_upper_log
        - yhat_price_next
        - predicted_return
    """
    price_df = load_price_history(data_path)

    # Ask Prophet for log-price prediction on each historical date
    future = price_df[["ds"]].copy()
    forecast = model.predict(future)[["ds", "yhat", "yhat_lower", "yhat_upper"]]

    merged = price_df.merge(forecast, on="ds", how="inner").sort_values("ds").reset_index(drop=True)

    # Build t -> t+1 structure
    bt = pd.DataFrame()
    bt["date_t"] = merged["ds"]
    bt["date_t_plus_1"] = merged["ds"].shift(-1)

    bt["price_t"] = merged["Typical_Price"]
    bt["price_t_plus_1"] = merged["Typical_Price"].shift(-1)

    bt["actual_return"] = bt["price_t_plus_1"] / bt["price_t"] - 1.0

    # Prophet log-price forecast for next day: use yhat(t+1), yhat_lower(t+1), yhat_upper(t+1)
    bt["yhat_log"] = merged["yhat"].shift(-1)
    bt["yhat_lower_log"] = merged["yhat_lower"].shift(-1)
    bt["yhat_upper_log"] = merged["yhat_upper"].shift(-1)

    # Convert log-price to price space and compute predicted return
    bt["yhat_price_next"] = np.exp(bt["yhat_log"])
    bt["predicted_return"] = bt["yhat_price_next"] / bt["price_t"] - 1.0

    # Drop last row (no t+1) and any incomplete data
    bt = bt.dropna().reset_index(drop=True)

    # Sanity check: required columns
    missing = REQUIRED_BT_COLUMNS.difference(bt.columns)
    if missing:
        raise ValueError(
            f"Internal error: backtest dataframe missing columns {missing}. "
            "Check data_loader implementation."
        )

    return bt


if __name__ == "__main__":
    # Debug helper: run as `python -m btc_engine.data_loader`
    from pprint import pprint
    from .prophet_model import load_prophet_model

    model = load_prophet_model(MODEL_PATH)
    df_bt = load_backtest_dataframe(model=model)
    print("bt_df built successfully from BTC-USD.csv and the saved Prophet model.")
    print("Shape:", df_bt.shape)
    print("Columns:")
    pprint(list(df_bt.columns))
    print(df_bt.head())
