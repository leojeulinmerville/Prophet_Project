# btc_engine/strategies.py
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def buy_and_hold_positions(bt_df: pd.DataFrame) -> pd.Series:
    """
    Strategy 0: Buy & Hold.
    Always fully invested (long 1x) in BTC.

    Parameters
    ----------
    bt_df:
        Backtest dataframe (index is used for the positions index).

    Returns
    -------
    pd.Series
        Position series in [-1, 0, 1], here always 1.0.
    """
    positions = pd.Series(1.0, index=bt_df.index, name="position")
    return positions


def directional_positions(bt_df: pd.DataFrame) -> pd.Series:
    """
    Strategy 1: Directional (long/cash/short) based on the sign of predicted_return.

    Long when predicted_return > 0,
    Short when predicted_return < 0,
    Flat when predicted_return == 0.

    Parameters
    ----------
    bt_df:
        Must contain the column 'predicted_return'.

    Returns
    -------
    pd.Series
        Position series in [-1, 0, 1].
    """
    pred = bt_df["predicted_return"].fillna(0.0).to_numpy()
    positions = np.sign(pred)
    return pd.Series(positions, index=bt_df.index, name="position")


def threshold_positions(
    bt_df: pd.DataFrame,
    threshold: float = 0.005,
) -> pd.Series:
    """
    Strategy 2: Threshold strategy.

    Trade only when |predicted_return| > threshold (e.g. 0.5%).

    - Long 1.0 when predicted_return > threshold
    - Short -1.0 when predicted_return < -threshold
    - Otherwise stay in cash (0.0)

    Parameters
    ----------
    bt_df:
        Must contain 'predicted_return'.
    threshold:
        Minimum absolute forecasted return (in decimal) to open a position.

    Returns
    -------
    pd.Series
        Position series in [-1, 0, 1].
    """
    pred = bt_df["predicted_return"].fillna(0.0).to_numpy()

    positions = np.zeros_like(pred)
    positions[pred > threshold] = 1.0
    positions[pred < -threshold] = -1.0

    return pd.Series(positions, index=bt_df.index, name="position")


def high_confidence_positions(
    bt_df: pd.DataFrame,
    min_abs_return: float = 0.01,
    lower_col: str = "yhat_lower_log",
    upper_col: str = "yhat_upper_log",
    price_col: str = "price_t",
) -> pd.Series:
    """
    Strategy 3: Conservative high-confidence strategy.

    - Compute an approximate confidence band on predicted returns using
      Prophet's log-price intervals.
    - Only take trades when:
        * |predicted_return| > min_abs_return
        * AND the return interval does not cross 0:
            - long if lower_predicted_return > 0
            - short if upper_predicted_return < 0

    Parameters
    ----------
    bt_df:
        Must contain 'predicted_return', 'price_t', and log-interval columns.
    min_abs_return:
        Minimum absolute forecasted return to consider a trade (e.g. 1%).
    lower_col, upper_col:
        Names of the log-interval columns from Prophet.
    price_col:
        Column with current price at time t.

    Returns
    -------
    pd.Series
        Position series in [-1, 0, 1].
    """
    if not {lower_col, upper_col, price_col, "predicted_return"}.issubset(bt_df.columns):
        raise ValueError(
            "high_confidence_positions requires "
            f"columns: {lower_col}, {upper_col}, {price_col}, predicted_return."
        )

    price_t = bt_df[price_col].to_numpy()
    yhat_lower_log = bt_df[lower_col].to_numpy()
    yhat_upper_log = bt_df[upper_col].to_numpy()
    pred = bt_df["predicted_return"].fillna(0.0).to_numpy()

    # Convert Prophet log-price intervals to price space, then to returns
    price_lower_next = np.exp(yhat_lower_log)
    price_upper_next = np.exp(yhat_upper_log)

    ret_lower = (price_lower_next - price_t) / price_t
    ret_upper = (price_upper_next - price_t) / price_t

    positions = np.zeros(len(bt_df))

    # Long: forecast > min_abs_return and lower bound > 0 (interval entirely positive)
    long_mask = (pred > min_abs_return) & (ret_lower > 0.0)

    # Short: forecast < -min_abs_return and upper bound < 0 (interval entirely negative)
    short_mask = (pred < -min_abs_return) & (ret_upper < 0.0)

    positions[long_mask] = 1.0
    positions[short_mask] = -1.0

    return pd.Series(positions, index=bt_df.index, name="position")
