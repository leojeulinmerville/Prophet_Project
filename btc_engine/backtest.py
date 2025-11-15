# btc_engine/backtest.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping

import numpy as np
import pandas as pd

from .config import RISK_FREE_RATE, TRADING_DAYS_PER_YEAR
from .risk import RiskMetrics, compute_risk_metrics


PositionFunc = Callable[..., pd.Series]


@dataclass
class BacktestResult:
    """
    Result of running a single strategy over the backtest dataframe.
    """
    name: str
    positions: pd.Series
    returns: pd.Series
    equity_curve: pd.Series
    risk: RiskMetrics


def compute_strategy_returns(
    bt_df: pd.DataFrame,
    positions: pd.Series,
    fee_rate: float = 0.001,
) -> pd.Series:
    """
    Compute daily strategy returns given positions and realized returns.

    Assumptions:
    - `bt_df["actual_return"]` is the realized BTC return from t to t+1.
    - Positions are leverage/weight in BTC (e.g., 1.0 = fully long, -1.0 = fully short).
    - Transaction cost is applied on changes in position:
        fee_cost_t = fee_rate * |position_t - position_{t-1}|

    Parameters
    ----------
    bt_df:
        Must contain column 'actual_return' in decimal form.
    positions:
        Position series aligned with bt_df.index.
    fee_rate:
        Per-trade fee in decimal form (0.001 = 0.1%).

    Returns
    -------
    pd.Series
        Daily strategy returns net of transaction costs.
    """
    if "actual_return" not in bt_df.columns:
        raise ValueError("bt_df must contain the 'actual_return' column.")

    positions = positions.reindex(bt_df.index).fillna(0.0)
    actual_returns = bt_df["actual_return"].astype(float)

    # Raw PnL from market exposure
    gross_returns = positions * actual_returns

    # Transaction costs from changes in position
    position_changes = positions.diff().abs().fillna(0.0)
    fee_costs = fee_rate * position_changes

    net_returns = gross_returns - fee_costs
    net_returns.name = "strategy_return"

    return net_returns


def run_single_strategy(
    name: str,
    bt_df: pd.DataFrame,
    position_func: PositionFunc,
    fee_rate: float = 0.001,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    **position_kwargs: Any,
) -> BacktestResult:
    """
    Run one strategy over the backtest dataframe.

    Parameters
    ----------
    name:
        Strategy name (identifier).
    bt_df:
        Backtest dataframe with at least 'actual_return'.
    position_func:
        Function that maps (bt_df, **position_kwargs) -> positions Series.
    fee_rate:
        Trading fee per change in position.
    periods_per_year:
        Used in risk metrics.
    position_kwargs:
        Additional parameters passed to the position function.

    Returns
    -------
    BacktestResult
    """
    positions = position_func(bt_df, **position_kwargs)
    returns = compute_strategy_returns(bt_df, positions, fee_rate=fee_rate)
    equity_curve = (1.0 + returns).cumprod()
    risk_metrics = compute_risk_metrics(
        returns,
        positions,
        periods_per_year=periods_per_year,
        risk_free_rate=RISK_FREE_RATE,
    )

    return BacktestResult(
        name=name,
        positions=positions,
        returns=returns,
        equity_curve=equity_curve,
        risk=risk_metrics,
    )


def run_strategies(
    bt_df: pd.DataFrame,
    strategies: Mapping[str, tuple[PositionFunc, Dict[str, Any]]],
    fee_rate: float = 0.001,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> Dict[str, BacktestResult]:
    """
    Run multiple strategies over the same backtest dataframe.

    Parameters
    ----------
    bt_df:
        Backtest dataframe.
    strategies:
        Mapping of strategy name -> (position_func, kwargs_dict).
    fee_rate:
        Trading cost per position change.
    periods_per_year:
        Used in risk metrics.

    Returns
    -------
    Dict[str, BacktestResult]
    """
    results: Dict[str, BacktestResult] = {}

    for name, (func, kwargs) in strategies.items():
        result = run_single_strategy(
            name=name,
            bt_df=bt_df,
            position_func=func,
            fee_rate=fee_rate,
            periods_per_year=periods_per_year,
            **(kwargs or {}),
        )
        results[name] = result

    return results
