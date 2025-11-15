# btc_engine/risk.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class RiskMetrics:
    """
    Container for basic portfolio statistics.
    All returns are in decimal form (e.g. 0.10 = +10%).
    """
    cumulative_return: float
    annualized_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    num_trades: int
    win_rate: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def compute_cumulative_return(returns: pd.Series) -> float:
    """Total cumulative return = Π(1 + r_t) - 1."""
    if returns.empty:
        return 0.0
    return float((1.0 + returns).prod() - 1.0)


def compute_annualized_volatility(
    returns: pd.Series,
    periods_per_year: int = 365,
) -> float:
    """Annualized volatility based on daily returns."""
    if returns.empty:
        return 0.0
    daily_vol = float(returns.std(ddof=1))
    return daily_vol * (periods_per_year ** 0.5)


def compute_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 365,
) -> float:
    """
    Sharpe ratio with risk-free rate expressed as annualized.

    Parameters
    ----------
    returns:
        Daily returns.
    risk_free_rate:
        Annual risk-free rate (e.g. 0.02 for 2%).
    periods_per_year:
        Number of return observations per year.
    """
    if returns.empty:
        return 0.0

    daily_rf = risk_free_rate / periods_per_year
    excess_returns = returns - daily_rf

    mean_excess = float(excess_returns.mean())
    vol = float(excess_returns.std(ddof=1))

    if vol == 0.0:
        return 0.0

    return mean_excess / vol * (periods_per_year ** 0.5)


def compute_max_drawdown(returns: pd.Series) -> float:
    """
    Max drawdown as a positive number (e.g. 0.35 = 35% peak-to-trough loss).

    Uses the equity curve built from the return series:
        equity_t = Π(1 + r_i) for i <= t
    """
    if returns.empty:
        return 0.0

    equity_curve = (1.0 + returns).cumprod()
    running_max = equity_curve.cummax()
    drawdowns = equity_curve / running_max - 1.0  # negative during drawdowns
    max_dd = float(drawdowns.min())
    return -max_dd  # return positive magnitude


def compute_num_trades(positions: pd.Series) -> int:
    """
    Approximate number of trades executed.

    Each change in position is considered a trade event.
    """
    if positions.empty:
        return 0
    changes = positions.diff().abs()
    # Ignore NaN on first period
    num = int(changes.fillna(0.0).astype(bool).sum())
    return num


def compute_win_rate(returns: pd.Series) -> float:
    """
    Fraction of periods with strictly positive return.
    """
    if returns.empty:
        return 0.0
    wins = (returns > 0.0).sum()
    total = returns.count()
    return float(wins / total) if total > 0 else 0.0


def compute_risk_metrics(
    returns: pd.Series,
    positions: pd.Series,
    periods_per_year: int = 365,
    risk_free_rate: float = 0.0,
) -> RiskMetrics:
    """
    Compute the full set of risk metrics for a strategy.

    Parameters
    ----------
    returns:
        Daily strategy returns in decimal form.
    positions:
        Daily positions (used for number of trades).
    periods_per_year:
        Number of observations per year (365 for daily).
    risk_free_rate:
        Annualized risk-free rate.

    Returns
    -------
    RiskMetrics
    """
    cum_ret = compute_cumulative_return(returns)
    ann_vol = compute_annualized_volatility(returns, periods_per_year=periods_per_year)
    sharpe = compute_sharpe_ratio(
        returns, risk_free_rate=risk_free_rate, periods_per_year=periods_per_year
    )
    max_dd = compute_max_drawdown(returns)
    n_trades = compute_num_trades(positions)
    win = compute_win_rate(returns)

    return RiskMetrics(
        cumulative_return=cum_ret,
        annualized_volatility=ann_vol,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        num_trades=n_trades,
        win_rate=win,
    )
