# btc_engine/advisor.py
from __future__ import annotations
from .data_loader import load_backtest_dataframe

import logging
from dataclasses import asdict
from typing import Dict, List

import numpy as np
import pandas as pd

from . import strategies as strat
from .backtest import BacktestResult, run_strategies
from .config import (
    DEFAULT_FEE_RATE,
    DEFAULT_FORECAST_HORIZON_DAYS,
    MODEL_PATH,
    RISK_PROFILE_STRATEGIES,
    TRADING_DAYS_PER_YEAR,
)
from .data_loader import load_backtest_dataframe
from .prophet_model import load_prophet_model, forecast_price_path

logger = logging.getLogger(__name__)
# Leave handler configuration to the application, but avoid "No handler" warnings.
if not logger.handlers:
    logger.addHandler(logging.NullHandler())


VALID_RISK_PROFILES = {"conservative", "balanced", "aggressive"}


def _build_strategy_specs() -> Dict[str, tuple]:
    """
    Internal helper defining the strategies available to the advisor.

    Returns
    -------
    Dict[str, tuple]
        name -> (position_function, kwargs_dict)
    """
    return {
        "buy_and_hold": (strat.buy_and_hold_positions, {}),
        "directional": (strat.directional_positions, {}),
        "threshold": (strat.threshold_positions, {"threshold": 0.01}),  # 1% instead of 0.5%

        "high_confidence": (
            strat.high_confidence_positions,
            {"min_abs_return": 0.01},  # 1% min absolute return
        ),
    }


def _select_strategy_for_profile(
    risk_profile: str,
    strategy_results: Dict[str, BacktestResult],
) -> BacktestResult:
    """
    Select the recommended strategy for a given risk profile.

    Rules:
    - Consider only strategies admissible for the risk profile.
    - Among them, keep only strategies with positive cumulative return.
    - Select the one with highest Sharpe ratio.
    - If no candidate beats Buy & Hold, fall back to Buy & Hold.

    Parameters
    ----------
    risk_profile:
        "conservative", "balanced" or "aggressive".
    strategy_results:
        Mapping of strategy name -> BacktestResult for all strategies.

    Returns
    -------
    BacktestResult
    """
    if risk_profile not in RISK_PROFILE_STRATEGIES:
        raise ValueError(
            f"Unknown risk profile '{risk_profile}'. "
            f"Valid values: {sorted(RISK_PROFILE_STRATEGIES.keys())}"
        )

    admissible = RISK_PROFILE_STRATEGIES[risk_profile]
    logger.info("Risk profile '%s' -> admissible strategies: %s", risk_profile, admissible)

    buy_hold_result = strategy_results["buy_and_hold"]

    # Filter to admissible strategies with positive cumulative return
    candidates: List[BacktestResult] = []
    for name in admissible:
        result = strategy_results[name]
        if result.risk.cumulative_return > 0.0:
            candidates.append(result)

    if not candidates:
        logger.info(
            "No admissible strategy with positive cumulative return. "
            "Recommending Buy & Hold."
        )
        return buy_hold_result

    # Pick best Sharpe ratio among candidates
    best = max(candidates, key=lambda r: r.risk.sharpe_ratio)

    # If best does not beat Buy & Hold on cumulative return, fallback
    if best.risk.cumulative_return <= buy_hold_result.risk.cumulative_return:
        logger.info(
            "Best admissible strategy (%s) does not beat Buy & Hold on cumulative "
            "return. Recommending Buy & Hold.",
            best.name,
        )
        return buy_hold_result

    logger.info(
        "Selected strategy '%s' for risk profile '%s' (Sharpe=%.2f, cum_ret=%.2f%%)",
        best.name,
        risk_profile,
        best.risk.sharpe_ratio,
        best.risk.cumulative_return * 100.0,
    )
    return best


def _simulate_capital_distribution(
    capital_usd: float,
    horizon_days: int,
    strategy_result: BacktestResult,
    forecast_df: pd.DataFrame,
    n_paths: int = 500,
) -> Dict[str, float]:
    """
    Simple Monte-Carlo around daily returns, combining historical strategy
    behaviour and the Prophet forecast as a gentle bias.

    The goal is to give a *rough* educational range, not a precise forecast.

    Parameters
    ----------
    capital_usd:
        Initial capital.
    horizon_days:
        Investment horizon in days.
    strategy_result:
        BacktestResult for the chosen strategy.
    forecast_df:
        Output of forecast_price_path() with predicted_return columns.
    n_paths:
        Number of Monte-Carlo paths.

    Returns
    -------
    dict
        Keys: capital_initial, capital_expected, capital_pessimistic, capital_optimistic
    """
    returns_hist = strategy_result.returns.dropna()

    if returns_hist.empty:
        # Degenerate case: no history, keep capital flat.
        capital_expected = capital_usd
        return {
            "capital_initial": capital_usd,
            "capital_expected": capital_expected,
            "capital_pessimistic": capital_expected,
            "capital_optimistic": capital_expected,
        }

    # Historical daily stats
    mu_hist = float(returns_hist.mean())
    sigma_hist = float(returns_hist.std(ddof=1))

    # Prophet forecast average return over horizon
    if "predicted_return" in forecast_df.columns and not forecast_df["predicted_return"].empty:
        mu_fore = float(forecast_df["predicted_return"].mean())
    else:
        mu_fore = mu_hist

    # Blend historical behaviour and forecast (50/50)
    mu_eff = 0.5 * mu_hist + 0.5 * mu_fore
    sigma_eff = sigma_hist

    if sigma_eff <= 0.0:
        # No volatility -> deterministic compounding
        final_capital = float(capital_usd * (1.0 + mu_eff) ** horizon_days)
        return {
            "capital_initial": capital_usd,
            "capital_expected": final_capital,
            "capital_pessimistic": final_capital,
            "capital_optimistic": final_capital,
        }

    # Monte-Carlo simulation of horizon_days daily returns
    rng = np.random.default_rng(seed=42)
    daily_returns = rng.normal(
        loc=mu_eff,
        scale=sigma_eff,
        size=(n_paths, horizon_days),
    )
    path_multipliers = (1.0 + daily_returns).prod(axis=1)
    final_capitals = capital_usd * path_multipliers

    capital_expected = float(np.median(final_capitals))
    capital_pessimistic = float(np.percentile(final_capitals, 5))
    capital_optimistic = float(np.percentile(final_capitals, 95))

    return {
        "capital_initial": capital_usd,
        "capital_expected": capital_expected,
        "capital_pessimistic": capital_pessimistic,
        "capital_optimistic": capital_optimistic,
    }


def recommend_investment_plan(
    capital_usd: float,
    risk_profile: str = "balanced",
    horizon_days: int = DEFAULT_FORECAST_HORIZON_DAYS,
    fee_rate: float = DEFAULT_FEE_RATE,
    model_path: str = str(MODEL_PATH),
) -> Dict[str, object]:
    """
    Core investor-facing API.

    Parameters
    ----------
    capital_usd:
        Initial investment capital in USD.
    risk_profile:
        One of {"conservative", "balanced", "aggressive"}.
    horizon_days:
        Forward horizon for the plan (in days).
    fee_rate:
        Trading fee rate used for backtests (e.g. 0.001 = 0.1%).
    model_path:
        Path to the saved Prophet model.

    Returns
    -------
    dict
        {
          "inputs": {...},
          "strategy": {
              "name": ...,
              "risk_metrics": {...},
          },
          "simulation": {
              "capital_initial": ...,
              "capital_expected": ...,
              "capital_pessimistic": ...,
              "capital_optimistic": ...,
          },
          "messages": [str, ...],
        }
    """
    if capital_usd <= 0.0:
        raise ValueError("capital_usd must be positive.")
    if risk_profile not in VALID_RISK_PROFILES:
        raise ValueError(
            f"risk_profile must be one of {sorted(VALID_RISK_PROFILES)}, "
            f"got '{risk_profile}'."
        )
    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive.")

    logger.info(
        "Starting investment plan recommendation: capital=%.2f, profile=%s, horizon=%d",
        capital_usd,
        risk_profile,
        horizon_days,
    )

    # 1. Load Prophet model
    logger.info("Loading Prophet model from %s", model_path)
    model = load_prophet_model(model_path)

    # 2. Forecast day-ahead returns over the horizon
    logger.info("Computing Prophet forecast for %d days ahead", horizon_days)
    forecast_df = forecast_price_path(model, periods=horizon_days)

    # 3. Build backtest dataframe from BTC-USD.csv and the loaded model
    logger.info("Building backtest dataframe from BTC-USD.csv using Prophet model")
    bt_df = load_backtest_dataframe(model=model)


    strategy_specs = _build_strategy_specs()
    logger.info("Running %d strategies on historical data", len(strategy_specs))
    strategy_results = run_strategies(
        bt_df=bt_df,
        strategies={name: (func, kwargs) for name, (func, kwargs) in strategy_specs.items()},
        fee_rate=fee_rate,
        periods_per_year=TRADING_DAYS_PER_YEAR,
    )

    # 4. Select strategy for this risk profile
    chosen_result = _select_strategy_for_profile(risk_profile, strategy_results)

    # 5. Simulate investing capital over the horizon using blended stats
    simulation = _simulate_capital_distribution(
        capital_usd=capital_usd,
        horizon_days=horizon_days,
        strategy_result=chosen_result,
        forecast_df=forecast_df,
    )

    # 6. Build human-readable messages (educational, non-hype)
    m = chosen_result.risk
    messages: List[str] = []

    messages.append(
        (
            f"With a {risk_profile} risk profile and {capital_usd:,.0f} USD over "
            f"{horizon_days} days, this engine recommends the strategy "
            f"'{chosen_result.name}'."
        )
    )

    messages.append(
        (
            f"Based on historical data and a simple Prophet forecast, a typical final "
            f"capital could be around {simulation['capital_expected']:,.0f} USD "
            f"(pessimistic ≈ {simulation['capital_pessimistic']:,.0f} USD, "
            f"optimistic ≈ {simulation['capital_optimistic']:,.0f} USD)."
        )
    )

    messages.append(
        (
            f"Over the backtest period, '{chosen_result.name}' achieved a cumulative "
            f"return of {m.cumulative_return:.1%}, annualized volatility of "
            f"{m.annualized_volatility:.1%}, Sharpe ratio {m.sharpe_ratio:.2f}, "
            f"max drawdown {m.max_drawdown:.1%}, win rate {m.win_rate:.1%}, "
            f"with {m.num_trades} trades executed."
        )
    )

    # Explicit honesty about limitations
    if chosen_result.name == "buy_and_hold":
        messages.append(
            "None of the simple Prophet-based trading rules clearly beat a passive "
            "buy-and-hold strategy on past data, so this engine recommends sticking "
            "with buy-and-hold for this risk profile."
        )
    else:
        messages.append(
            "The recommended rule outperformed buy-and-hold historically for this "
            "risk profile, but past performance does NOT guarantee future results."
        )

    messages.append(
        "This tool is purely educational. It uses a basic Prophet model and simple "
        "backtests, and it is NOT financial advice or a guarantee of profit."
    )

    result: Dict[str, object] = {
        "inputs": {
            "capital_usd": capital_usd,
            "risk_profile": risk_profile,
            "horizon_days": horizon_days,
            "fee_rate": fee_rate,
            "model_path": model_path,
        },
        "strategy": {
            "name": chosen_result.name,
            "risk_metrics": asdict(chosen_result.risk),
        },
        "simulation": simulation,
        "messages": messages,
    }

    return result
