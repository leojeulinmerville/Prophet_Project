# dashboard.py
from __future__ import annotations

import streamlit as st
import pandas as pd

from btc_engine import recommend_investment_plan
from btc_engine.config import MODEL_PATH, TRADING_DAYS_PER_YEAR
from btc_engine.prophet_model import load_prophet_model
from btc_engine.data_loader import load_backtest_dataframe
from btc_engine.backtest import run_strategies
from btc_engine import strategies as strat
from btc_engine.risk import compute_risk_metrics


def get_all_strategy_results(model, fee_rate: float):
    """
    Utility to compute full metrics for all strategies, for display.
    """
    bt_df = load_backtest_dataframe(model=model)

    strategy_specs = {
        "buy_and_hold": (strat.buy_and_hold_positions, {}),
        "directional": (strat.directional_positions, {}),
        "threshold": (strat.threshold_positions, {"threshold": 0.005}),
        "high_confidence": (strat.high_confidence_positions, {"min_abs_return": 0.01}),
    }

    results = run_strategies(
        bt_df=bt_df,
        strategies=strategy_specs,
        fee_rate=fee_rate,
        periods_per_year=TRADING_DAYS_PER_YEAR,
    )

    return bt_df, results


def main():
    st.set_page_config(
        page_title="BTC Prophet Investment Engine (Educational)",
        layout="wide",
    )

    st.title("PRATMESH PROMISED RICH ! " \
    "🧪 Bitcoin Investment Engine – Testing Dashboard")
    st.caption(
        "Powered by a saved Prophet model and simple backtests. "
        "Educational only – NOT financial advice."
    )

    # === Sidebar inputs ===
    st.sidebar.header("Inputs")

    capital = st.sidebar.number_input(
        "Capital (USD)", min_value=100.0, value=1000.0, step=100.0
    )

    risk_profile = st.sidebar.selectbox(
        "Risk profile", options=["conservative", "balanced", "aggressive"], index=1
    )

    horizon = st.sidebar.slider(
        "Horizon (days)", min_value=7, max_value=180, value=30, step=1
    )

    fee_rate = st.sidebar.slider(
        "Fee rate per trade (decimal)", min_value=0.0, max_value=0.005, value=0.001, step=0.0005
    )

    show_debug = st.sidebar.checkbox("Show debug outputs (JSON, raw metrics)", value=False)

    st.sidebar.markdown("---")
    st.sidebar.caption(f"Model path: `{MODEL_PATH}`")

    # === Load model once per session ===
    @st.cache_resource
    def _load_model_cached():
        return load_prophet_model(MODEL_PATH)

    model = _load_model_cached()

    # === Call main advisor endpoint ===
    if st.button("Run recommendation"):
        result = recommend_investment_plan(
            capital_usd=capital,
            risk_profile=risk_profile,
            horizon_days=horizon,
            fee_rate=fee_rate,
            model_path=str(MODEL_PATH),
        )

        st.subheader("Summary")
        for msg in result["messages"]:
            st.write(msg)

        # Show simulation numbers
        sim = result["simulation"]
        st.subheader("Simulated capital over horizon")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Initial capital", f"{sim['capital_initial']:,.0f} USD")
        c2.metric("Expected", f"{sim['capital_expected']:,.0f} USD")
        c3.metric("Pessimistic (~5th pct.)", f"{sim['capital_pessimistic']:,.0f} USD")
        c4.metric("Optimistic (~95th pct.)", f"{sim['capital_optimistic']:,.0f} USD")

        # === All strategy metrics & equity curves ===
        st.subheader("Strategy comparison (historical backtest)")

        bt_df, all_results = get_all_strategy_results(model=model, fee_rate=fee_rate)

        # Metrics table
        rows = []
        for name, res in all_results.items():
            m = res.risk
            rows.append(
                {
                    "strategy": name,
                    "cum_return_%": m.cumulative_return * 100.0,
                    "ann_vol_%": m.annualized_volatility * 100.0,
                    "sharpe": m.sharpe_ratio,
                    "max_drawdown_%": m.max_drawdown * 100.0,
                    "win_rate_%": m.win_rate * 100.0,
                    "num_trades": m.num_trades,
                }
            )
        metrics_df = pd.DataFrame(rows).set_index("strategy")
        st.dataframe(metrics_df.style.format(precision=2))

        # Equity curves
        st.subheader("Equity curves (normalized, starting at 1.0)")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        for name, res in all_results.items():
            ax.plot(bt_df["date_t"], res.equity_curve, label=name)
        ax.set_xlabel("Date")
        ax.set_ylabel("Equity (x starting capital)")
        ax.legend()
        st.pyplot(fig)

        # Optional debug JSON
        if show_debug:
            st.subheader("Raw engine output (debug)")
            st.json(result)


if __name__ == "__main__":
    main()
