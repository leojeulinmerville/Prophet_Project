# btc_engine/cli.py
from __future__ import annotations

import argparse
from typing import Optional

from .advisor import recommend_investment_plan
from .config import DEFAULT_FEE_RATE, DEFAULT_FORECAST_HORIZON_DAYS, MODEL_PATH


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Educational Bitcoin investment engine based on a saved Prophet model. "
            "This is NOT financial advice."
        )
    )
    parser.add_argument(
        "--capital",
        type=float,
        required=True,
        help="Initial capital in USD (e.g. 1000).",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="balanced",
        choices=["conservative", "balanced", "aggressive"],
        help="Risk profile (default: balanced).",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=DEFAULT_FORECAST_HORIZON_DAYS,
        help="Investment horizon in days (default: 30).",
    )
    parser.add_argument(
        "--fee-rate",
        type=float,
        default=DEFAULT_FEE_RATE,
        help="Trading fee rate per position change (default: 0.001 = 0.1%%).",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(MODEL_PATH),
        help="Path to the saved Prophet model joblib file.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    result = recommend_investment_plan(
        capital_usd=args.capital,
        risk_profile=args.profile,
        horizon_days=args.horizon,
        fee_rate=args.fee_rate,
        model_path=args.model_path,
    )

    # Print the human-readable messages
    for line in result["messages"]:
        print(line)


if __name__ == "__main__":
    main()
