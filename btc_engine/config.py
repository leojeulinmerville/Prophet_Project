# btc_engine/config.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

# Project root = .../Prophet_Project
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

# Raw data & model paths
DATA_PATH: Path = PROJECT_ROOT / "BTC-USD.csv"
MODEL_PATH: Path = PROJECT_ROOT / "models" / "prophet_m_improved_da.joblib"

# Market / risk configuration
TRADING_DAYS_PER_YEAR: int = 365  # crypto trades 24/7, we approximate with 365
RISK_FREE_RATE: float = 0.0       # educational, assume 0 risk-free rate

# Trading costs
DEFAULT_FEE_RATE: float = 0.001   # 0.1% per trade, approximate

# Default forecasting horizon
DEFAULT_FORECAST_HORIZON_DAYS: int = 30

# Risk-profile -> admissible strategies mapping
RISK_PROFILE_STRATEGIES: Dict[str, Tuple[str, ...]] = {
    "conservative": ("buy_and_hold", "high_confidence"),
    "balanced": ("buy_and_hold", "threshold"),
    "aggressive": ("directional", "threshold"),
}
