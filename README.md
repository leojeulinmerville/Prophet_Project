# BTC Prophet Investment Engine (Educational)

This mini-project wraps your existing **day-ahead Prophet model** for Bitcoin into a small,
realistic investment engine:

- Uses your saved Prophet model (`models/prophet_m_improved_da.joblib`) as a **support tool**.
- Reuses your **backtest dataframe** (`bt_df.csv`) exported from the notebook.
- Exposes an **investor-facing API**: input = capital + risk profile, output = recommended
  strategy + projected capital range and risk metrics.
- Stays honest: this is **not** a magic trading bot, only an educational tool.

---

## 1. Install dependencies

In your virtual environment:

```bash
pip install -r requirements.txt
