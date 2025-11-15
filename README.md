# Bitcoin Day-Ahead Prophet Engine (Educational)

This project wraps a saved **Prophet** model for Bitcoin into a small, testable
“investment engine”.

- **Core idea**: use a *day-ahead* Prophet model (`m_improved_da`) as a **support tool**
  to explore simple trading rules, *not* as an autonomous trading bot.
- **Input**: capital, risk profile, horizon.
- **Output**: recommended strategy + historical risk metrics + a rough simulation
  of potential capital over the chosen horizon.

> ⚠️ This project is **purely educational**.  
> It is **not** financial advice, and it is **not** production-grade trading software.
