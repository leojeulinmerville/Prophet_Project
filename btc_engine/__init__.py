# btc_engine/__init__.py
"""
btc_engine: Educational Bitcoin investment engine built around a saved Prophet model.

The main public entry point is `recommend_investment_plan`.
"""

from .advisor import recommend_investment_plan

__all__ = ["recommend_investment_plan"]

__version__ = "0.1.0"
