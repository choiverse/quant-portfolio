"""Performance and risk metrics for a daily return series.

Deliberately identical to projects 02 and 03's module, so that a Sharpe ratio
quoted in this project means exactly what a Sharpe ratio quoted in those ones
means — a model that "wins" here must win on the same yardstick.
All functions take a pandas Series of periodic (daily) simple returns and
assume ``periods_per_year`` trading days for annualization (252 by default).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def equity_curve(returns: pd.Series, start: float = 1.0) -> pd.Series:
    """Cumulative growth of 1 unit of capital."""
    return start * (1.0 + returns).cumprod()


def total_return(returns: pd.Series) -> float:
    return float((1.0 + returns).prod() - 1.0)


def cagr(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    n = len(returns)
    if n == 0:
        return np.nan
    growth = (1.0 + returns).prod()
    if growth <= 0:
        return -1.0
    return float(growth ** (periods_per_year / n) - 1.0)


def ann_volatility(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    return float(returns.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series,
    risk_free: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    """Annualized Sharpe ratio. ``risk_free`` is an annual rate."""
    excess = returns - risk_free / periods_per_year
    sd = excess.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return np.nan
    return float(excess.mean() / sd * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: pd.Series,
    risk_free: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    """Sortino ratio — like Sharpe but penalizes only downside deviation."""
    excess = returns - risk_free / periods_per_year
    downside = excess[excess < 0]
    dd = np.sqrt((downside ** 2).mean()) if len(downside) else np.nan
    if not dd or np.isnan(dd):
        return np.nan
    return float(excess.mean() / dd * np.sqrt(periods_per_year))


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Running drawdown (fraction below the prior peak) of the equity curve."""
    eq = equity_curve(returns)
    peak = eq.cummax()
    return eq / peak - 1.0


def max_drawdown(returns: pd.Series) -> float:
    return float(drawdown_series(returns).min())


def calmar_ratio(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    mdd = abs(max_drawdown(returns))
    if mdd == 0:
        return np.nan
    return float(cagr(returns, periods_per_year) / mdd)


def hit_rate(returns: pd.Series) -> float:
    """Fraction of periods with a strictly positive return."""
    nz = returns[returns != 0]
    if len(nz) == 0:
        return np.nan
    return float((nz > 0).mean())


def performance_summary(
    returns: pd.Series,
    risk_free: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
    name: str | None = None,
) -> pd.Series:
    """Bundle the headline metrics into a labelled Series (one strategy)."""
    return pd.Series(
        {
            "Total Return": total_return(returns),
            "CAGR": cagr(returns, periods_per_year),
            "Ann. Volatility": ann_volatility(returns, periods_per_year),
            "Sharpe": sharpe_ratio(returns, risk_free, periods_per_year),
            "Sortino": sortino_ratio(returns, risk_free, periods_per_year),
            "Max Drawdown": max_drawdown(returns),
            "Calmar": calmar_ratio(returns, periods_per_year),
            "Hit Rate": hit_rate(returns),
        },
        name=name or returns.name or "strategy",
    )
