"""Cross-sectional predictors, and the transform that makes them comparable.

Thirteen features, all computable from daily OHLCV alone, grouped into four
families: trend, reversal, risk, and liquidity/attention. The dataset carries
no fundamentals, no sector labels and no share counts, so there is no value
factor, no sector neutralisation and no market cap here — dollar volume stands
in for size, which is a proxy and is flagged as one in the data card.

Two rules apply to every function in this module and are worth stating once:

**Every feature is causal by construction.** A feature dated ``t`` uses only
data through ``t``. All the rolling windows are backward-looking, and none of
them are centred. This is checked directly by a test that rewrites the last row
of the panel and requires every earlier feature value to come back identical.

**Every feature is standardized within the date, not within the series.** The
learner is asked a cross-sectional question — which of today's 470 names will
outperform the others — so a feature's *level* is meaningless and only its rank
among today's names matters. Standardizing across time instead would feed the
model the market's volatility regime through the back door: in a calm month
every name's 21-day volatility is low, and a model trained on raw levels learns
"low vol = predict the calm-market average return", which is a statement about
the calendar rather than about the stock.

The default cross-sectional transform is a **rank** mapped to [-0.5, 0.5]
rather than a z-score. Financial features have heavy tails, and one name with
a 300% turnover spike moves a cross-sectional z-score for every other name in
the book. The rank is immune to that and costs only the magnitude information,
which the tail makes untrustworthy anyway. ``cs_zscore`` is provided too, and
§3 of the write-up reports what changes when it is used instead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------
# Cross-sectional transforms
# --------------------------------------------------------------------------


def cs_rank(frame: pd.DataFrame) -> pd.DataFrame:
    """Rank within each date, mapped to [-0.5, 0.5]. NaNs stay NaN.

    ``(rank - 1) / (n - 1) - 0.5`` where ``n`` is the number of *valid* names
    on that date, so the scale does not drift as coverage changes.
    """
    ranked = frame.rank(axis=1, method="average")
    counts = frame.notna().sum(axis=1)
    denom = (counts - 1).replace(0, np.nan)
    return (ranked.sub(1.0)).div(denom, axis=0) - 0.5


def cs_zscore(frame: pd.DataFrame, clip: float = 5.0) -> pd.DataFrame:
    """Standardize within each date, clipped at ``clip`` standard deviations."""
    mu = frame.mean(axis=1)
    sd = frame.std(axis=1, ddof=1).replace(0, np.nan)
    z = frame.sub(mu, axis=0).div(sd, axis=0)
    return z.clip(-clip, clip)


def cs_demean(frame: pd.DataFrame) -> pd.DataFrame:
    """Subtract the cross-sectional mean, leaving the scale alone."""
    return frame.sub(frame.mean(axis=1), axis=0)


def winsorize(frame: pd.DataFrame, limit: float = 0.01) -> pd.DataFrame:
    """Clip each date's cross-section to its ``[limit, 1-limit]`` quantiles."""
    lo = frame.quantile(limit, axis=1)
    hi = frame.quantile(1.0 - limit, axis=1)
    return frame.clip(lower=lo, upper=hi, axis=0)


# --------------------------------------------------------------------------
# Trend and reversal
# --------------------------------------------------------------------------


def momentum(prices: pd.DataFrame, lookback: int = 252, skip: int = 21) -> pd.DataFrame:
    """Trailing return from ``t-lookback`` to ``t-skip``.

    The ``skip`` is the standard one-month gap that keeps classic momentum
    from being contaminated by short-term reversal — the two effects have
    opposite signs over the most recent month, and a 12-0 momentum signal is
    partly a bet against the reversal factor that appears separately below.
    """
    return prices.shift(skip) / prices.shift(lookback) - 1.0


def reversal(prices: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
    """Negative of the trailing ``lookback``-day return: recent losers score high."""
    return -(prices / prices.shift(lookback) - 1.0)


def distance_from_high(prices: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    """Price relative to its trailing 52-week maximum, in [0, 1].

    A separate effect from momentum despite the obvious correlation — George
    and Hwang (2004) found the proximity-to-high measure carries information
    momentum does not, and the two disagree for any stock that ran up early in
    the window and then went sideways.
    """
    return prices / prices.rolling(window, min_periods=window).max()


# --------------------------------------------------------------------------
# Risk
# --------------------------------------------------------------------------


def realized_vol(returns: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """Annualized trailing standard deviation of daily returns."""
    return returns.rolling(window, min_periods=window).std(ddof=1) * np.sqrt(TRADING_DAYS)


def rolling_beta_idio(
    returns: pd.DataFrame,
    market: pd.Series,
    window: int = 252,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rolling market beta and the residual (idiosyncratic) volatility.

    Both come out of the same trailing OLS of the stock on the market, so they
    are computed together rather than twice. For a univariate regression the
    residual variance is ``var(r_i) - beta^2 * var(r_m)`` exactly, which avoids
    ever forming the residual series — a real saving over 470 names.

    Negative residual variances are possible only through floating-point error
    when a stock is almost exactly the market; they are clipped at zero.
    """
    var_m = market.rolling(window, min_periods=window).var(ddof=1)
    cov = returns.rolling(window, min_periods=window).cov(market, ddof=1)
    beta = cov.div(var_m.replace(0, np.nan), axis=0)

    var_i = returns.rolling(window, min_periods=window).var(ddof=1)
    resid_var = var_i.sub(beta.pow(2).mul(var_m, axis=0)).clip(lower=0.0)
    return beta, np.sqrt(resid_var) * np.sqrt(TRADING_DAYS)


def return_skew(returns: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    """Trailing skewness of daily returns — the lottery-preference proxy.

    Stocks with positively skewed returns are, on the Barberis-Huang and Bali
    et al. account, systematically overpriced by investors who like the small
    chance of a large payoff. This is the closest that OHLCV alone gets to
    that effect.
    """
    return returns.rolling(window, min_periods=window).skew()


# --------------------------------------------------------------------------
# Liquidity and attention
# --------------------------------------------------------------------------


def log_dollar_volume(prices: pd.DataFrame, volume: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    """Log of average daily dollar volume — the size proxy this dataset allows.

    Not market cap. Dollar volume correlates with size but also with
    turnover, so this feature carries a liquidity component that a genuine
    size factor would not. The data card says so; there is no share count in
    the file to do better.
    """
    dollar = (prices * volume).rolling(window, min_periods=window).mean()
    return np.log(dollar.where(dollar > 0))


def volume_shock(volume: pd.DataFrame, short: int = 5, long: int = 63) -> pd.DataFrame:
    """Log ratio of recent to normal volume — abnormal trading activity.

    The attention proxy: a name trading at three times its usual volume is
    being looked at, whatever the reason.
    """
    fast = volume.rolling(short, min_periods=short).mean()
    slow = volume.rolling(long, min_periods=long).mean()
    ratio = (fast / slow.replace(0, np.nan)).where(lambda r: r > 0)
    return np.log(ratio)


def amihud_illiquidity(
    returns: pd.DataFrame,
    prices: pd.DataFrame,
    volume: pd.DataFrame,
    window: int = 21,
) -> pd.DataFrame:
    """Amihud (2002): average |return| per dollar traded, in log space.

    The price impact proxy — how far the price moves for a given amount of
    trading. Logged because the raw ratio spans several orders of magnitude
    across the universe.
    """
    dollar = (prices * volume).replace(0, np.nan)
    impact = (returns.abs() / dollar).rolling(window, min_periods=window).mean()
    return np.log(impact.where(impact > 0))


def high_low_range(
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    window: int = 21,
) -> pd.DataFrame:
    """Average intraday range as a fraction of close — a spread/volatility proxy.

    Uses the two OHLCV fields projects 02 and 03 never touched. It carries
    information beyond close-to-close volatility: a name that gaps and then
    recovers within the day has range without daily volatility.
    """
    return ((high - low) / close.replace(0, np.nan)).rolling(window, min_periods=window).mean()


# --------------------------------------------------------------------------
# The feature set
# --------------------------------------------------------------------------

FEATURE_NAMES = [
    "mom_12_1",
    "mom_6_1",
    "rev_5",
    "rev_21",
    "dist_high_52w",
    "vol_21",
    "vol_252",
    "beta_252",
    "idio_vol_252",
    "skew_63",
    "dollar_volume",
    "volume_shock",
    "amihud",
    "hl_range",
]

# Which family each feature belongs to — used by the figures, and by the
# write-up's claim that the model's importance is concentrated in one of them.
FEATURE_FAMILY = {
    "mom_12_1": "trend",
    "mom_6_1": "trend",
    "dist_high_52w": "trend",
    "rev_5": "reversal",
    "rev_21": "reversal",
    "vol_21": "risk",
    "vol_252": "risk",
    "beta_252": "risk",
    "idio_vol_252": "risk",
    "skew_63": "risk",
    "dollar_volume": "liquidity",
    "volume_shock": "liquidity",
    "amihud": "liquidity",
    "hl_range": "liquidity",
}


def build_features(
    panel: dict[str, pd.DataFrame],
    transform: str = "rank",
) -> dict[str, pd.DataFrame]:
    """Build the whole feature set from the raw matrices.

    ``panel`` is the dict returned by ``data.load_panel``. ``transform`` is
    ``"rank"`` (default), ``"zscore"``, or ``"raw"`` — the last one is for the
    exploratory figure that shows what the untransformed distributions look
    like, and is never fed to a model.
    """
    close, volume = panel["close"], panel["volume"]
    high, low = panel["high"], panel["low"]
    returns = close.pct_change()
    market = returns.mean(axis=1)

    beta, idio = rolling_beta_idio(returns, market, window=252)

    raw = {
        "mom_12_1": momentum(close, 252, 21),
        "mom_6_1": momentum(close, 126, 21),
        "rev_5": reversal(close, 5),
        "rev_21": reversal(close, 21),
        "dist_high_52w": distance_from_high(close, 252),
        "vol_21": realized_vol(returns, 21),
        "vol_252": realized_vol(returns, 252),
        "beta_252": beta,
        "idio_vol_252": idio,
        "skew_63": return_skew(returns, 63),
        "dollar_volume": log_dollar_volume(close, volume, 63),
        "volume_shock": volume_shock(volume, 5, 63),
        "amihud": amihud_illiquidity(returns, close, volume, 21),
        "hl_range": high_low_range(high, low, close, 21),
    }

    if transform == "raw":
        return raw
    if transform == "rank":
        return {k: cs_rank(v) for k, v in raw.items()}
    if transform == "zscore":
        return {k: cs_zscore(winsorize(v)) for k, v in raw.items()}
    raise ValueError("transform must be 'rank', 'zscore' or 'raw'")


def build_target(
    close: pd.DataFrame,
    horizon: int = 5,
    neutralize: bool = True,
) -> pd.DataFrame:
    """The label: forward ``horizon``-day return, cross-sectionally demeaned.

    Demeaning is what makes this a *relative* prediction problem, and it is not
    cosmetic. Left in levels, the single most predictable component of every
    stock's forward return is the market's forward return, which is common to
    all of them: a model would spend its capacity forecasting the index and
    score well on R^2 while ranking the cross-section no better than chance.
    The strategy built on top is dollar-neutral, so the market component is
    exactly the part it cannot trade.
    """
    fwd = close.shift(-horizon) / close - 1.0
    return cs_demean(fwd) if neutralize else fwd
