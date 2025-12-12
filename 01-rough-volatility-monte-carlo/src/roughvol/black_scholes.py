"""Black-Scholes closed form and implied-volatility inversion.

These are the *validation anchors* for the Monte Carlo engine: any simulator
must reproduce the Black-Scholes price in the flat-volatility limit, and the
smile plots require inverting model prices back into implied vols.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def bs_price(S0, K, T, r, sigma, option="call"):
    """Black-Scholes-Merton price of a European vanilla option.

    Vectorized over K/sigma (NumPy broadcasting). ``sigma`` is the annualized
    volatility, ``r`` the continuously-compounded rate.
    """
    S0, K, T, sigma = map(np.asarray, (S0, K, T, sigma))
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
    call = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    if option == "call":
        price = call
    else:  # put via put-call parity
        price = call - S0 + K * np.exp(-r * T)
    # Intrinsic value in the sigma -> 0 / T -> 0 corner.
    intrinsic = np.maximum(S0 - K, 0.0) if option == "call" else np.maximum(K - S0, 0.0)
    return np.where((sigma <= 0) | (T <= 0), intrinsic, price)


def bs_vega(S0, K, T, r, sigma):
    """dPrice/dSigma — used as a sanity/plotting helper."""
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S0 * norm.pdf(d1) * np.sqrt(T)


def implied_vol(price, S0, K, T, r, option="call", lo=1e-4, hi=5.0):
    """Invert a European option price to its Black-Scholes implied volatility.

    Uses Brent's method between ``lo`` and ``hi``. Returns NaN when the price is
    outside the no-arbitrage bounds (so bad points drop out of smile plots
    instead of throwing).
    """
    intrinsic = (
        max(S0 - K * np.exp(-r * T), 0.0)
        if option == "call"
        else max(K * np.exp(-r * T) - S0, 0.0)
    )
    upper = S0 if option == "call" else K * np.exp(-r * T)
    if not (intrinsic - 1e-10 < price < upper):
        return np.nan

    def objective(sig):
        return float(bs_price(S0, K, T, r, sig, option)) - price

    try:
        return brentq(objective, lo, hi, maxiter=200, xtol=1e-8)
    except ValueError:
        return np.nan
