"""Build implied-volatility smiles/skews from a model via Monte Carlo.

The economically interesting output of a stochastic-volatility model is the
*shape* of the implied-vol surface it generates. Rough volatility is prized
because it reproduces the steep short-maturity skew seen in real markets, with
an at-the-money skew that decays as a power law T^{H-1/2}.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .black_scholes import implied_vol
from .mc_pricer import price_strikes


def smile_for_maturity(model_factory, T, strikes, n_paths=200_000, seed=0):
    """Implied-vol smile at one maturity (single simulation, all strikes).

    ``model_factory`` is a callable T -> model. Returns a DataFrame indexed by
    strike with price and implied vol.
    """
    model = model_factory(T)
    F = model.S0 * np.exp(model.r * T)
    # Price the out-of-the-money option at each strike (put below F, call above);
    # OTM options carry the reliable Monte Carlo signal on each wing.
    opts = ["put" if K < F else "call" for K in strikes]
    ests = price_strikes(model, strikes, option=opts, n_paths=n_paths, seed=seed)
    rows = []
    for K, opt, est in zip(strikes, opts, ests):
        # Drop points whose price is not statistically distinguishable from zero.
        reliable = est.price > 3 * est.std_error
        iv = implied_vol(est.price, model.S0, K, T, model.r, option=opt) if reliable else np.nan
        rows.append({"K": K, "log_moneyness": np.log(K / model.S0),
                     "price": est.price, "iv": iv})
    return pd.DataFrame(rows).set_index("K")


def atm_skew(model_factory, T, n_paths=300_000, seed=0, bump=0.05):
    """At-the-money implied-vol skew d(IV)/d(log-moneyness) near moneyness 0.

    Central difference across two strikes bracketing the forward, both priced
    off the same simulation.
    """
    model = model_factory(T)
    F = model.S0 * np.exp(model.r * T)
    K_lo, K_hi = F * np.exp(-bump), F * np.exp(bump)
    ests = price_strikes(model, [K_lo, K_hi], option="call", n_paths=n_paths, seed=seed)
    iv_lo = implied_vol(ests[0].price, model.S0, K_lo, T, model.r, "call")
    iv_hi = implied_vol(ests[1].price, model.S0, K_hi, T, model.r, "call")
    dk = np.log(K_hi / model.S0) - np.log(K_lo / model.S0)
    return (iv_hi - iv_lo) / dk


def skew_term_structure(model_factory, maturities, n_paths=300_000, seed=0):
    """ATM skew across maturities, plus the fitted power-law exponent.

    Returns (DataFrame[T, skew], fitted_slope). Rough-volatility theory predicts
    log|skew| ~ (H - 1/2) log T for small T, so the fitted slope should be close
    to H - 1/2.
    """
    rows = [{"T": T, "skew": atm_skew(model_factory, T, n_paths, seed)} for T in maturities]
    df = pd.DataFrame(rows).set_index("T")
    mask = df["skew"].abs() > 0
    slope = np.polyfit(np.log(df.index[mask]), np.log(df["skew"].abs()[mask]), 1)[0]
    return df, float(slope)
