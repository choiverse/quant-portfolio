"""Stage 2 of the pipeline: estimate volatility and split the sample into regimes.

Fits the three volatility estimators and the two-state HMM to the equal-weight
market return, and writes the regime classification the backtest attribution
consumes.

    python scripts/fit_regimes.py
    python scripts/fit_regimes.py --data data/sample_prices.csv --outdir /tmp/smoke
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from _common import base_parser, ensure_dirs, load_panel, saved
from statarb import data as sdata, plotting, regimes as rg, volatility as vol  # noqa: E402


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--burn-in", type=int, default=504,
                   help="days used to fit the HMM before filtering forward; "
                        "nothing after this window informs any label")
    args = p.parse_args()

    _, figures, tables = ensure_dirs(args.outdir)

    print("[1/4] loading the panel")
    _, prices, rets = load_panel(args.data)
    market = sdata.market_return(rets)
    print(f"      market proxy: equal-weight mean of {prices.shape[1]} names, "
          f"{len(market)} days")

    print("[2/4] volatility estimators")
    realized = vol.realized_vol(market, window=21)
    ewma = vol.ewma_vol(market)
    garch = vol.fit_garch11(market)
    print(f"      GARCH(1,1)  omega={garch.omega:.3e}  alpha={garch.alpha:.4f}  "
          f"beta={garch.beta:.4f}")
    print(f"                  persistence {garch.persistence:.4f}  "
          f"variance half-life {garch.half_life:.1f}d  "
          f"long-run vol {np.sqrt(garch.long_run_var * 252):.2%}  "
          f"converged={garch.converged}")

    print("[3/4] hidden Markov regimes")
    full = rg.fit_regimes(market, seed=0)
    summary = full.summary()
    print(summary.round(4).to_string())

    causal = rg.fit_causal(market, burn_in=args.burn_in, seed=0)
    labels = rg.causal_labels(market, burn_in=args.burn_in, seed=0)
    counts = labels.value_counts()
    print(f"      causal labels after a {args.burn_in}-day burn-in: "
          + ", ".join(f"{k} {v}" for k, v in counts.items()))

    tercile = rg.vol_tercile_regimes(ewma)
    agreement = pd.concat([labels, tercile], axis=1).dropna()
    if len(agreement):
        overlap = float(
            ((agreement["regime"] == "turbulent") == (agreement["vol_tercile"] == "high"))
            .mean()
        )
        print(f"      HMM 'turbulent' vs EWMA top tercile agree on {overlap:.1%} of days")

    print("[4/4] writing tables and figures")
    garch_row = pd.DataFrame([{
        "omega": garch.omega, "alpha": garch.alpha, "beta": garch.beta,
        "mu": garch.mu, "persistence": garch.persistence,
        "variance_half_life_days": garch.half_life,
        "long_run_ann_vol": float(np.sqrt(garch.long_run_var * 252)),
        "loglik": garch.loglik, "converged": garch.converged,
    }])

    classification = pd.DataFrame({
        "p_turbulent_causal": causal,
        "regime": labels,
        "p_turbulent_smoothed": full.smoothed,
        "realized_vol": realized,
        "ewma_vol": ewma,
        "garch_vol": garch.sigma,
        "vol_tercile": tercile,
    })

    for name, frame, index in (
        ("regime_params", summary, True),
        ("garch_fit", garch_row, False),
        ("regime_classification", classification, True),
    ):
        path = tables / f"{name}.csv"
        frame.to_csv(path, index=index, float_format="%.6g")
        saved(path)

    fig3 = figures / "03_regimes.png"
    plotting.regime_figure(
        market=market, prob_turbulent=causal, regime=labels,
        summary=summary, savepath=str(fig3),
    )
    saved(fig3)

    fig4 = figures / "04_vol_estimators.png"
    plotting.volatility_figure(
        realized=realized, ewma=ewma, garch=garch.sigma,
        garch_fit=garch, savepath=str(fig4),
    )
    saved(fig4)


if __name__ == "__main__":
    main()
