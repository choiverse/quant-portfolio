"""Stage 2 — interrogate the signals themselves, before portfolio construction.

An equity curve confounds the signal with the portfolio rules and the cost model
stacked on top of it. This stage strips those away and asks the narrower
question: *does the score rank next month's cross-section at all?*

    reports/figures/03_factor_diagnostics.png
    reports/tables/ic_summary.csv
    reports/tables/ic_decay.csv
    reports/tables/quantile_returns.csv

Usage
-----
    python scripts/factor_diagnostics.py --data data/all_stocks_5yr.csv
"""

from __future__ import annotations

import pandas as pd

from _common import base_parser, ensure_dirs, load_panel, saved

from quantbt import diagnostics, signals
from quantbt.plotting import factor_diagnostics_figure

HORIZON = 21  # one month forward — matches the momentum book's rebalance


def main() -> None:
    p = base_parser("Signal-level diagnostics (IC, quantiles, decay).")
    p.add_argument("--quantiles", type=int, default=5)
    args = p.parse_args()
    _, FIGURES, TABLES = ensure_dirs(args.outdir)

    print(f"[1/4] Loading {args.data} ...")
    raw, prices, rets = load_panel(args.data)

    print("[2/4] Building signals ...")
    scores = {
        "Momentum (12-1)": signals.cross_sectional_momentum(prices, lookback=252, skip=21),
        "Reversal (1w)": signals.short_term_reversal(prices, lookback=5),
    }

    # The momentum score needs a 252-day lookback before it exists at all. Compare
    # quantile books on the sample where *every* signal is live, otherwise the
    # bars differ because of their start dates rather than their content.
    common_start = max(
        s.dropna(how="all").index[0] for s in scores.values()
    )
    print(f"      common evaluation sample starts {common_start:%Y-%m-%d}")
    rets_common = rets.loc[common_start:]

    print("[3/4] Computing IC, decay and quantile portfolios ...")
    ic_series, ic_stats, decay, quantile_ann, quantile_daily = {}, {}, {}, {}, {}
    for name, score in scores.items():
        ic = diagnostics.information_coefficient(score, rets, horizon=HORIZON)
        ic_series[name] = ic
        ic_stats[name] = diagnostics.ic_summary(ic)
        decay[name] = diagnostics.ic_decay(score, rets)
        daily, ann = diagnostics.quantile_performance(
            score.loc[common_start:], rets_common,
            n_quantiles=args.quantiles, rebalance_every=21, cost_bps=0.0,
        )
        quantile_ann[name] = ann
        quantile_daily[name] = daily
        print(f"      {name:18s} mean IC {ic.mean():+.4f}  IC IR {ic.mean()/ic.std(ddof=1):+.3f}")

    ic_table = pd.DataFrame(ic_stats)
    ic_table.to_csv(TABLES / "ic_summary.csv")
    pd.concat({k: v for k, v in decay.items()}, axis=1).to_csv(TABLES / "ic_decay.csv")
    pd.DataFrame(quantile_ann).to_csv(TABLES / "quantile_returns.csv")

    print("[4/4] Drawing figure ...")
    factor_diagnostics_figure(
        ic_series, quantile_ann, decay,
        savepath=str(FIGURES / "03_factor_diagnostics.png"),
    )

    print("\n=== IC summary (21-day forward, Spearman) ===")
    with pd.option_context("display.float_format", lambda x: f"{x:0.4f}"):
        print(ic_table)

    print(f"\n=== Annualized gross return by quantile (Q1 low → Q{args.quantiles} high) ===")
    with pd.option_context("display.float_format", lambda x: f"{x:0.2%}"):
        print(pd.DataFrame(quantile_ann))

    for path in (FIGURES / "03_factor_diagnostics.png", TABLES / "ic_summary.csv",
                 TABLES / "ic_decay.csv", TABLES / "quantile_returns.csv"):
        saved(path)


if __name__ == "__main__":
    main()
