"""Stage 3 — the backtest itself, plus the cost and robustness studies.

Runs three books (12-1 momentum, 1-week reversal, equal-weight benchmark) and
writes the tearsheet along with the figures that say how much of the result
survives contact with trading frictions and parameter choices.

    reports/tearsheet.png                        (headline figure)
    reports/figures/04_cost_analysis.png
    reports/figures/05_rolling_risk.png
    reports/figures/06_robustness.png
    reports/metrics_summary.csv
    reports/tables/cost_sensitivity.csv
    reports/tables/turnover_summary.csv
    reports/tables/parameter_grid_*.csv

Usage
-----
    python scripts/run_backtest.py --data data/all_stocks_5yr.csv --cost-bps 10
    python scripts/run_backtest.py --quick        # skip the parameter grid
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from _common import base_parser, ensure_dirs, load_panel, saved

from quantbt import diagnostics, metrics, signals
from quantbt.backtest import Backtester
from quantbt.plotting import (
    cost_analysis_figure,
    robustness_figure,
    rolling_risk_figure,
    tearsheet,
)


def main() -> None:
    p = base_parser("Systematic cross-sectional equity backtest.")
    p.add_argument("--cost-bps", type=float, default=10.0)
    p.add_argument("--rebalance", type=int, default=21)
    p.add_argument("--quantile", type=float, default=0.2)
    p.add_argument("--quick", action="store_true",
                   help="skip the parameter-grid sweep (much faster)")
    args = p.parse_args()
    REPORTS, FIGURES, TABLES = ensure_dirs(args.outdir)

    print(f"[1/6] Loading {args.data} ...")
    raw, prices, rets = load_panel(args.data)
    print(f"      {prices.shape[1]} tickers x {prices.shape[0]} trading days "
          f"({prices.index.min().date()} -> {prices.index.max().date()})")

    print("[2/6] Building signals ...")
    mom_score = signals.cross_sectional_momentum(prices, lookback=252, skip=21)
    rev_score = signals.short_term_reversal(prices, lookback=5)

    mom_w = signals.long_short_weights(mom_score, quantile=args.quantile)
    rev_w = signals.long_short_weights(rev_score, quantile=args.quantile)
    bench_w = signals.long_only_benchmark(prices)

    print("[3/6] Running backtests ...")
    bt = Backtester(rets, rebalance_every=args.rebalance, cost_bps=args.cost_bps)
    bt_fast = Backtester(rets, rebalance_every=5, cost_bps=args.cost_bps)  # weekly signal
    bt_bench = Backtester(rets, rebalance_every=args.rebalance, cost_bps=0.0)

    mom = bt.run(mom_w, name="Momentum (12-1)")
    rev = bt_fast.run(rev_w, name="Reversal (1w)")
    bench = bt_bench.run(bench_w, name="Equal-Weight Benchmark")

    results = {r.name: r.returns for r in (mom, rev, bench)}
    traded = {mom.name: mom, rev.name: rev}

    summ = pd.DataFrame(
        {label: metrics.performance_summary(r, name=label) for label, r in results.items()}
    )
    summ.to_csv(REPORTS / "metrics_summary.csv")

    print("[4/6] Cost sensitivity ...")
    cost_grid = np.arange(0.0, 41.0, 2.0)
    sens = {
        mom.name: diagnostics.cost_sensitivity(mom_w, rets, cost_grid, args.rebalance),
        rev.name: diagnostics.cost_sensitivity(rev_w, rets, cost_grid, 5),
    }
    pd.concat(sens, axis=1).to_csv(TABLES / "cost_sensitivity.csv")

    turnover_rows = {}
    for name, res in traded.items():
        to = res.turnover[res.turnover > 0]
        be = diagnostics.breakeven_cost(sens[name])
        turnover_rows[name] = pd.Series({
            "Avg one-way turnover per rebalance": to.mean(),
            "Median turnover": to.median(),
            "Rebalances": float(len(to)),
            "Annualized turnover": to.mean() * (252 / (args.rebalance if name == mom.name else 5)),
            "Gross Sharpe": metrics.sharpe_ratio(res.gross_returns),
            "Net Sharpe": metrics.sharpe_ratio(res.returns),
            "Break-even cost (bps)": be,
        })
    turnover = pd.DataFrame(turnover_rows)
    turnover.to_csv(TABLES / "turnover_summary.csv")

    print("[5/6] Drawing figures ...")
    tearsheet(
        results,
        flagship=mom.name,
        title="Cross-Sectional Equity Strategies (S&P 500, 2013–2018)",
        savepath=str(REPORTS / "tearsheet.png"),
    )
    cost_analysis_figure(traded, sens, savepath=str(FIGURES / "04_cost_analysis.png"))
    rolling_risk_figure(results, flagship=mom.name,
                        savepath=str(FIGURES / "05_rolling_risk.png"))

    if args.quick:
        print("[6/6] Skipping parameter grid (--quick).")
    else:
        print("[6/6] Parameter robustness sweep (this is the slow part) ...")
        grids = {}
        grids["Momentum — lookback x quantile"] = diagnostics.parameter_grid(
            prices, rets,
            signal_fn=lambda px, lb: signals.cross_sectional_momentum(px, lookback=lb, skip=21),
            lookbacks=[63, 126, 189, 252, 378],
            quantiles=[0.1, 0.2, 0.3, 0.4],
            weight_fn=signals.long_short_weights,
            rebalance_every=args.rebalance,
            cost_bps=args.cost_bps,
        )
        grids["Reversal — lookback x quantile"] = diagnostics.parameter_grid(
            prices, rets,
            signal_fn=lambda px, lb: signals.short_term_reversal(px, lookback=lb),
            lookbacks=[1, 3, 5, 10, 21],
            quantiles=[0.1, 0.2, 0.3, 0.4],
            weight_fn=signals.long_short_weights,
            rebalance_every=5,
            cost_bps=args.cost_bps,
        )
        for name, g in grids.items():
            g.to_csv(TABLES / f"parameter_grid_{name.split()[0].lower()}.csv")
        robustness_figure(grids, savepath=str(FIGURES / "06_robustness.png"))

    print("\n=== Performance summary ===")
    with pd.option_context("display.float_format", lambda x: f"{x:0.3f}"):
        print(summ)

    print("\n=== Turnover & cost ===")
    with pd.option_context("display.float_format", lambda x: f"{x:0.3f}"):
        print(turnover)

    for path in (REPORTS / "tearsheet.png", REPORTS / "metrics_summary.csv",
                 FIGURES / "04_cost_analysis.png", FIGURES / "05_rolling_risk.png",
                 TABLES / "cost_sensitivity.csv", TABLES / "turnover_summary.csv"):
        saved(path)
    if not args.quick:
        saved(FIGURES / "06_robustness.png")


if __name__ == "__main__":
    main()
