"""Stage 3 of the pipeline: backtest the pairs and decompose the result by regime.

Produces the headline of the project — the conditional split — plus the cost
sensitivity and the parameter robustness grid that say whether the headline
depends on a lucky choice.

    python scripts/run_backtest.py
    python scripts/run_backtest.py --quick          # skip the robustness grid
    python scripts/run_backtest.py --data data/sample_prices.csv --outdir /tmp/smoke
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from _common import base_parser, ensure_dirs, load_panel, saved
from statarb import (  # noqa: E402
    attribution,
    backtest,
    data as sdata,
    metrics,
    plotting,
    regimes as rg,
    strategy,
    volatility as vol,
)

COST_BPS = 10.0
COST_GRID = [0, 2, 5, 7.5, 10, 15, 20, 30, 50]
ENTRY_GRID = [1.5, 2.0, 2.5, 3.0]
MAXPAIRS_GRID = [10, 20, 40]


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--top-k", type=int, default=1000)
    p.add_argument("--max-pairs", type=int, default=20)
    p.add_argument("--burn-in", type=int, default=504)
    p.add_argument("--quick", action="store_true",
                   help="skip the parameter robustness grid")
    args = p.parse_args()

    _, figures, tables = ensure_dirs(args.outdir)

    print("[1/6] loading the panel")
    _, prices, rets = load_panel(args.data)
    market = sdata.market_return(rets)

    print("[2/6] walk-forward backtest")
    result, wf = strategy.run_strategy(
        prices, cost_bps=COST_BPS,
        top_k=args.top_k, max_pairs=args.max_pairs, verbose=True,
    )

    # Statistics are computed only over days a trading window actually covered.
    # The book is flat by construction before the first trading window and
    # again after the last one, and padding the series with those zeros would
    # flatten the volatility and drag every ratio toward zero.
    start, end = wf.trading_start, wf.trading_end
    net = strategy.trim(result.returns, wf)
    gross = strategy.trim(result.gross_returns, wf)
    turnover = strategy.trim(result.turnover, wf)
    print(f"      traded span {start.date()} .. {end.date()} ({len(net)} days)")

    print("[3/6] regimes")
    labels = rg.causal_labels(market, burn_in=args.burn_in, seed=0)
    ewma = vol.ewma_vol(market)

    print("[4/6] attribution")
    reg_table = attribution.regime_table(net, gross, labels, turnover)
    contrib = attribution.contribution(net, labels)
    turn_profile = attribution.turnover_profile(turnover, labels, cost_bps=COST_BPS)
    print(reg_table.round(4).to_string())

    summary = pd.concat(
        [
            metrics.performance_summary(gross, name="gross"),
            metrics.performance_summary(net, name=f"net of {COST_BPS:g} bps"),
        ],
        axis=1,
    )
    boot = attribution.block_bootstrap_sharpe(net, seed=0)
    extra = pd.DataFrame(
        [
            {"metric": "Breakeven cost (bps)", "value": result.breakeven_cost_bps()},
            {"metric": "Avg daily turnover", "value": float(turnover.mean())},
            {"metric": "Avg gross exposure",
             "value": float(strategy.trim(result.gross_exposure, wf).mean())},
            {"metric": "Net Sharpe s.e.", "value": attribution.sharpe_stderr(net)},
            {"metric": "Net Sharpe boot lo (90%)", "value": boot["lo"]},
            {"metric": "Net Sharpe boot hi (90%)", "value": boot["hi"]},
            {"metric": "Trading days", "value": float(len(net))},
        ]
    )
    print(summary.round(4).to_string())
    print(f"      breakeven {result.breakeven_cost_bps():.2f} bps · "
          f"net Sharpe {metrics.sharpe_ratio(net):+.3f} "
          f"[{boot['lo']:+.2f}, {boot['hi']:+.2f}]")

    print("[5/6] cost sensitivity and robustness")
    sweep = backtest.cost_sweep_from(gross, turnover, COST_GRID)
    assert np.isclose(sweep.loc[COST_BPS, "Sharpe"], metrics.sharpe_ratio(net)), (
        "cost sweep disagrees with the headline at the charged cost"
    )

    grid_rows = []
    if not args.quick:
        for entry in ENTRY_GRID:
            for mp in MAXPAIRS_GRID:
                r, w = strategy.run_strategy(
                    prices, cost_bps=COST_BPS, top_k=args.top_k,
                    max_pairs=mp, entry=entry,
                )
                n_ = strategy.trim(r.returns, w)
                g_ = strategy.trim(r.gross_returns, w)
                t_ = strategy.trim(r.turnover, w)
                per_regime = attribution.regime_table(n_, g_, labels, t_, n_boot=400)
                row = {
                    "entry": entry, "max_pairs": mp,
                    "gross_sharpe": metrics.sharpe_ratio(g_),
                    "net_sharpe": metrics.sharpe_ratio(n_),
                    "breakeven_bps": r.breakeven_cost_bps(),
                    "avg_turnover": float(t_.mean()),
                }
                for reg in ("calm", "turbulent"):
                    row[f"net_sharpe_{reg}"] = (
                        float(per_regime.loc[reg, "net_sharpe"])
                        if reg in per_regime.index else np.nan
                    )
                grid_rows.append(row)
                print(f"      entry {entry:>4} max_pairs {mp:>3}  "
                      f"net {row['net_sharpe']:+.3f}  "
                      f"calm {row['net_sharpe_calm']:+.3f}  "
                      f"turbulent {row['net_sharpe_turbulent']:+.3f}")
    grid = pd.DataFrame(grid_rows)

    print("[6/6] writing tables and figures")
    for name, frame, index in (
        ("metrics_summary", summary, True),
        ("metrics_extra", extra, False),
        ("regime_attribution", reg_table, True),
        ("regime_contribution", contrib, True),
        ("turnover_by_regime", turn_profile, True),
        ("cost_sensitivity", sweep, True),
        ("robustness_grid", grid, False),
        ("window_performance", wf.window_table, False),
    ):
        if frame is None or (hasattr(frame, "empty") and frame.empty):
            continue
        path = tables / f"{name}.csv"
        frame.to_csv(path, index=index, float_format="%.6g")
        saved(path)

    trimmed = backtest.BacktestResult(
        returns=net, gross_returns=gross,
        weights=strategy.trim(result.weights, wf),
        turnover=turnover, costs=strategy.trim(result.costs, wf), name="pairs",
    )

    fig5 = figures / "05_performance.png"
    plotting.performance_figure(trimmed, sweep, charged_bps=COST_BPS,
                                savepath=str(fig5))
    saved(fig5)

    fig6 = figures / "06_regime_attribution.png"
    plotting.attribution_figure(reg_table, contrib, net, labels, savepath=str(fig6))
    saved(fig6)

    reports = figures.parent
    composite = reports / "pairs_regime_results.png"
    plotting.headline_figure(
        trimmed, reg_table, labels.reindex(net.index),
        ewma.reindex(net.index), savepath=str(composite),
    )
    saved(composite)


if __name__ == "__main__":
    main()
