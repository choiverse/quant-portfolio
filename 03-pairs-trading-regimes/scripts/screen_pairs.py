"""Stage 1 of the pipeline: screen the universe for tradable pairs.

Runs the walk-forward screen and writes the evidence needed to judge it — not
just which pairs were selected, but how many tests were run to find them and
how many rejections the null alone would have produced.

    python scripts/screen_pairs.py
    python scripts/screen_pairs.py --data data/sample_prices.csv --outdir /tmp/smoke
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from _common import base_parser, ensure_dirs, load_panel, saved
from statarb import data as sdata, pairs, plotting, strategy  # noqa: E402


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--top-k", type=int, default=1000,
                   help="candidates carried from the distance screen into the "
                        "cointegration test, per window")
    p.add_argument("--max-pairs", type=int, default=20,
                   help="pairs actually traded per window")
    args = p.parse_args()

    _, figures, tables = ensure_dirs(args.outdir)

    print("[1/4] loading the panel")
    _, prices, _ = load_panel(args.data)
    log_prices = sdata.to_log_prices(prices)
    print(f"      {prices.shape[1]} names with a complete history, "
          f"{prices.shape[0]} trading days "
          f"({prices.index[0].date()} .. {prices.index[-1].date()})")
    print(f"      {pairs.n_possible_pairs(prices.shape[1]):,} possible pairs")

    print("[2/4] walk-forward screening")
    wf = strategy.run_walk_forward(
        log_prices, top_k=args.top_k, max_pairs=args.max_pairs, verbose=True
    )

    window_table = wf.window_table
    selected = wf.selected_pairs
    candidates = pd.concat(
        [w.candidates.assign(trading_start=w.trading_start) for w in wf.windows],
        ignore_index=True,
    )

    print("[3/4] writing tables")
    multiplicity = pd.DataFrame(
        [
            {
                "quantity": "pairs possible in the universe",
                "value": int(window_table["pairs_possible"].iloc[0]),
            },
            {
                "quantity": "pairs tested (all windows)",
                "value": int(window_table["pairs_tested"].sum()),
            },
            {
                "quantity": "rejections at 5%",
                "value": int(window_table["pairs_passed"].sum()),
            },
            {
                "quantity": "rejections expected under the null",
                "value": float(window_table["expected_false_positives"].sum()),
            },
            {
                "quantity": "rejections expected had every pair been tested",
                "value": float(window_table["pairs_possible"].iloc[0]
                               * 0.05 * len(window_table)),
            },
            {
                "quantity": "pairs actually traded",
                "value": int(window_table["pairs_traded"].sum()),
            },
        ]
    )

    for name, frame in (
        ("window_screen", window_table),
        ("selected_pairs", selected),
        ("screen_multiplicity", multiplicity),
        ("candidate_tests", candidates),
    ):
        path = tables / f"{name}.csv"
        frame.to_csv(path, index=False, float_format="%.6g")
        saved(path)

    print("[4/4] figures")
    fig1 = figures / "01_screening.png"
    plotting.screening_figure(
        candidates=candidates,
        window_table=window_table,
        n_possible=int(window_table["pairs_possible"].iloc[0]),
        traded_half_life=selected["half_life"],
        savepath=str(fig1),
    )
    saved(fig1)

    # Illustrate the mechanics on a *representative* pair, not the best one.
    # The most strongly cointegrated pair in the sample is always the fastest
    # reverting one — the Engle-Granger statistic rewards exactly that — and
    # its half-life of a day or two makes it the least typical thing to show.
    # Take the pair whose half-life is closest to the median of those traded.
    if not selected.empty:
        median_hl = selected["half_life"].median()
        pick = (selected["half_life"] - median_hl).abs().idxmin()
        row = selected.loc[pick]
        window_start = row["trading_start"]
        spec = next(s for s in wf.specs[window_start] if s.name == row["pair"])
        report = next(w for w in wf.windows if w.trading_start == window_start)
        span = log_prices.loc[report.formation_start:report.trading_end]
        spread = pairs.spread_series(span, spec)
        z = pairs.zscore(spread, spec)
        pos = pairs.pair_positions(z)

        fig2 = figures / "02_spread_example.png"
        plotting.spread_figure(
            spread=spread, z=z, position=pos, spec=spec,
            formation_end=report.formation_end, savepath=str(fig2),
        )
        saved(fig2)
        print(f"      example pair {spec.name}: EG {spec.stat:.2f}, "
              f"half-life {spec.half_life:.0f}d")

    total_tested = int(window_table["pairs_tested"].sum())
    total_passed = int(window_table["pairs_passed"].sum())
    expected = float(window_table["expected_false_positives"].sum())
    print(f"\n{total_passed:,} of {total_tested:,} tests rejected at 5%; "
          f"{expected:,.0f} were expected under the null "
          f"({total_passed / expected:.1f}x).")


if __name__ == "__main__":
    main()
