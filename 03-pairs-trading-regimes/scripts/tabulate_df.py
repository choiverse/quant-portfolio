"""Simulate the Dickey-Fuller null distributions and write the quantile table.

This is a *build* step, not a report step. Its output —
``src/statarb/tables/df_quantiles.csv`` — is a model constant: the table that
turns an ADF statistic into a p-value. It is committed so that the library
works out of the box, and regenerable so that nobody has to take it on faith.

Three null distributions are tabulated, each from random walks that have a
unit root by construction:

  regression="c",  n_series=1   ADF with a constant, on observed data
  regression="ct", n_series=1   ADF with constant and trend
  regression="c",  n_series=2   Engle-Granger residual, cointegrating
                                regression with a constant

The 1/5/10% points of this table are checked against MacKinnon (2010) by
``scripts/validate.py``. Usage:

    python scripts/tabulate_df.py                 # the committed settings
    python scripts/tabulate_df.py --reps 5000     # a quick smoke run
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from _common import ROOT, saved         # sets up sys.path for the import below
from statarb import cointegration as coint  # noqa: E402

# The quantile grid. Denser in the left tail, which is the only region a
# cointegration test ever reads: p-values above ~0.2 are all "no evidence".
QUANTILES = np.array(
    [0.0005, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.025, 0.03,
     0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.125, 0.15, 0.175, 0.20,
     0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
)

CASES = [("c", 1), ("ct", 1), ("c", 2)]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reps", type=int, default=200_000,
                   help="replications per case")
    p.add_argument("--nobs", type=int, default=2_000,
                   help="series length; large enough to approximate the "
                        "asymptotic distribution")
    p.add_argument("--seed", type=int, default=20260726)
    args = p.parse_args()

    outdir = ROOT / "src" / "statarb" / "tables"
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Simulating Dickey-Fuller null distributions "
          f"({args.reps:,} reps x {args.nobs:,} obs per case)\n")

    rows = []
    for i, (regression, n_series) in enumerate(CASES):
        t0 = time.perf_counter()
        stats = coint.simulate_df_stats(
            nobs=args.nobs,
            n_reps=args.reps,
            regression=regression,
            n_series=n_series,
            seed=args.seed + i,
        )
        qs = np.quantile(stats, QUANTILES)
        for q, s in zip(QUANTILES, qs):
            rows.append(
                {"regression": regression, "n_series": n_series,
                 "quantile": q, "stat": s}
            )
        published = coint.mackinnon_crit(regression, n_series, args.nobs)
        line = "  ".join(
            f"{int(lv * 100)}%: {np.quantile(stats, lv):+.3f} "
            f"(MacKinnon {published[lv]:+.3f})"
            for lv in (0.01, 0.05, 0.10)
        )
        print(f"  {regression:>2}, N={n_series}   {line}"
              f"   [{time.perf_counter() - t0:.1f}s]")

    table = pd.DataFrame(rows)
    path = outdir / "df_quantiles.csv"
    table.to_csv(path, index=False, float_format="%.6f")
    print()
    saved(path)


if __name__ == "__main__":
    main()
