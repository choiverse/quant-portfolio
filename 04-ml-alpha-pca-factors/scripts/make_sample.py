"""Build the small committed sample used by CI.

The full 28 MB panel is not in git (see ``data/README.md``), so CI exercises
the pipeline against a committed subset instead. The subset is chosen to be
*hard enough to be a real test*: 60 tickers is enough cross-section for the
quintile portfolios to hold 12 names a side, which is the smallest book the
signal construction will accept.

    python scripts/make_sample.py
"""

from __future__ import annotations

from _common import ROOT, base_parser  # noqa: F401 — sets up sys.path
from mlalpha import data as mdata  # noqa: E402


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--n-tickers", type=int, default=60)
    p.add_argument("--out", default=str(ROOT / "data" / "sample_prices.csv"))
    args = p.parse_args()

    raw = mdata.load_prices(args.data)
    counts = raw.groupby("Name")["date"].count()
    complete = counts[counts == counts.max()].index

    # Evenly spaced through the alphabetical list rather than the first N,
    # so the sample is not all one letter and, more importantly, not all one
    # part of whatever ordering the source file happens to carry.
    step = max(1, len(complete) // args.n_tickers)
    chosen = sorted(complete)[::step][: args.n_tickers]

    sample = raw[raw["Name"].isin(chosen)].reset_index(drop=True)
    sample.to_csv(args.out, index=False)
    print(f"{len(chosen)} tickers, {len(sample):,} rows -> {args.out}")


if __name__ == "__main__":
    main()
