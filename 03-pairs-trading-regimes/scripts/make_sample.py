"""Build the committed sample panel used by CI.

The full 28 MB file is not in the repository, so CI exercises the pipeline
against a small slice of it. The slice is not random: pair screening needs
names that plausibly co-move, so this takes the tickers that appear most often
among the closest pairs by the stage-1 distance metric over the whole sample,
which guarantees the smoke run has something to find.

That selection is done on the full sample and is therefore look-ahead by
construction. It is fine here and nowhere else: the sample exists to prove the
code runs, and no result computed from it is ever reported.

    python scripts/make_sample.py --n-tickers 40
"""

from __future__ import annotations

import argparse
from collections import Counter

from _common import DEFAULT_DATA, ROOT, load_panel, saved
from statarb import data as sdata, pairs  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default=str(DEFAULT_DATA))
    p.add_argument("--n-tickers", type=int, default=40)
    p.add_argument("--out", default=str(ROOT / "data" / "sample_prices.csv"))
    args = p.parse_args()

    raw, prices, _ = load_panel(args.data)
    log_prices = sdata.to_log_prices(prices)

    close = pairs.distance_screen(log_prices, top_k=500)
    counts = Counter(list(close["a"]) + list(close["b"]))
    keep = [t for t, _ in counts.most_common(args.n_tickers)]
    keep = sorted(keep)

    subset = raw[raw["Name"].isin(keep)].copy()
    subset.to_csv(args.out, index=False)

    print(f"{len(keep)} tickers, {len(subset):,} rows "
          f"({subset['date'].min().date()} .. {subset['date'].max().date()})")
    print("  " + ", ".join(keep))
    saved(type(ROOT)(args.out))


if __name__ == "__main__":
    main()
