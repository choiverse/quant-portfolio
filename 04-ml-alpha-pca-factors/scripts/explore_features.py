"""Stage 2 — the features, and whether their relationships hold up.

Scores every feature on its own against the forward return, separately in the
period the models train on and the period they are tested on, and reports how
many keep their sign. This is the reference point the models have to beat and,
as it turns out, the explanation for why they do not.

Produces ``02_features`` and the feature tables.

    python scripts/explore_features.py
    python scripts/explore_features.py --data data/sample_prices.csv --outdir /tmp/smoke
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from _common import base_parser, ensure_dirs, load_matrices, saved  # noqa: F401
from mlalpha import data as mdata  # noqa: E402
from mlalpha import diagnostics, features, plotting  # noqa: E402


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--initial-train", type=int, default=504,
                   help="training days before the first out-of-sample fold; the "
                        "split date is taken from here so this table and the "
                        "walk-forward describe the same two periods")
    args = p.parse_args()

    _, figures, tables = ensure_dirs(args.outdir)

    print("[1/3] building features")
    panel = load_matrices(args.data)
    close = panel["close"]
    feats = features.build_features(panel)
    target = features.build_target(close, horizon=args.horizon)

    design = mdata.stack_panel(feats, target)
    dates = pd.DatetimeIndex(design.index.get_level_values(0).unique())
    split = dates[min(args.initial_train, len(dates) - 1)]
    print(f"      {len(feats)} features, {len(design):,} usable rows, "
          f"{len(dates)} dates")
    print(f"      training period ends {split.date()}")

    print("[2/3] scoring each feature on its own")
    ic_split = diagnostics.ic_split_table(
        feats, target, split, family=features.FEATURE_FAMILY, dates=dates
    )
    ic_split.to_csv(tables / "feature_ic.csv")
    saved(tables / "feature_ic.csv")

    n_held = int(ic_split["sign_held"].sum())
    print(f"      {n_held} of {len(ic_split)} features keep the sign of their IC "
          f"out of sample")
    print(f"      strongest in training: {ic_split.index[0]} "
          f"(IR {ic_split['IR_train'].iloc[0]:+.3f} -> "
          f"{ic_split['IR_oos'].iloc[0]:+.3f} out of sample)")

    corr = design[list(feats)].corr(method="spearman")
    corr.to_csv(tables / "feature_correlation.csv")
    saved(tables / "feature_correlation.csv")

    off = corr.to_numpy()[~np.eye(len(corr), dtype=bool)]
    facts = pd.Series(
        {
            "n_features": len(feats),
            "n_rows": len(design),
            "n_dates": len(dates),
            "split_date": str(split.date()),
            "n_sign_held": n_held,
            "max_abs_feature_corr": float(np.abs(off).max()),
            "mean_abs_feature_corr": float(np.abs(off).mean()),
            "best_train_IR": float(ic_split["IR_train"].abs().max()),
            "best_oos_IR": float(ic_split["IR_oos"].abs().max()),
        },
        name="value",
    )
    facts.to_csv(tables / "feature_facts.csv")
    saved(tables / "feature_facts.csv")

    print("[3/3] figure")
    path = plotting.figure_features(
        ic_split=ic_split,
        corr=corr,
        path=figures / "02_features.png",
        source=f"S&P 500 daily OHLCV · {args.horizon}-day forward return, "
               f"cross-sectionally demeaned · training period ends {split.date()}",
    )
    saved(path)


if __name__ == "__main__":
    main()
