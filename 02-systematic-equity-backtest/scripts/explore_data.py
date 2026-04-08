"""Stage 1 — profile the dataset before any strategy logic runs.

Produces the evidence behind the data card in ``data/README.md``:

    reports/figures/01_data_overview.png
    reports/figures/02_return_distribution.png
    reports/tables/data_profile.csv
    reports/tables/data_quality_checks.csv
    reports/tables/return_distribution_stats.csv

Usage
-----
    python scripts/explore_data.py --data data/all_stocks_5yr.csv
"""

from __future__ import annotations

from _common import base_parser, ensure_dirs, load_panel, saved

from quantbt import eda
from quantbt.plotting import data_overview_figure, return_distribution_figure


def main() -> None:
    args = base_parser("Dataset profiling for the S&P 500 equity panel.").parse_args()
    _, FIGURES, TABLES = ensure_dirs(args.outdir)

    print(f"[1/3] Loading {args.data} ...")
    raw, prices, rets = load_panel(args.data)

    print("[2/3] Profiling the panel ...")
    profile = eda.panel_summary(raw, prices)
    quality = eda.quality_flags(raw)
    dist = eda.return_distribution_stats(rets)

    violations = eda.integrity_violations(raw)
    extremes = eda.extreme_moves(raw, threshold=0.5)

    profile.to_csv(TABLES / "data_profile.csv", header=True)
    quality.to_csv(TABLES / "data_quality_checks.csv")
    dist.to_csv(TABLES / "return_distribution_stats.csv", header=True)
    violations.to_csv(TABLES / "data_integrity_violations.csv", index=False)
    extremes.to_csv(TABLES / "extreme_moves.csv", index=False)

    print("[3/3] Drawing figures ...")
    data_overview_figure(raw, prices, savepath=str(FIGURES / "01_data_overview.png"))
    return_distribution_figure(rets, savepath=str(FIGURES / "02_return_distribution.png"))

    print("\n=== Dataset profile ===")
    for k, v in profile.items():
        print(f"  {k:38s} {v}")

    print("\n=== Integrity checks ===")
    for name, row in quality.iterrows():
        print(f"  {name:44s} {int(row['count']):>8,}  {row['verdict']}")

    print("\n=== Rows failing a structural check ===")
    print(violations.to_string(index=False))

    print("\n=== Single-day moves beyond ±50% (corporate-action suspects) ===")
    print(extremes.to_string(index=False))

    print("\n=== Return distribution ===")
    for k, v in dist.items():
        print(f"  {k:22s} {v:,.4f}" if abs(v) < 1e4 else f"  {k:22s} {v:,.0f}")

    for p in (FIGURES / "01_data_overview.png", FIGURES / "02_return_distribution.png",
              TABLES / "data_profile.csv", TABLES / "data_quality_checks.csv",
              TABLES / "return_distribution_stats.csv",
              TABLES / "data_integrity_violations.csv", TABLES / "extreme_moves.csv"):
        saved(p)


if __name__ == "__main__":
    main()
