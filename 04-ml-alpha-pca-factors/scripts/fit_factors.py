"""Stage 1 — the factor structure: how much of the cross-section is not noise.

Decomposes the return panel, compares the eigenvalue spectrum against the
Marchenko-Pastur law, identifies what the leading components are, and records
how many factors the causal rolling model retains over time.

Produces ``01_factor_structure`` and the spectrum tables. Nothing here touches
the learners; it is the reference frame everything later is judged against.

    python scripts/fit_factors.py
    python scripts/fit_factors.py --data data/sample_prices.csv --outdir /tmp/smoke
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from _common import base_parser, ensure_dirs, load_matrices, saved  # noqa: F401
from mlalpha import data as mdata  # noqa: E402
from mlalpha import pca, plotting  # noqa: E402


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--window", type=int, default=252,
                   help="trailing window for the rolling factor model")
    p.add_argument("--step", type=int, default=21,
                   help="how often the rolling factor model is refitted")
    args = p.parse_args()

    _, figures, tables = ensure_dirs(args.outdir)

    print("[1/3] loading the panel")
    panel = load_matrices(args.data)
    close = panel["close"]
    rets = mdata.to_returns(close)
    market = mdata.market_return(rets)
    n_obs, n_assets = rets.shape
    print(f"      {n_assets} tickers x {n_obs} days "
          f"({rets.index[0].date()} .. {rets.index[-1].date()})")

    print("[2/3] decomposing")
    model = pca.fit_pca(rets)
    sigma2 = model.noise_variance()
    lo, hi = pca.mp_edges(n_obs, n_assets)
    lo_a, hi_a = pca.mp_edges(n_obs, n_assets, sigma2)
    k_plain = model.n_significant()
    k_adj = model.n_significant(adjust_noise=True)

    spectrum = pd.DataFrame(
        {
            "eigenvalue": model.eigenvalues,
            "variance_share": model.eigenvalues / model.eigenvalues.sum(),
            "cumulative_share": np.cumsum(model.eigenvalues) / model.eigenvalues.sum(),
            "above_mp_edge": model.eigenvalues > hi,
            "above_adjusted_edge": model.eigenvalues > hi_a,
        },
        index=pd.Index(np.arange(1, n_assets + 1), name="component"),
    )
    spectrum.to_csv(tables / "factor_spectrum.csv")
    saved(tables / "factor_spectrum.csv")

    top = pca.fit_pca(rets, n_components=min(10, n_assets))
    scores = top.transform(rets)
    corr = pca.market_correlation(rets, top)
    summary = pd.DataFrame(
        {
            "eigenvalue": top.eigenvalues[: top.n_components],
            "variance_share": top.explained_variance_ratio,
            "corr_with_market": corr.to_numpy(),
            "ann_volatility": scores.std(ddof=1).to_numpy() * np.sqrt(252),
        },
        index=scores.columns,
    )
    summary.to_csv(tables / "factor_summary.csv")
    saved(tables / "factor_summary.csv")

    facts = pd.Series(
        {
            "n_assets": n_assets,
            "n_obs": n_obs,
            "q": n_assets / n_obs,
            "mp_upper_edge": hi,
            "mp_lower_edge": lo,
            "noise_variance_sigma2": sigma2,
            "mp_upper_edge_adjusted": hi_a,
            "n_significant": k_plain,
            "n_significant_adjusted": k_adj,
            "largest_eigenvalue": model.eigenvalues[0],
            "pc1_variance_share": model.eigenvalues[0] / model.eigenvalues.sum(),
            "pc1_corr_market": float(corr.iloc[0]),
            "top5_variance_share": float(model.eigenvalues[:5].sum() / model.eigenvalues.sum()),
        },
        name="value",
    )
    facts.to_csv(tables / "factor_facts.csv")
    saved(tables / "factor_facts.csv")

    print(f"      q = {n_assets / n_obs:.3f}, noise edge {hi:.3f} "
          f"(market-adjusted {hi_a:.3f})")
    print(f"      {k_plain} eigenvalues above the plain edge, {k_adj} above the adjusted one")
    print(f"      PC1 carries {model.eigenvalues[0] / model.eigenvalues.sum():.1%} "
          f"of variance and correlates {corr.iloc[0]:.4f} with the equal-weight market")

    print("[3/3] rolling factor count and the figure")
    _, counts = pca.rolling_residuals(rets, window=args.window, step=args.step)
    counts.rename("n_factors").to_frame().dropna().to_csv(tables / "factor_counts.csv")
    saved(tables / "factor_counts.csv")

    factor_rets = pca.factor_portfolio_returns(rets, top)
    path = plotting.figure_factor_structure(
        eigenvalues=model.eigenvalues,
        n_obs=n_obs,
        n_assets=n_assets,
        sigma2=sigma2,
        factor_returns=factor_rets,
        market=market,
        factor_count=counts,
        path=figures / "01_factor_structure.png",
        source=f"S&P 500 daily closes, {rets.index[0].date()} to {rets.index[-1].date()} · "
               f"{n_assets} names with complete histories · correlation-matrix PCA",
    )
    saved(path)


if __name__ == "__main__":
    main()
