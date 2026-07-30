"""Stage 3 — train the models, trade them, and ask what the return actually was.

The whole experiment in one pass:

  1. purged walk-forward for three models: best single feature, ridge, boosting
  2. signal diagnostics — IC, decay, quantile profile, in-sample vs out-of-sample
  3. the regularisation sweep, which shows the failure is not model variance
  4. portfolios, with and without the staggered construction, net of costs
  5. factor attribution: how much of the P&L was exposure rather than alpha
  6. the residual experiment: models trained to predict idiosyncratic returns
  7. a robustness grid over horizon, training window and target

Produces figures 03-06, the composite headline image, and every table the
write-up quotes.

    python scripts/run_models.py
    python scripts/run_models.py --data data/sample_prices.csv --quick --outdir /tmp/smoke
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from _common import base_parser, ensure_dirs, load_matrices, saved  # noqa: F401
from mlalpha import (  # noqa: E402
    attribution,
    backtest,
    crossval,
    data as mdata,
    diagnostics,
    features,
    metrics,
    models,
    pca,
    pipeline,
    plotting,
    signals,
)

COST_GRID = [0, 2, 5, 7.5, 10, 15, 20, 30]
ALPHA_GRID = [0.01, 0.1, 1.0, 10.0, 100.0, 1_000.0, 10_000.0, 100_000.0]


def build_models(quick: bool) -> dict:
    """The three learners. ``quick`` shrinks only the boosted ensemble."""
    n_trees = 60 if quick else 200
    return {
        "best_feature": pipeline.BestFeatureModel,
        "ridge": lambda: models.RidgeRegression(alpha=100.0),
        "gbm": lambda: models.GradientBoostingRegressor(
            n_estimators=n_trees, learning_rate=0.05, max_depth=3,
            min_samples_leaf=500, subsample=0.7, seed=0,
        ),
    }


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--initial-train", type=int, default=504)
    p.add_argument("--test-size", type=int, default=63)
    p.add_argument("--embargo", type=int, default=5)
    p.add_argument("--quantile", type=float, default=0.2)
    p.add_argument("--cost-bps", type=float, default=10.0)
    p.add_argument("--quick", action="store_true",
                   help="fewer trees and skip the robustness grid")
    args = p.parse_args()

    _, figures, tables = ensure_dirs(args.outdir)
    h = args.horizon

    # ---------------------------------------------------------------- setup
    print("[1/8] building the design matrix")
    panel = load_matrices(args.data)
    close = panel["close"]
    rets = mdata.to_returns(close)
    feats = features.build_features(panel)
    target = features.build_target(close, horizon=h)
    design = mdata.stack_panel(feats, target)
    row_dates = design.index.get_level_values(0)
    feature_names = list(feats)

    folds = crossval.purged_walk_forward(
        row_dates, horizon=h, initial_train=args.initial_train,
        test_size=args.test_size, embargo=args.embargo, min_test_size=21,
    )
    fold_table = crossval.fold_table(folds)
    fold_table.to_csv(tables / "folds.csv")
    saved(tables / "folds.csv")
    print(f"      {len(design):,} rows, {len(folds)} folds, "
          f"{fold_table['n_purged'].iloc[0]:,} rows purged per fold")
    print(f"      out of sample {folds[0].test_start.date()} .. {folds[-1].test_end.date()}")

    # ------------------------------------------------------------ 2. models
    print("[2/8] walk-forward")
    specs = build_models(args.quick)
    results, ic_by_model = {}, {}
    for name, factory in specs.items():
        t0 = time.perf_counter()
        results[name] = pipeline.run_walk_forward(
            design, folds, factory, name, feature_names=feature_names
        )
        ic_by_model[name] = diagnostics.rank_ic(results[name].signal, target)
        print(f"      {name:<13} {time.perf_counter() - t0:6.1f}s  "
              f"IC {ic_by_model[name].mean():+.5f}  "
              f"IR {ic_by_model[name].mean() / ic_by_model[name].std(ddof=1):+.4f}")

    ic_table = pd.DataFrame(
        {n: diagnostics.ic_summary(ic, horizon=h) for n, ic in ic_by_model.items()}
    ).T
    ic_table["oos_r2"] = [results[n].r2(design["y"]) for n in ic_table.index]
    ic_table.to_csv(tables / "model_ic.csv")
    saved(tables / "model_ic.csv")

    # in-sample skill, for the comparison that makes the failure legible
    X = design[feature_names].to_numpy(dtype=float)
    y = design["y"].to_numpy(dtype=float)
    is_rows = []
    for name, factory in specs.items():
        m = factory()
        m.fit(X, y)
        pred = pd.Series(m.predict(X), index=design.index).unstack("ticker")
        ic_is = diagnostics.rank_ic(pred, target)
        is_rows.append(
            {
                "model": name,
                "in_sample": float(ic_is.mean()),
                "out_of_sample": float(ic_by_model[name].mean()),
                "in_sample_IR": float(ic_is.mean() / ic_is.std(ddof=1)),
                "out_of_sample_IR": float(
                    ic_by_model[name].mean() / ic_by_model[name].std(ddof=1)
                ),
            }
        )
    is_oos = pd.DataFrame(is_rows).set_index("model")
    is_oos.to_csv(tables / "in_sample_vs_oos.csv")
    saved(tables / "in_sample_vs_oos.csv")

    # ------------------------------------------------- 3. the alpha sweep
    print("[3/8] regularisation sweep")
    sweep_rows = []
    for alpha in ALPHA_GRID:
        r = pipeline.run_walk_forward(
            design, folds, lambda a=alpha: models.RidgeRegression(alpha=a),
            f"ridge_{alpha}", feature_names=feature_names,
        )
        ic = diagnostics.rank_ic(r.signal, target)
        m = models.RidgeRegression(alpha=alpha).fit(X, y)
        pred = pd.Series(m.predict(X), index=design.index).unstack("ticker")
        ic_is = diagnostics.rank_ic(pred, target)
        sweep_rows.append(
            {
                "alpha": alpha,
                "oos_ic": float(ic.mean()),
                "in_sample_ic": float(ic_is.mean()),
                "effective_dof": m.effective_dof,
            }
        )
    alpha_sweep = pd.DataFrame(sweep_rows).set_index("alpha")
    alpha_sweep.to_csv(tables / "ridge_alpha_sweep.csv")
    saved(tables / "ridge_alpha_sweep.csv")
    print(f"      OOS IC ranges {alpha_sweep['oos_ic'].min():+.5f} to "
          f"{alpha_sweep['oos_ic'].max():+.5f} across 7 orders of magnitude of alpha")

    # ------------------------------------------- 4. boosting learning curve
    print("[4/8] boosting learning curve and importances")
    gbm_imp = results["gbm"].importances.mean()
    gbm_imp.rename("importance").to_frame().sort_values(
        "importance", ascending=False
    ).to_csv(tables / "feature_importance.csv")
    saved(tables / "feature_importance.csv")

    every = max(1, (60 if args.quick else 200) // 20)
    staged_rows = []
    staged_preds: dict[int, list[pd.Series]] = {}
    for fold in folds:
        gbm = build_models(args.quick)["gbm"]()
        gbm.fit(X[fold.train_rows], y[fold.train_rows])
        for n_trees, pred in gbm.staged_predict(X[fold.test_rows], every=every):
            staged_preds.setdefault(n_trees, []).append(
                pd.Series(pred, index=design.index[fold.test_rows])
            )
    for n_trees, pieces in sorted(staged_preds.items()):
        sig = pd.concat(pieces).unstack("ticker")
        ic = diagnostics.rank_ic(sig, target)
        staged_rows.append({"n_trees": n_trees, "oos_ic": float(ic.mean())})
    staged = pd.DataFrame(staged_rows).set_index("n_trees")
    staged.to_csv(tables / "gbm_learning_curve.csv")
    saved(tables / "gbm_learning_curve.csv")

    # --------------------------------------------------- 5. the portfolios
    print("[5/8] portfolios")
    engine = backtest.Backtester(rets, cost_bps=args.cost_bps)
    runs, runs_daily = {}, {}
    for name, r in results.items():
        runs[name] = engine.run(
            signals.signal_to_book(r.signal, horizon=h, quantile=args.quantile),
            name=name,
        )
        runs_daily[name] = engine.run(
            signals.signal_to_book(r.signal, horizon=h, quantile=args.quantile,
                                   stagger=False),
            name=f"{name}_daily",
        )

    perf_rows = []
    for name, run in runs.items():
        s = run.summary()
        lo, hi = backtest.block_bootstrap_sharpe(run.returns)
        perf_rows.append(
            {
                "model": name,
                **s.to_dict(),
                "Gross Sharpe": metrics.sharpe_ratio(run.gross_returns),
                "Sharpe 90% low": lo,
                "Sharpe 90% high": hi,
                "Mean turnover": float(run.turnover.mean()),
                "Holding period (d)": run.holding_period(),
                "Break-even bps": run.breakeven_cost_bps(),
            }
        )
    perf = pd.DataFrame(perf_rows).set_index("model")
    perf.to_csv(tables / "metrics_summary.csv")
    saved(tables / "metrics_summary.csv")
    for name, row in perf.iterrows():
        print(f"      {name:<13} net Sharpe {row['Sharpe']:+.3f}  "
              f"gross {row['Gross Sharpe']:+.3f}  "
              f"turnover {row['Mean turnover']:.3f}  "
              f"hold {row['Holding period (d)']:.1f}d  "
              f"break-even {row['Break-even bps']:.1f}bps")

    turnover = pd.DataFrame(
        {
            "staggered": {n: runs[n].turnover.mean() for n in runs},
            "daily": {n: runs_daily[n].turnover.mean() for n in runs_daily},
            "staggered_sharpe": {n: metrics.sharpe_ratio(runs[n].returns) for n in runs},
            "daily_sharpe": {
                n: metrics.sharpe_ratio(runs_daily[n].returns) for n in runs_daily
            },
        }
    )
    turnover.to_csv(tables / "turnover_comparison.csv")
    saved(tables / "turnover_comparison.csv")

    cost_sweeps = {
        n: backtest.cost_sweep_from(run.gross_returns, run.turnover, COST_GRID)
        for n, run in runs.items()
    }
    pd.concat(cost_sweeps, names=["model"]).to_csv(tables / "cost_sensitivity.csv")
    saved(tables / "cost_sensitivity.csv")

    # ------------------------------------------------------ 6. attribution
    print("[6/8] factor attribution")
    train_rets = rets.loc[: folds[0].test_start].iloc[:-1]
    factor_model = pca.fit_pca(train_rets, n_components=5)
    factor_rets = pca.factor_portfolio_returns(rets, factor_model)

    decomposition = attribution.compare_models(
        {n: run.returns for n, run in runs.items()}, factor_rets
    )
    decomposition.to_csv(tables / "attribution.csv")
    saved(tables / "attribution.csv")
    print(decomposition[["Total ann. return", "Factor-explained",
                         "Residual alpha", "Alpha t-stat (NW)", "R-squared"]]
          .round(4).to_string())

    best = perf["Gross Sharpe"].idxmax()
    rolling_beta = attribution.rolling_exposure(
        runs[best].returns, factor_rets["PC1"], window=126
    )
    neutral_book = signals.neutralize(
        signals.signal_to_book(results[best].signal, horizon=h, quantile=args.quantile),
        factor_model.loadings_frame,
    )
    neutral_run = engine.run(neutral_book, name=f"{best}_neutral")
    neutral_equity = {
        f"{plotting.MODEL_LABELS[best]}": runs[best].equity,
        f"{plotting.MODEL_LABELS[best]}, factor-neutral": neutral_run.equity,
    }
    pd.DataFrame(
        {
            "raw": runs[best].summary(),
            "factor_neutral": neutral_run.summary(),
        }
    ).to_csv(tables / "neutralization.csv")
    saved(tables / "neutralization.csv")

    # ------------------------------------------- 7. the residual experiment
    print("[7/8] residual-target experiment")
    resid, _ = pca.rolling_residuals(rets, window=252, step=21)
    fwd_resid = resid.rolling(h).sum().shift(-h)
    fwd_resid = fwd_resid.sub(fwd_resid.mean(axis=1), axis=0)

    design_r = mdata.stack_panel(feats, fwd_resid)
    folds_r = crossval.purged_walk_forward(
        design_r.index.get_level_values(0), horizon=h,
        initial_train=args.initial_train, test_size=args.test_size,
        embargo=args.embargo, min_test_size=21,
    )
    residual_rows = []
    for name, factory in specs.items():
        r = pipeline.run_walk_forward(
            design_r, folds_r, factory, name, feature_names=feature_names
        )
        ic_res = diagnostics.rank_ic(r.signal, fwd_resid)
        ic_tot = diagnostics.rank_ic(r.signal, target)
        run = engine.run(
            signals.signal_to_book(r.signal, horizon=h, quantile=args.quantile),
            name=f"{name}_resid",
        )
        residual_rows.append(
            {
                "model": name,
                "vs residual return": float(ic_res.mean() / ic_res.std(ddof=1)),
                "vs total return": float(ic_tot.mean() / ic_tot.std(ddof=1)),
                "IC vs residual": float(ic_res.mean()),
                "IC vs total": float(ic_tot.mean()),
                "net Sharpe": metrics.sharpe_ratio(run.returns),
                "gross Sharpe": metrics.sharpe_ratio(run.gross_returns),
                "break-even bps": run.breakeven_cost_bps(),
            }
        )
    residual_ic = pd.DataFrame(residual_rows).set_index("model")
    residual_ic.to_csv(tables / "residual_experiment.csv")
    saved(tables / "residual_experiment.csv")
    print(residual_ic.round(4).to_string())

    # ------------------------------------------------ 8. robustness + plots
    if not args.quick:
        print("[8/8] robustness grid")
        grid_rows = []
        for horizon in (1, 5, 21):
            tgt_total = features.build_target(close, horizon=horizon)
            r_h = resid.rolling(horizon).sum().shift(-horizon)
            tgt_resid = r_h.sub(r_h.mean(axis=1), axis=0)
            for tname, tgt in (("total", tgt_total), ("residual", tgt_resid)):
                d = mdata.stack_panel(feats, tgt)
                dts = d.index.get_level_values(0)
                for wname, expanding in (("expanding", True), ("rolling", False)):
                    f = crossval.purged_walk_forward(
                        dts, horizon=horizon, initial_train=args.initial_train,
                        test_size=args.test_size, embargo=args.embargo,
                        min_test_size=21, expanding=expanding,
                    )
                    r = pipeline.run_walk_forward(
                        d, f, lambda: models.RidgeRegression(alpha=100.0),
                        "ridge", feature_names=feature_names,
                    )
                    run = engine.run(
                        signals.signal_to_book(r.signal, horizon=horizon,
                                               quantile=args.quantile),
                        name="grid",
                    )
                    ic_tot = diagnostics.rank_ic(r.signal, tgt_total)
                    grid_rows.append(
                        {
                            "horizon": horizon,
                            "target": tname,
                            "window": wname,
                            "IC vs total": float(ic_tot.mean()),
                            "IR vs total": float(ic_tot.mean() / ic_tot.std(ddof=1)),
                            "gross Sharpe": metrics.sharpe_ratio(run.gross_returns),
                            "net Sharpe": metrics.sharpe_ratio(run.returns),
                            "turnover": float(run.turnover.mean()),
                            "break-even bps": run.breakeven_cost_bps(),
                        }
                    )
        grid = pd.DataFrame(grid_rows).set_index(["horizon", "target", "window"])
        grid.to_csv(tables / "robustness_grid.csv")
        saved(tables / "robustness_grid.csv")
        n_pos = int((grid["net Sharpe"] > 0).sum())
        print(f"      {n_pos} of {len(grid)} cells have a positive net Sharpe")
    else:
        print("[8/8] robustness grid skipped (--quick)")

    print("      figures")
    decay = diagnostics.ic_decay(results[best].signal, close)
    decay.to_csv(tables / "ic_decay.csv")
    saved(tables / "ic_decay.csv")

    quantiles = diagnostics.quantile_returns(results[best].signal, target, 5)
    quantiles.mean().to_frame("mean_forward_return").to_csv(
        tables / "quantile_returns.csv"
    )
    saved(tables / "quantile_returns.csv")

    source = (f"S&P 500 daily OHLCV · {h}-day horizon · purged walk-forward, "
              f"{folds[0].test_start.date()} to {folds[-1].test_end.date()} · "
              f"{args.cost_bps:.0f} bps per unit of one-way turnover")

    saved(plotting.figure_signal_quality(
        ic_by_model=ic_by_model, decay=decay,
        quantiles=quantiles.mean().iloc[:5],
        path=figures / "03_signal_quality.png", source=source))
    saved(plotting.figure_models(
        is_oos=is_oos, alpha_sweep=alpha_sweep, importances=gbm_imp,
        staged=staged, path=figures / "04_models.png", source=source))
    saved(plotting.figure_performance(
        equity={n: r.equity for n, r in runs.items()},
        cost_sweep=cost_sweeps,
        turnover=turnover,
        drawdown={n: metrics.drawdown_series(r.returns) for n, r in runs.items()},
        path=figures / "05_performance.png", source=source))
    saved(plotting.figure_attribution(
        decomposition=decomposition, rolling_beta=rolling_beta,
        neutral_equity=neutral_equity, residual_ic=residual_ic,
        path=figures / "06_attribution.png", source=source))

    eig = pca.fit_pca(rets).eigenvalues
    ic_split = diagnostics.ic_split_table(
        feats, target, folds[0].test_start, family=features.FEATURE_FAMILY,
        dates=pd.DatetimeIndex(row_dates.unique()),
    )
    saved(plotting.figure_headline(
        eigenvalues=eig, n_obs=len(rets), n_assets=rets.shape[1],
        ic_split=ic_split,
        equity={n: r.equity for n, r in runs.items()},
        decomposition=decomposition,
        path=(figures.parent / "ml_alpha_results.png"), source=source))


if __name__ == "__main__":
    main()
