"""The walk-forward pipeline: screen, trade, repeat.

This is the piece that has to be right for anything else to mean anything. The
whole study rests on one claim — that no information from a trading window was
used to choose or calibrate the pairs traded in it — and that claim lives here,
in the slicing.

Each iteration:

    [--------- formation (252d) ---------][---- trading (126d) ----]
     screen pairs, fit alpha/beta/mu/sigma  apply them, unchanged

Windows advance by ``trading_days``, so the trading segments tile the sample
without overlapping and their returns concatenate into one series. Formation
segments do overlap, which is fine — they are inputs, not results.

``run_walk_forward`` is deliberately a pure function of the price panel. That
is what makes the look-ahead test in the suite possible: rewrite the last day
of prices, re-run, and every earlier day of P&L has to come back bit-identical.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import backtest, data, pairs


@dataclass
class WindowReport:
    """What one formation/trading cycle selected and why."""

    formation_start: pd.Timestamp
    formation_end: pd.Timestamp
    trading_start: pd.Timestamp
    trading_end: pd.Timestamp
    n_tested: int
    n_possible: int
    n_passed: int
    n_selected: int
    expected_false_positives: float
    median_half_life: float
    median_stat: float
    candidates: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)

    def as_row(self) -> dict:
        return {
            "formation_start": self.formation_start,
            "formation_end": self.formation_end,
            "trading_start": self.trading_start,
            "trading_end": self.trading_end,
            "pairs_possible": self.n_possible,
            "pairs_tested": self.n_tested,
            "pairs_passed": self.n_passed,
            "pairs_traded": self.n_selected,
            "expected_false_positives": self.expected_false_positives,
            "median_half_life": self.median_half_life,
            "median_eg_stat": self.median_stat,
        }


@dataclass
class WalkForwardResult:
    """Target weights over the whole sample plus the per-window audit trail."""

    weights: pd.DataFrame
    positions: pd.DataFrame
    windows: list[WindowReport]
    specs: dict[pd.Timestamp, list[pairs.PairSpec]]

    @property
    def window_table(self) -> pd.DataFrame:
        return pd.DataFrame([w.as_row() for w in self.windows])

    @property
    def trading_start(self) -> pd.Timestamp:
        """First day a position could be held."""
        return self.windows[0].trading_start

    @property
    def trading_end(self) -> pd.Timestamp:
        """Last day a trading window covered.

        The weight panel spans the whole sample: it is zero through the first
        formation window, and zero again after the last trading window ends
        because there is no formation window left to follow. Reporting
        statistics over the full panel would pad the return series with months
        of guaranteed zeros at both ends, flattening the volatility and
        dragging every ratio toward zero. Every performance number in this
        project is computed on ``[trading_start, trading_end]``.
        """
        return self.windows[-1].trading_end

    @property
    def selected_pairs(self) -> pd.DataFrame:
        """Every pair traded, in every window, with its formation statistics."""
        rows = []
        for start, specs in self.specs.items():
            for s in specs:
                rows.append(
                    {
                        "trading_start": start, "pair": s.name,
                        "y": s.y, "x": s.x, "beta": s.beta,
                        "eg_stat": s.stat, "pvalue": s.pvalue,
                        "half_life": s.half_life, "ssd": s.ssd,
                        "spread_sd": s.sigma,
                    }
                )
        return pd.DataFrame(rows)


def run_walk_forward(
    log_prices: pd.DataFrame,
    formation_days: int = 252,
    trading_days: int = 126,
    top_k: int = 1000,
    level: float = 0.05,
    max_pairs: int | None = 20,
    max_half_life: float | None = 60.0,
    entry: float = 2.0,
    exit: float = 0.5,
    stop: float = 4.0,
    max_lag: int = 10,
    verbose: bool = False,
) -> WalkForwardResult:
    """Screen and trade over rolling formation/trading windows."""
    windows = data.split_walk_forward(
        log_prices.index, formation_days=formation_days, trading_days=trading_days
    )
    if not windows:
        raise ValueError(
            f"{len(log_prices)} days is too short for a {formation_days}+"
            f"{trading_days} walk-forward"
        )

    weights = pd.DataFrame(0.0, index=log_prices.index, columns=log_prices.columns)
    position_frames: list[pd.DataFrame] = []
    reports: list[WindowReport] = []
    specs_by_window: dict[pd.Timestamp, list[pairs.PairSpec]] = {}

    for f_slice, t_slice in windows:
        form = log_prices.iloc[f_slice]
        trade = log_prices.iloc[t_slice]

        # A name with no formation history cannot be screened, and a name that
        # stops trading mid-window cannot be held; require both.
        usable = form.columns[form.notna().all() & trade.notna().all()]
        form, trade = form[usable], trade[usable]

        screen = pairs.screen_window(
            form,
            top_k=top_k,
            level=level,
            max_lag=max_lag,
            max_half_life=max_half_life,
            max_pairs=max_pairs,
        )

        w, pos = pairs.window_weights(
            trade, screen.pairs, entry=entry, exit=exit, stop=stop
        )
        weights.loc[w.index, w.columns] = w.to_numpy()
        position_frames.append(pos)

        t_start = trade.index[0]
        specs_by_window[t_start] = screen.pairs
        hl = [s.half_life for s in screen.pairs]
        st = [s.stat for s in screen.pairs]
        reports.append(
            WindowReport(
                formation_start=form.index[0],
                formation_end=form.index[-1],
                trading_start=t_start,
                trading_end=trade.index[-1],
                n_tested=screen.n_tested,
                n_possible=screen.n_possible,
                n_passed=screen.n_passed,
                n_selected=len(screen.pairs),
                expected_false_positives=screen.expected_false_positives,
                median_half_life=float(np.median(hl)) if hl else np.nan,
                median_stat=float(np.median(st)) if st else np.nan,
                candidates=screen.table,
            )
        )
        if verbose:
            print(
                f"  {t_start.date()} .. {trade.index[-1].date()}   "
                f"tested {screen.n_tested:>5}  passed {screen.n_passed:>4}  "
                f"traded {len(screen.pairs):>3}  "
                f"(expected false positives {screen.expected_false_positives:.0f})"
            )

    positions = (
        pd.concat(position_frames, axis=0).sort_index()
        if position_frames
        else pd.DataFrame(index=log_prices.index)
    )
    return WalkForwardResult(
        weights=weights,
        positions=positions,
        windows=reports,
        specs=specs_by_window,
    )


def run_strategy(
    prices: pd.DataFrame,
    cost_bps: float = 10.0,
    **kwargs,
) -> tuple[backtest.BacktestResult, WalkForwardResult]:
    """Prices in, backtest out. The end-to-end path, used by the tests."""
    log_prices = data.to_log_prices(prices)
    returns = data.to_returns(prices)

    wf = run_walk_forward(log_prices, **kwargs)
    engine = backtest.PairBacktester(returns, cost_bps=cost_bps)
    result = engine.run(wf.weights, name="pairs")
    return result, wf


def trim(obj, wf: WalkForwardResult):
    """Restrict a series or frame to the days a trading window covered."""
    return obj.loc[wf.trading_start:wf.trading_end]
