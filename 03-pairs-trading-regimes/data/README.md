# Data card — S&P 500 daily OHLCV, 2013–2018 (pair-trading view)

This project uses the **same raw file as project 02**. Rather than duplicate 28 MB and a
duplicate audit, the provenance, schema, licence, checksum and integrity findings live in one
place:

> **[`../../02-systematic-equity-backtest/data/README.md`](../../02-systematic-equity-backtest/data/README.md)**
> — `all_stocks_5yr.csv`, Kaggle `camnugent/sandp500`, 2013-02-08 → 2018-02-07, 619,040 rows,
> 505 tickers, SHA-256 `6aea253c…e716cf`, CC0. That card documents the 20 defective rows
> (impossible OHLC, missing fields) and the 4 unadjusted corporate actions it found.

This card covers only what is **different when the same data is used for pairs**, and the
answer is: the biases are the same biases, but three of them bite considerably harder.

### Obtaining it

`scripts/_common.py` looks for the file in two places, in order:

1. `03-pairs-trading-regimes/data/all_stocks_5yr.csv`
2. `02-systematic-equity-backtest/data/all_stocks_5yr.csv`

so a checkout that already has it for project 02 needs no second copy. Either path works; the
file is git-ignored in both. A 40-ticker sample is committed here instead — see §5.

---

## 1. The universe filter is stricter, and that is not free

Project 02 kept names present on ≥98% of trading days and forward-filled the rest. This
project requires a **complete** history (`min_obs_frac=1.0`), which takes the universe from
505 names to **470**.

The reason is specific to pairs. A cross-sectional factor holds 100 names, so one patched gap
moves a portfolio weight by 1%. A pair holds two, and a gap in either leg silently changes the
data the hedge ratio was fitted on and the moments the z-score is standardised by. Both
distortions point the same way: toward a spread that looks tidier than it was.

## 2. Survivorship bias, and why it is worse here

Requiring five complete years of history *is* the condition "this company neither delisted,
merged, nor was removed from the index". Project 02 already showed this bias is visible
directly in the file — listed names rise from 476 to 505 across the sample and never fall.

For a cross-sectional factor that is a return-level bias. For a pair trade it is closer to a
bias in the *thing being traded*. A pairs strategy's characteristic loss is exactly the event
this filter deletes: one leg is acquired, halted, or repriced by a shock the other leg does
not share, and the spread never comes back. Every such event in 2013–2018 has been removed
from this universe before the screen sees it.

**Consequence: the results in this project are optimistic by an amount this dataset cannot
measure.** No correction is applied, because none is possible without delisting returns. It is
recorded here and repeated in the project README's limitations.

## 3. No sector labels

The file carries no sector, industry or fundamental classification. The screen is therefore
sector-agnostic: pairs are formed on price behaviour alone, and some selected pairs will have
no economic reason to co-move. Standard practice restricts pair formation to within-sector
candidates precisely to reduce spurious matches. That restriction cannot be applied here.

Its absence has an accounting consequence too: without a sector restriction the field of
possible pairs is the full **110,215**, which is what the multiplicity arithmetic in the
README is computed against.

## 4. The market proxy is equal-weight

The regime model is fitted to a market return series, and the dataset carries no share counts
or market caps, so the proxy is the **equal-weight cross-sectional mean** of daily returns.
Against a cap-weighted index this over-weights smaller constituents and is modestly more
volatile. It is used only to *classify days into regimes*, never to compute performance, and
the HMM's labels agree with the top tercile of EWMA volatility on 82.9% of days — so the
classification is not delicately dependent on this choice.

## 5. Derived quantities

| quantity | definition | used for |
|---|---|---|
| price matrix | `close` pivoted to date × ticker, complete histories only | everything |
| log prices | `ln(close)` | spreads, hedge ratios — a log spread with a fixed hedge ratio is a fixed *ratio* of dollar exposures |
| simple returns | `pct_change` | P&L aggregation |
| market return | equal-weight mean of simple returns | regime classification |

Prices are as-traded, not total-return. Dividends are ignored on both legs of every pair,
which is a smaller error for a spread than for a directional position but not zero: the two
legs generally have different yields, so a long-run drift is left in the spread that the
formation-window intercept absorbs and then holds fixed for 126 days.

## 6. Committed sample

`sample_prices.csv` — **40 tickers, 50,360 rows**, the full 2013-02-08 → 2018-02-07 range, so
a smoke run still gets one complete 252-day formation plus 126-day trading window.

The tickers are not random: they are the names appearing most often among the closest pairs by
the stage-1 distance metric, so the screen has something to find. That selection is made on
the full sample and is therefore look-ahead by construction. It is acceptable here and nowhere
else, because the sample exists only to prove the code runs — **no result computed from it is
ever reported.** CI writes its output to a temporary directory and then asserts the committed
reports are unchanged.

Regenerate with `python scripts/make_sample.py`.

Tickers: ADBE, AET, AON, AOS, APH, AVGO, BDX, CB, CMCSA, CMS, CTAS, DTE, EA, FIS, FISV, GD,
HD, HON, HUM, INTU, IT, ITW, MA, MAS, MHK, MMC, MMM, MSFT, NDAQ, NEE, NOC, SHW, SNPS, SYK,
TRV, TSS, TXN, UNH, V, WM.

## 7. Known limitations of this dataset for this question

| limitation | effect on the results |
|---|---|
| survivorship filter removes the pair trade's characteristic loss | results optimistic, unquantifiable |
| no delisting returns | the tail of the loss distribution is missing entirely |
| no sector labels | more spurious pairs; no within-sector restriction available |
| close-to-close only, no intraday | a 4-day half-life is measured on daily bars; execution slippage inside the day is invisible |
| no bid-ask spreads | costs are a flat bps assumption, not the security-specific reality |
| no borrow costs or shorting constraints | every short leg is assumed freely available at zero fee |
| 5 years, one market regime cycle | 220 turbulent days is the sample the headline rests on |
| as-traded prices | dividends ignored on both legs |
