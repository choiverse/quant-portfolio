# Data card — S&P 500 daily OHLCV, 2013–2018 (supervised-learning view)

This project uses the **same raw file as projects 02 and 03**. Rather than a third copy of
28 MB and a third copy of the audit, the provenance, schema, licence, checksum and integrity
findings live in one place:

> **[`../../02-systematic-equity-backtest/data/README.md`](../../02-systematic-equity-backtest/data/README.md)**
> — `all_stocks_5yr.csv`, Kaggle `camnugent/sandp500`, 2013-02-08 → 2018-02-07, 619,040 rows,
> 505 tickers, SHA-256 `6aea253c…e716cf`, CC0. That card documents the 20 defective rows
> (impossible OHLC, missing fields) and the 4 unadjusted corporate actions it found.

This card covers only what is **different when the same data is used to train a model**. Three
things are: the universe filter is stricter again and for a new reason, two OHLCV fields that
the earlier projects never touched are now load-bearing, and the sample is short enough that
the train/test split is itself a limitation rather than a detail.

### Obtaining it

`scripts/_common.py` looks for the file in two places, in order:

1. `04-ml-alpha-pca-factors/data/all_stocks_5yr.csv`
2. `02-systematic-equity-backtest/data/all_stocks_5yr.csv`

so a checkout that already has it for project 02 or 03 needs no further copy. Either path
works; the file is git-ignored in all three. A 60-ticker sample is committed here instead —
see §6.

---

## 1. The universe: 505 → 468 names, and the two that hurt

| filter | names kept |
|---|---|
| all tickers in the file | 505 |
| complete `close` history (projects 03 and 04) | 470 |
| **complete `high` and `low` history as well (this project)** | **468** |

The step from 470 to 468 is worth naming because it is entirely my own filter's doing.
**REGN** and **VRTX** are dropped for exactly **one missing `high`/`low` bar each** — REGN on
2015-06-09, VRTX on 2015-05-12. Two large-cap biotechnology names leave the universe over a
single absent field in five years of data.

That is a bad trade and it is recorded here rather than smoothed over. The defensible fix is
to interpolate a single isolated missing bar and keep the name; the reason it is not done is
that any imputation rule is a modelling choice that would need its own validation, and adding
one for two names would not change a single conclusion in this project. The cost is two names
out of 470, or 0.4% of the cross-section.

The reason the filter is strict at all is different from project 03's. There, a gap in either
leg silently changed what a hedge ratio was fitted on. Here the problem is that a learner
trained on a panel with ragged coverage learns the **coverage pattern** along with the signal:
a name that appears halfway through the sample carries a systematically different feature
distribution, because short histories mean fresher listings and different volatility. Dropping
the incomplete names costs breadth and buys a design matrix whose rows are exchangeable.

## 2. Survivorship bias — unchanged in kind, and now inside the training label

Project 02 showed this bias is visible directly in the file: listed names rise from 476 to 505
across the sample and never fall. Requiring five complete years *is* the condition "this
company neither delisted, merged, nor was removed from the index".

For this project the consequence has a specific shape. The label is a forward return, so a
company that was acquired or delisted has no label at all — it is not that its return is
mismeasured, it is that the model is never shown the case. Every relationship the model
learns is conditional on survival, and every relationship it is tested on is too. **A negative
result is therefore conservative in the right direction** — the strategy failed on the easy
version of the problem — but a positive one would have been optimistic by an amount this
dataset cannot measure. No correction is applied, because none is possible without delisting
returns.

## 3. Two new fields, and four rows where one of them is zero

Projects 02 and 03 used `close` and nothing else. This project also uses:

- **`volume`**, for `dollar_volume`, `volume_shock` and `amihud`
- **`high` / `low`**, for `hl_range`

`volume` is exactly **0** on four rows: DHR and O on 2016-01-12, UA on 2016-04-07, FTV on
2016-07-01. A zero-volume day for an S&P 500 name is not a real trading halt in three of the
four cases; it is a data defect of the same family as the ones project 02's audit catalogued.
The features that divide by dollar volume take a logarithm, so these rows become `NaN` and are
dropped by `stack_panel` rather than becoming infinities. That is four rows out of 468,936.

## 4. No fundamentals, no sector, no share count

The file carries prices and volume. It carries no earnings, no book value, no sector code and
no shares outstanding. Three consequences, each of which limits what the feature set can be:

- **There is no value factor and no quality factor.** The 14 features are entirely technical.
  A study of whether machine learning adds anything over a linear model is a weaker study when
  the inputs are all price-derived, because the interactions ML is good at finding are more
  plausible between fundamentals and prices than among price transforms alone.
- **There is no size factor.** `dollar_volume` stands in for it and is not the same thing: it
  correlates with market cap but also with turnover, so it carries a liquidity component a
  genuine size factor would not.
- **No sector neutralisation is possible.** The PCA factors partly recover sector structure —
  that is what components 2 through ~14 largely are — which is one reason the factor
  attribution in §5 of the write-up is done against statistical factors rather than sector
  dummies. It is a substitute, not an equivalent.

## 5. The sample is short, and the split makes it shorter

| | |
|---|---|
| Raw panel | 1,259 trading days |
| Lost to the 252-day feature burn-in | 252 |
| Lost to the 5-day forward label | 5 |
| **Usable design matrix** | **1,002 dates × 468 names = 468,936 rows** |
| Reserved for the first training window | 504 (two years) |
| **Out-of-sample evaluation** | **498 days ≈ 1.98 years, 2016-02-10 → 2018-01-31** |

Two years of out-of-sample data is thin for a claim about a machine-learning strategy, and it
covers one market environment: a recovery from the early-2016 selloff followed by the
low-volatility grind of 2017. It contains no 2008 and no 2020. The write-up's limitations
section repeats this; it is the binding constraint on every number in the project.

The burn-in is not negotiable given the feature set — 12-1 momentum needs 252 days by
definition — so the only way to buy more out-of-sample data from this file would be to shorten
the training window, which trades one kind of unreliability for another.

## 6. The committed sample

`data/sample_prices.csv` — 60 tickers, 75,540 rows, generated by
`python scripts/make_sample.py`. Names are taken at even spacing through the alphabetically
sorted list of complete-history tickers, so the sample is not all one letter and not all one
region of whatever ordering the source file happens to carry.

Sixty is not arbitrary. The strategy holds the top and bottom quintile, so 60 names is 12 a
side — the smallest book for which the portfolio construction is exercising the same code
path it does on the full panel, and above the 20-name floor below which `long_short_weights`
returns a flat book. CI runs the whole pipeline against it with `--outdir` pointed at a
scratch directory, so a smoke run can never overwrite the committed results, which were
generated from the full 619,040-row file.

**Numbers produced from the sample are not comparable with the ones in the write-up** and are
not meant to be — 60 names is a different cross-section, and the write-up's own §2 is about
how unstable cross-sectional relationships are.

## 7. Licence

CC0 (public domain) as published on Kaggle. See project 02's card for the full statement.
