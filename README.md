# Quant Research & Development Portfolio

![python](https://img.shields.io/badge/python-3.11%2B-blue)
![tests](https://img.shields.io/badge/tests-25%20passing-brightgreen)
![figures](https://img.shields.io/badge/figures-6%20generated-informational)
![licence](https://img.shields.io/badge/licence-MIT-green)

Self-contained quantitative-finance projects, each built end-to-end from raw
data or first principles and written up for **quant research / quant developer**
work.

Three rules every project follows:

1. **Reproducible from code.** One command regenerates every figure and table in
   the write-up. Seeds are fixed and recorded; no result is hand-copied.
2. **Validated against something known.** Where a result can be checked against a
   closed form, an exact moment, or a theoretical scaling law, there is a script
   that does exactly that and exits non-zero if it fails.
3. **Written up honestly.** Negative results are reported as results, and known
   limitations get their own section rather than a footnote.

---

## Projects

### [01 · Rough Volatility Monte Carlo](01-rough-volatility-monte-carlo)

A from-scratch Monte Carlo engine for the rough Bergomi model, built to test
whether it reproduces the empirical signature that launched the rough-volatility
literature: an at-the-money implied-vol skew decaying as `T^(H−1/2)`.

[![rough volatility results](01-rough-volatility-monte-carlo/reports/rough_vol_results.png)](01-rough-volatility-monte-carlo)

| | |
|---|---|
| **Result** | fitted skew exponent **−0.407** vs theoretical **−0.400** |
| **Independent check** | Hurst exponent recovered from the paths: **0.099** (specified 0.100) |
| **Validation** | 6/6 correctness gates · 25 unit tests |
| **Performance** | 200k paths × 100 steps in **1.6 s** (FFT-vectorized hybrid scheme) |
| **Notable finding** | the antithetic standard error was being computed wrong, making the technique look useless — and once fixed, stacking it with a control variate is *worse* than the control variate alone |

`Monte Carlo` · `fractional processes` · `variance reduction` · `implied volatility`
· `FFT optimization` · `OOP model design`

---

## Roadmap

| # | Project | Core skills | Status |
|---|---|---|---|
| 01 | [Rough Volatility Monte Carlo](01-rough-volatility-monte-carlo) | Monte Carlo, fractional processes, variance reduction, FFT | ✅ Done |
| — | Systematic Cross-Sectional Equity Backtest | panel data, backtesting, factor signals, costs, risk metrics | ⏳ Planned |
| — | Volatility & Pairs-Trading / Regime Study | time-series modeling, cointegration, EWMA volatility | ⏳ Planned |
| — | ML Alpha Signals & PCA Factor Model | feature engineering, PCA, walk-forward validation | ⏳ Planned |
| — | Portfolio Optimization & Efficient Frontier | mean-variance optimization, covariance estimation, risk parity | ⏳ Planned |
| — | Quant Research Tearsheet & Visualization | perceptually-sound financial charts, performance attribution | ⏳ Planned |
| — | Market Data Warehouse & SQL Analytics | schema design, ER modeling, SQL analytics, Python ETL | ⏳ Planned |

## Why these projects

Quant research and quant dev desks look for a specific stack. Taken together
these projects are chosen to cover it:

- **Simulation & numerics** — Monte Carlo, stochastic processes, variance
  reduction, convergence analysis, vectorized performance work
- **Data engineering** — cleaning and shaping messy market panels, and auditing
  them before trusting them
- **Statistics & ML** — signal construction, rank correlation, dimensionality
  reduction, validation without leakage
- **Financial reasoning** — derivative pricing, risk-adjusted performance,
  transaction costs, survivorship and look-ahead bias
- **Software craft** — reusable, documented, *tested* Python packages, not
  throwaway notebooks

## Repository conventions

Each `NN-*` folder is independent and follows the same structure:

```
NN-project-name/
  README.md          the write-up: question, figures, results, limitations
  requirements.txt   pinned minimum versions
  data/README.md     the data card — provenance, schema, biases, licence
  docs/              derivations and methodology, where there is maths
  src/<package>/     the reusable library
  scripts/           runnable entry points, numbered by pipeline stage
  tests/             pytest suite
  reports/           generated figures/ and tables/ — never hand-edited
```

- **Data cards are mandatory**, including for projects with no external dataset —
  Project 01's card documents its simulated data-generating process, every seed,
  and its known limitations.
- **Large raw datasets are not committed** (see each project's `data/README.md`
  for provenance and a SHA-256 checksum); a small sample is included so the code
  can be smoke-tested.
- **Figures share one theme and one colorblind-validated palette** across the
  whole portfolio, checked for CVD separation and contrast against the chart
  surface.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

# Project 01 — no data download needed
cd 01-rough-volatility-monte-carlo
pip install -r requirements.txt
python scripts/validate.py         # 6/6 gates
python scripts/run_experiment.py   # the headline figure
```

Run every test in the portfolio:

```bash
pytest -q          # 25 tests
```

## Licence

MIT — see [LICENSE](LICENSE).

---
*Author: Jae Woong Choi*
