# Method

Derivations and implementation notes for the estimators in `src/statarb/`. Everything here is
built from the regressions and likelihoods up — there is no `statsmodels` dependency — which
means the awkward parts (critical values, identification, initialisation) have to be dealt
with rather than imported. Each section ends with the check in `scripts/validate.py` that
holds it honest.

---

## 1. The Augmented Dickey-Fuller regression

A series `y_t` has a unit root if `y_t = y_{t-1} + e_t`. Write the general AR(1) as
`y_t = phi * y_{t-1} + e_t` and subtract `y_{t-1}`:

```
dy_t = gamma * y_{t-1} + e_t ,        gamma = phi - 1
```

so the unit root is `gamma = 0` and stationarity is `gamma < 0`. Serial correlation in `e_t`
biases the estimate, so the regression is *augmented* with lagged differences:

```
dy_t = gamma*y_{t-1} + sum_{i=1..p} delta_i * dy_{t-i} + (deterministic) + e_t
```

The test statistic is the t-statistic on `gamma`. The catch is that under the null it does
**not** follow a t-distribution: `y_{t-1}` is non-stationary, the usual central limit argument
fails, and the statistic converges to a functional of Brownian motion whose quantiles are far
into the left tail — around −2.86 at 5% with a constant, against −1.65 for a normal. Using
normal critical values would reject roughly four times too often.

**Lag selection.** `p` is chosen by AIC over `0..max_lag`, with `max_lag` from Schwert's rule
`ceil(12*(T/100)^0.25)`. All candidates are fitted on the *same* sample — the rows available
at `max_lag` — because AIC compares likelihoods, and a model fitted on more observations has a
mechanically different one.

**Deterministic terms** are removed by Frisch-Waugh rather than stacked into the design:
projecting a constant (and trend) out of both `y_{t-1}` and `dy_t` leaves the t-statistic on
`gamma` unchanged and lets the whole Monte Carlo run as one matrix operation over 200,000
replications at once.

### Critical values, twice

Two independent sources are kept, and `validate.py` compares them:

1. **MacKinnon (2010) response surfaces**, hard-coded in `cointegration.MACKINNON`. Finite
   sample critical values are `tau(T) = tau_inf + b1/T + b2/T^2 + b3/T^3`. These drive every
   accept/reject decision in the project.
2. **This project's own simulation**, `scripts/tabulate_df.py`, which generates random walks,
   computes the statistic, and tabulates its quantiles into
   `src/statarb/tables/df_quantiles.csv`. This is what turns a statistic into a p-value.

With 200,000 replications of length 2,000 the two agree to **0.004 or better** at every
tabulated point:

| case | 1% | 5% | 10% |
|---|---|---|---|
| constant, N=1 | −3.437 (MacKinnon −3.434) | −2.863 (−2.863) | −2.570 (−2.568) |
| constant+trend, N=1 | −3.962 (−3.963) | −3.410 (−3.413) | −3.128 (−3.128) |
| constant, N=2 (residual) | −3.902 (−3.902) | −3.335 (−3.339) | −3.044 (−3.047) |

That agreement is the evidence that the regression is implemented correctly. A sign error, a
misaligned lag, or mishandled deterministic terms would move the simulated table off the
published one, and gate 1 would fail.

> **Gates 1–3.** Simulated quantiles match MacKinnon to <0.05; the test rejects true random
> walks at 5% (0.048 observed over 1,000 walks, 0.29 s.e. from nominal); it rejects a
> stationary AR(1) with `phi=0.9` on 1,000 of 1,000 draws.

---

## 2. Why a residual needs its own distribution

The Engle-Granger procedure is two steps:

1. Estimate the cointegrating regression `y_t = alpha + beta*x_t + s_t` by OLS.
2. Test `s_t` — the residual — for a unit root.

Step 2 cannot use the ordinary ADF critical values. OLS chose `beta` to *minimise* the
residual variance, so the residual is more stationary-looking than an observed series would
be, and the null distribution shifts left. MacKinnon indexes this by `N`, the number of series
in the cointegrating regression: `N=1` for observed data, `N=2` for a two-variable pair. At
T=500 the 5% points are −2.87 and −3.35 respectively — using the wrong one would roughly
double the false-positive rate.

Implementation detail that follows from this: the residual is tested with **no constant of its
own** (it is mean zero by construction) but scored against the **`N=2`, constant** table,
because the cointegrating regression that produced it did include a constant. That is the only
place in the codebase where the regression and the critical-value lookup differ, and it is why
`adf_test` takes a separate `crit_regression` argument.

**Both orderings are tried.** Regressing `y` on `x` and `x` on `y` give different residuals
and different statistics; the more negative is kept. This is one extra test per pair and a
genuine source of selection bias, which the multiplicity accounting in the README reflects.

> **Gate 4.** On a synthetic pair built as `y = 0.5 + 1.4x + s` with `x` a random walk and `s`
> an AR(1), the recovered hedge ratio is 1.408 against a true 1.400.

---

## 3. The spread's half-life

Fit `ds_t = a + b*s_{t-1} + e_t`. With `phi = 1 + b`, a shock decays as `phi^t`, so it halves
at

```
t_half = ln(2) / -ln(phi)
```

Most implementations use `ln(2) / -b`, the first-order expansion. It is fine for slow
reversion and visibly wrong for fast: at `phi = 0.9` it gives 6.93 days against the true 6.58,
a 5% overstatement that grows as `phi` falls. Since this project uses the half-life as a
*filter threshold*, and since the pairs it selects revert in days rather than months, the
exact form is used.

A subtlety worth stating: on a finite sample a true random walk produces a *negative* `b` more
often than not — the same Dickey-Fuller downward bias the critical values above exist to
correct. So a random walk returns a large finite half-life, not an infinite one, and the
filter rather than a test for `inf` is what excludes it.

> **Gate 5.** An AR(1) with `phi = 0.95` has a true half-life of 13.51 days; the estimator
> returns 13.79.

---

## 4. GARCH(1,1) by maximum likelihood

```
r_t = mu + eps_t ,   eps_t = sigma_t * z_t ,   z_t ~ N(0,1)
sigma2_t = omega + alpha * eps2_{t-1} + beta * sigma2_{t-1}
```

Gaussian log-likelihood, with `sigma2_0` seeded at the sample variance:

```
logL = -0.5 * sum_t [ ln(2*pi) + ln(sigma2_t) + eps2_t / sigma2_t ]
```

Two implementation points that matter more than the algebra:

**Rescaling.** Daily returns in decimals put `omega` around `1e-6` while `alpha` and `beta` are
around `0.1` — six orders of magnitude apart, which makes the numerical gradient useless.
Returns are multiplied by 100 before optimising and the parameters scaled back afterwards. In
percent units every parameter is order `0.01` to `1`.

**Constraints.** `omega > 0`, `alpha, beta >= 0` and `alpha + beta < 1` are imposed via SLSQP.
The last is what makes the variance process stationary; without it the optimiser will happily
find an explosive fit with a higher in-sample likelihood. Three starting points are used
because the likelihood is nearly flat along the `alpha`/`beta` ridge.

### The identification limit, stated rather than hidden

Repeated fits on simulated paths with known parameters (`omega=0.02, alpha=0.08, beta=0.90` in
percent units):

| n | alpha | beta | alpha+beta |
|---|---|---|---|
| 2,000 | 0.075 ± 0.010 | 0.906 ± 0.013 | 0.981 ± 0.006 |
| 4,000 | 0.086 ± 0.004 | 0.893 ± 0.005 | 0.979 ± 0.004 |
| 10,000 | 0.083 ± 0.005 | 0.897 ± 0.005 | 0.980 ± 0.003 |
| 40,000 | 0.082 ± 0.002 | 0.898 ± 0.003 | 0.979 ± 0.002 |

The persistence `alpha + beta` is pinned down at every sample size. The split between `alpha`
and `beta` is not, and tightens only slowly, because the likelihood is nearly flat along the
ridge that trades one against the other. This is why gate 6 tests persistence: a gate on
`alpha` alone would be testing luck. It is also the quantity that matters — the variance
half-life `ln(0.5)/ln(alpha+beta)` depends on the sum, not the split.

On the S&P 500 equal-weight market return the fit gives `alpha = 0.179`, `beta = 0.743`,
persistence 0.922, a variance half-life of 8.6 days and a long-run annualised volatility of
13.0%.

> **Gate 6.** Persistence recovered to within 0.005 of the true 0.98 on a simulated path.

---

## 5. The hidden Markov model

Two latent states, Gaussian emissions with state-specific mean and variance, Markov
transitions. Fitted by Baum-Welch (EM).

**Scaled forward-backward.** The unscaled forward recursion multiplies a density per time
step, and the joint density of a thousand observations underflows to zero within a few hundred
steps — every posterior becomes 0/0. Rescaling `alpha_t` to sum to 1 at each step fixes it,
and the scale factors are not discarded: they are the one-step predictive likelihoods, so
`logL = sum_t ln(c_t)`.

**E-step.**
```
gamma_t(i)   = P(state_t = i | x)                    normalised alpha_t * beta_t
xi_t(i,j)    = P(state_t = i, state_{t+1} = j | x)   normalised over (i,j)
```

**M-step.**
```
pi_i     = gamma_1(i)
A_ij     = sum_t xi_t(i,j) / sum_t gamma_t(i)
mu_j     = sum_t gamma_t(j) x_t / sum_t gamma_t(j)
var_j    = sum_t gamma_t(j) (x_t - mu_j)^2 / sum_t gamma_t(j)
```

**Initialisation.** Random starts are standard and are a poor fit here: the likelihood has a
degenerate corner where one state collapses onto a single point with zero variance, and a
random start finds it often enough to matter. EM is instead seeded by splitting the sample on
`|x - mean(x)|`, which starts it near the calm/turbulent structure it is meant to find, with a
persistent (0.95 diagonal) transition matrix.

**Label switching.** Relabelling the states leaves the likelihood unchanged, so without a
convention "state 1" would mean something different on every run. States are always returned
sorted by emission variance: state 0 is calm.

**Underflow guard.** An observation far enough into the tails of *every* state has zero
emission density in all of them and the posterior row sums to zero. Such a point carries no
information about which state produced it, so the filtered estimate is kept rather than
dividing by zero.

### Filtered vs smoothed, and why it is not a stylistic choice

- `filter` returns `P(state_t | x_1..x_t)` — past only.
- `smooth` returns `P(state_t | x_1..x_T)` — the whole sample, including the future.

Smoothed labels are sharper and make better charts. They are also unusable for anything
touching a trading decision or a performance attribution. The rule in this project: **smoothed
for description, filtered for attribution.**

Even a filtered path is not fully causal if the *parameters* came from the whole sample, so
`fit_causal` estimates on a 504-day burn-in and filters forward with those parameters frozen.
Days inside the burn-in are returned as `NaN` rather than back-filled — they have no
out-of-sample label, and every attribution function drops them.

On the market series the fitted states are:

| state | ann. vol | stay prob | expected run | share of days |
|---|---|---|---|---|
| calm | 8.5% | 0.971 | 34.0 days | 77.2% |
| turbulent | 19.8% | 0.924 | 13.2 days | 22.8% |

An independent cross-check: the HMM's turbulent label agrees with the top tercile of EWMA
volatility on **82.9%** of days. The two methods share no machinery, so the agreement is
evidence that "turbulent" is picking up something real rather than a quirk of EM.

> **Gate 7.** On a simulated two-regime path the Viterbi decoding recovers the hidden states
> with 97.8% accuracy, and the fitted annualised volatilities (0.095 / 0.315) match the true
> ones (0.095 / 0.317).

---

## 6. The walk-forward protocol

```
[--------- formation, 252 days ---------][---- trading, 126 days ----]
 select pairs, fit alpha/beta/mu/sigma     apply them, unchanged
```

Windows advance by 126 days, so trading segments tile the sample without overlapping and their
returns concatenate into one series. Formation segments overlap, which is fine — they are
inputs, not results.

**Frozen at the end of formation, and never re-estimated during trading:**

| quantity | why it must be frozen |
|---|---|
| which pairs to trade | selecting on the trading window is the classic hindsight bias |
| `alpha`, `beta` | a hedge ratio refitted on the traded data fits the P&L, not the relationship |
| `mu`, `sigma` of the spread | standardising a window by its own mean guarantees it looks mean-reverting |

The z-score point is the easiest to get wrong and the most damaging. `z_t = (s_t - mu) / sigma`
with `mu` and `sigma` from the *trading* window centres the spread on its own realised mean,
which manufactures reversion out of any series at all.

**Execution.** Weights decided from the close of day `t` earn the return of day `t+1`.
Everything downstream — turnover, costs, exposure — is defined on the executed book
`weights.shift(1)`, so a cost can never be charged on a day the position was not yet held.

**Reporting span.** The weight panel spans the whole sample but is flat by construction before
the first trading window and after the last. Every performance number is computed on
`[trading_start, trading_end]` = 2014-02-10 to 2017-08-09, 882 trading days. Padding with the
flat ends would compress the volatility and drag every ratio toward zero.

> **Gate 8.** The entire pipeline is re-run on a panel whose final row has been multiplied by
> 1.5, and every earlier day of P&L must come back bit-identical. Selection, hedge ratios,
> spread moments and z-scores are all estimated from data, so one misplaced `shift` anywhere
> in that chain shows up here. Observed difference over 798 earlier days: exactly 0.

---

## 7. Position sizing

The spread is in logs, so `d(spread) ≈ r_y - beta * r_x`: one dollar of `y` against `beta`
dollars of `x`. Dividing by `1 + |beta|` normalises every pair to gross exposure 1, so a pair
with a hedge ratio of 3 does not silently receive four times the capital of one with a hedge
ratio of 1.

Capital is split equally across the pairs selected in a window. Idle capital stays idle rather
than levering the active pairs to a constant gross — otherwise a window in which only one pair
triggers would quietly become a full-size single-pair bet. The consequence is a low average
gross exposure (0.23 of a 1.0 budget), which is the honest depiction of a strategy that is
flat most of the time.

**Entry, exit, stop** at `|z| > 2`, `|z| < 0.5`, `|z| > 4`. A pair that hits the stop is
abandoned for the rest of the window and not re-entered even if the z-score returns. A spread
four formation standard deviations away is evidence that the relationship estimated during
formation has broken — a merger, a guidance cut, a sector rotation — and averaging into that
is the trade that has historically ended pairs desks.

---

## 8. Uncertainty

Every regime Sharpe is reported with two measures of uncertainty, because the headline of the
study is a difference between subsamples and one of them has 220 days in it.

**Asymptotic standard error.** `SE(SR) = sqrt((1 + SR^2/2)/n)`, scaled to annual. It assumes
i.i.d. returns, which these are not, so it is the optimistic bound.

**Moving-block bootstrap.** Resampling individual days would destroy the serial dependence
that a pair book generates — positions are held for days at a time. Contiguous blocks of 21
days are resampled instead, preserving dependence inside each block and randomising only the
order of blocks. 2,000 resamples, 90% percentile interval.

Both are reported in `reports/tables/regime_attribution.csv` and drawn as error bars on the
headline figure. Where they disagree, believe the bootstrap.
