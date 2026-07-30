# Method

Derivations and implementation notes for `src/mlalpha/`. Everything here is built from the
linear algebra and the loss functions up — there is no `scikit-learn` dependency — which means
the awkward parts (the shrinkage filter, the split search, the noise edge, the purging rule)
have to be dealt with rather than imported. Each section ends with the check in
`scripts/validate.py` that holds it honest.

---

## 1. The problem, stated precisely

The panel has `T` dates and `N` names. For each `(t, i)` there are `p = 14` features known at
the close of day `t`, and a label

```
y_{t,i} = r_{t,i -> t+h}  -  (1/N) * sum_j r_{t,j -> t+h}
```

the `h`-day forward return with the cross-sectional mean removed. Stacked, that is a design
matrix of 468,936 rows.

**Why the label is demeaned.** Left in levels, the most predictable component of every stock's
forward return is the market's, which is common to all of them. A model would spend its
capacity forecasting the index, score respectably on `R^2`, and rank the cross-section no
better than chance. The strategy built on top is dollar-neutral, so the market component is
precisely the part it cannot trade. Demeaning makes the regression target the same object the
portfolio bets on.

**Why the features are ranked within the date.** Every feature is mapped to its
cross-sectional rank on `[-0.5, 0.5]`. Two reasons. Financial features have heavy tails, and
one name with a 300% volume spike moves a cross-sectional z-score for every other name in the
book. And standardising across *time* instead would feed the model the market's volatility
regime through the back door: in a calm month every name's 21-day volatility is low, so a
model trained on raw levels learns "low vol → predict the calm-market average", which is a
statement about the calendar, not about the stock.

**Why `R^2` is the wrong metric.** A 5-day cross-sectional return is overwhelmingly noise. A
model with genuine ranking skill still posts an out-of-sample `R^2` near zero, and usually
below it — the small amount of signal does not pay for the variance of its own predictions.
The strategy needs something much weaker than magnitude accuracy: it needs the *ordering* to
beat chance, because it buys the top quintile and sells the bottom. That is the information
coefficient, the within-date rank correlation between prediction and outcome. `R^2` is still
reported, in `reports/tables/model_ic.csv`, precisely so the gap between the two is visible.

---

## 2. Principal components by SVD

Let `Z` be the panel centred and scaled to unit column variance. The sample correlation matrix
is `C = Z'Z / (T-1)`. Rather than form `C` and decompose it, take the SVD of the scaled panel:

```
Z / sqrt(T-1) = U S V'        =>       C = V S^2 V'
```

so the eigenvalues of `C` are `S^2` and the loadings are the columns of `V`, obtained without
ever forming `Z'Z`. That matters here: forming the Gram matrix squares the condition number,
and with 468 columns of daily data the loss of precision lands on the *small* eigenvalues —
exactly the ones being compared against a theoretical edge.

**Correlation, not covariance.** On the covariance matrix a handful of high-volatility names
dominate the leading components purely because they are volatile, and the "factors" that
emerge are largely a list of the noisiest stocks. Standardising also makes §3 exact, because
the Marchenko-Pastur law is stated for unit-variance data.

**Sign convention.** The SVD determines each loading vector only up to sign, so an unpinned
PC1 can flip between two fits of nearly identical data and turn a stable factor exposure into
a series that changes sign at a window boundary. The convention: the loadings sum to a
non-negative number, which points PC1 the same way as the market. For components whose
loadings sum to ~0 by construction — every component after the first, being orthogonal to
something market-like — the sign of the largest-magnitude loading is fixed instead.

> **Gate 1** builds a matrix whose sample covariance is *exactly* a chosen `C`: start from `Z`
> with orthonormal columns, so its sample covariance is exactly `I`, then form `Z C^(1/2)`.
> The recovered eigenvalues must match the ones written down to 1e-10. No Monte Carlo
> convergence is involved, so a real error cannot hide inside sampling noise.

---

## 3. How many factors are real: the Marchenko-Pastur law

Take `N` independent series of length `T`, with `q = N/T` fixed as both grow. The eigenvalues
of their sample correlation matrix do not concentrate at 1 — they spread across

```
[ (1 - sqrt(q))^2 ,  (1 + sqrt(q))^2 ]
```

with the density

```
rho(x) = sqrt((h - x)(x - l)) / (2 pi q x)  ,     l, h the edges above
```

and nothing outside it. So an eigenvalue above the upper edge is structure that a random
matrix of the same shape could not have produced, and one inside the bulk is indistinguishable
from noise however suggestive the scree plot looks.

This is the whole reason the factor count is not chosen by eye. At `N = 468, T = 252` the edge
sits at **5.58**: a component has to carry more than five times an average name's variance
before it is distinguishable from noise. "The first twenty components look important" is not
an argument; this is.

### The market adjustment, and which way it cuts

The plain edge assumes the entire correlation matrix is noise, which on equity data is wrong
in a specific direction: the market mode alone carries ~30% of the total variance, and that
variance is not available to the noise bulk. Since the eigenvalues of a correlation matrix sum
to exactly `N`,

```
sigma^2 = (sum of eigenvalues, excluding the largest) / N  =  1 - lambda_1 / N
```

is the share of variance the bulk actually has (Laloux et al. 1999), and the edge becomes
`sigma^2 (1 + sqrt(q))^2`.

**Note the direction**, because it is the opposite of what "correcting for noise" sounds like.
Shrinking `sigma^2` *narrows* the bulk and *lowers* the edge, so **more** components come out
significant, not fewer. The unadjusted edge is the conservative one — it hands the market's
entire variance to the noise and then asks what could not be noise, which systematically
under-counts factors. Over a 252-day window the count moves from 11 to 15. Both are reported.

It is applied **once**, not iterated to a fixed point. Iterating is tempting: each pass
reclassifies more eigenvalues as deviating, which lowers `sigma^2`, which narrows the edge,
which reclassifies more. On real equity data that recursion does not converge to anything
meaningful — an early version of this code drove `sigma^2` to 0.11 and declared 283 of 468
components significant. The one-step version is the published one.

> **Gate 2** generates twelve panels of pure noise and compares the pooled empirical eigenvalue
> distribution against the theoretical CDF by a Kolmogorov-Smirnov distance. This is the
> analogue of project 03's simulated-ADF-critical-values gate: the project generates its own
> null and checks it against the published theory, so a mis-scaled implementation cannot
> quietly move every factor-count decision in the study. **Gate 3** then requires zero factors
> on noise and exactly `k` on a panel built with `k`.

---

## 4. Ridge regression through the SVD

With `Z` the standardized design and `SVD(Z) = U S V'`, the ridge solution is

```
beta(alpha) = V diag( s_j / (s_j^2 + alpha) ) U' y
```

The intercept is never penalised — it is recovered from the means afterwards, which is the
same thing as leaving its column out of the penalty matrix and is less error-prone.

Writing it this way makes the behaviour legible. Ridge is not "shrink everything a bit": it
applies a filter `s_j^2 / (s_j^2 + alpha)` to each singular direction, so directions the data
constrains well (`s_j^2 >> alpha`) pass through untouched and directions it barely constrains
are suppressed. Summing that filter gives the **effective degrees of freedom**

```
df(alpha) = sum_j s_j^2 / (s_j^2 + alpha)
```

which equals `p` at `alpha = 0` and falls toward 0 as the penalty grows. It is the honest way
to say how much a ridge has actually constrained a fit; the nominal count of 14 does not change
with `alpha` and so says nothing.

Solving by SVD rather than inverting `Z'Z + alpha I` costs a little arithmetic and buys
numerical stability when the features are nearly collinear — which 14 overlapping technical
signals certainly are.

> **Gate 5** checks the exact identity: on an orthonormal design (`Z'Z = I`, so every `s_j = 1`)
> ridge must equal OLS divided by `1 + alpha`. An implementation that penalised the intercept,
> or scaled `alpha` by the sample size, fails here and could otherwise be hidden by choosing a
> different `alpha`. **Gate 6** checks coefficient recovery on a known linear DGP and that
> `alpha = 0` reproduces least squares to machine precision.

---

## 5. Regression trees, and the histogram split search

A CART regression tree splits a node to maximise the reduction in squared error. Writing `S`
for the sum of the target over a side and `n` for its count, the reduction from splitting a
node into `L` and `R` is

```
gain = S_L^2 / n_L  +  S_R^2 / n_R  -  S^2 / n
```

which needs only sums and counts — no variances, no second passes.

**The performance problem.** Evaluating every candidate threshold by sorting each feature at
each node is `O(n log n)` per feature per node, and on 468,936 rows in pure Python that is
unusable. The fix is the one LightGBM uses: quantise each feature to 64 bins **once**, then
accumulate the target sums into those bins with `np.bincount` and turn every candidate split
into a cumulative sum. A feature then costs one pass over the node's rows regardless of how
many split points it has, which makes the whole 200-tree fit `O(n)` per feature per depth
level. The 300-tree Friedman benchmark fits in about a second.

Two implementation details are load-bearing:

- **The binner is fitted on training rows only.** A binner fitted on the test set — or on both
  — lets the test period's feature distribution set the split candidates. It is a mild leak
  and an almost invisible one.
- **The node's rows are gathered once, transposed.** Indexing `binned[rows, j]` inside the
  feature loop repeats the same expensive gather 14 times per node; hoisting it out is most of
  the runtime.

**What binning costs.** A split can only fall on a bin edge. On 500 continuous rows in 64 bins
that costs at most ~4% of the best available gain at the root; on this project's features,
which are cross-sectional ranks with far fewer distinct values than bins, it costs nothing.

> **Gate 7** states the precise claim and tests it: among the thresholds the binning *can*
> express, the search must find the best one exactly. Brute force is run over the binner's own
> edges and the two must agree to 1e-9. The looser question — what binning itself costs — is
> reported in the gate's detail line rather than the statistic, because it is a property of
> the bin count and not of the search.

---

## 6. Gradient boosting

Under squared-error loss the negative gradient of the loss at stage `m` is just the residual,
so a gradient-boosting machine reduces to something very short:

```
F_0(x) = mean(y)
for m = 1..M:
    r_i   = y_i - F_{m-1}(x_i)
    h_m   = regression tree fitted to r
    F_m   = F_{m-1} + nu * h_m
```

`nu` is the learning rate. The same skeleton with a different gradient gives logistic or Huber
boosting, which is the reason for writing it out rather than importing it.

`subsample < 1` fits each tree on a random subset of rows (Friedman 2002). On this data it is
not a speed optimisation — it is regularisation, and it matters more than the tree count.

The defaults are deliberately timid: depth 3, 200 trees, learning rate 0.05, 500 rows minimum
per leaf. A cross-sectional equity signal has a signal-to-noise ratio around 1%; a model with
the capacity to fit it exactly is a model that has fitted the noise.

> **Gate 8** requires the ensemble to reach `R^2 > 0.85` out of sample on Friedman #1 — which
> contains an `x1*x2` interaction and a quadratic term, so a linear model has a ceiling on it
> no amount of data removes — and to beat ridge by a wide margin. The second half is what would
> catch a "boosting" implementation that had quietly collapsed to a constant plus a linear fit.
> The unit tests add that the training loss is monotone non-increasing (it must be, stage by
> stage, under squared error) and that the five pure-noise features take under 1% of the
> importance.

---

## 7. Purging and embargoing

The label is a 5-day forward return, so the observation dated Monday is still resolving on
Friday. Any split that puts Monday in training and Wednesday in testing has let three days of
the test period's returns into the training set — not through a mis-shifted feature, but
through the label itself. The model never sees a future feature and still learns from future
prices.

A naive walk-forward does not escape this: the boundary between train and test is exactly
where the overlap lives. López de Prado's fix has two parts.

**Purge.** Drop from training every observation whose label window reaches into the test
period. With training ending at `T0` and a horizon of `h`, that is the last `h` training dates.

**Embargo.** Drop a further `e` days before the test period. Purging handles the mechanical
overlap; the embargo handles serial correlation, which purging misses. A training row from the
day before the test period has a label that does not overlap, but it is drawn from a market
state so close to the test period's that it functions as a partial copy — features and returns
are both autocorrelated across a boundary that thin.

With `h = 5` and `e = 5`, ten dates × 468 names = 4,680 rows leave each training set, out of
231,192 in the first fold. The cost is 2%; the benefit is that the out-of-sample number means
what it says.

> **Gate 9** carries its own control, which is the point. "No overlaps found" is worthless
> unless the same counter finds overlaps when purging is switched off, so the gate reports both
> and passes only if the unpurged count is strictly positive (500) while the purged count is
> exactly zero.

---

## 8. The staggered portfolio

The model answers "what will happen over the next week", so rebalancing to a fresh prediction
every day trades a position five times over the horizon it was meant to be held for, and pays
each time.

The fix is Jegadeesh and Titman's (1993) overlapping-portfolio construction: run `h`
sub-books, each rebalanced once every `h` days and holding `1/h` of the capital. The aggregate
book is then the trailing `h`-day average of the daily target weights — a rolling mean, and
exactly the right one, because it holds each day's prediction for precisely the horizon it was
trained to predict.

Turnover falls by roughly a factor of `h` because only `1/h` of the book rolls on any given
day, while the average holding period rises to the horizon. In this project it is the single
largest lever on net performance and §4 of the write-up quantifies it.

---

## 9. Was it alpha? Newey-West attribution

A long/short book produces a return series; regressing it on the principal components of the
same universe says what that series *is*:

```
r_t = alpha + sum_k beta_k F_kt + e_t
```

`beta_k F_kt` is what a portfolio of the statistical factors would have delivered — the model
discovered an exposure, not an edge. `alpha` is what remains.

The standard errors have to be Newey-West, and that is not a formality. The staggered book
holds the same positions for a week *by construction*, so its daily returns are autocorrelated
by design; ordinary OLS standard errors on such a series understate the uncertainty by a factor
that grows with the overlap, and an alpha t-statistic of 2.5 can become 1.4 once it is
accounted for. The estimator is

```
S = sum_t u_t u_t'  +  sum_{l=1..L} (1 - l/(L+1)) (Gamma_l + Gamma_l')
```

with `u_t = x_t e_t` and `Gamma_l` the lag-`l` cross-product. The Bartlett weights
`1 - l/(L+1)` are what make `S` positive semi-definite; a plain truncated sum is not, and can
produce a negative variance on exactly the kind of series being fitted here. `L` defaults to
`floor(4 (n/100)^(2/9))`.

The decomposition into factor-explained and residual parts is exact — the two components sum
to the realized annualized mean by construction — so the table it produces cannot quietly fail
to add up.

---

## 10. The residualization trap

This section documents a bug, because it produced a beautiful result and every other check in
the project passed while it was there.

To ask "can the model predict the part of a return a factor portfolio does not already give
you", the returns are residualized against a causal rolling PCA: fit on the trailing 252 days,
apply forward for 21, refit. The residual is `r_t` minus the part spanned by the retained
components.

The first implementation computed that by un-standardizing the reconstruction the same way the
fit standardized it — subtract the mean, divide by the scale, project, multiply by the scale,
**add the mean back**. The stored mean is each stock's average daily return over the *fitting*
window. So every residual had the stock's trailing 252-day drift subtracted from it, and a past
winner carried a mechanically negative forward residual.

The consequence: on a synthetic panel with **no predictability whatsoever**, momentum
"predicted" the forward residual return with an IC information ratio of **−1.34**. On the real
panel it produced a stable, plausible, large signal that would have been this project's
headline result. The eigenvalues were right, the factor counts were right, and no future data
was used anywhere.

The fix is to scale but not centre when applying the model. Fitting must centre — a covariance
matrix is defined on centred data — but applying must not, which leaves the residual a pure
linear function of the same day's cross-section: `r_t - B f_t`, no drift term anywhere.

> **Gate 4** is the check that catches it, and it is worth noting what makes the null correct.
> The panel must have **no cross-sectional dispersion in drift**. Give each stock its own
> constant drift and momentum becomes genuinely predictive — of the total return and, more
> strongly, of the residual, because removing the factor variance leaves the drift a larger
> share of what remains. That is a real effect, not an artefact, so a null containing it cannot
> test for artefacts. A wrong version of this gate, built on a panel with per-stock drift,
> "failed" the corrected code at +0.74 and would have sent the fix in the wrong direction.

---

## References

- Marchenko, V. and Pastur, L. (1967). *Distribution of eigenvalues for some sets of random
  matrices.* Mathematics of the USSR-Sbornik.
- Laloux, L., Cizeau, P., Bouchaud, J.-P. and Potters, M. (1999). *Noise dressing of financial
  correlation matrices.* Physical Review Letters 83.
- Hoerl, A. and Kennard, R. (1970). *Ridge regression: biased estimation for nonorthogonal
  problems.* Technometrics 12.
- Hastie, T., Tibshirani, R. and Friedman, J. (2009). *The Elements of Statistical Learning*,
  2nd ed. — §3.4.1 for the ridge filter and effective degrees of freedom.
- Breiman, L., Friedman, J., Olshen, R. and Stone, C. (1984). *Classification and Regression
  Trees.*
- Friedman, J. (2001). *Greedy function approximation: a gradient boosting machine.* Annals of
  Statistics 29.
- Friedman, J. (2002). *Stochastic gradient boosting.* Computational Statistics and Data
  Analysis 38.
- Ke, G. et al. (2017). *LightGBM: a highly efficient gradient boosting decision tree.* NIPS —
  for the histogram-based split search.
- López de Prado, M. (2018). *Advances in Financial Machine Learning*, ch. 7 — purging and
  embargoing.
- Jegadeesh, N. and Titman, S. (1993). *Returns to buying winners and selling losers.* Journal
  of Finance 48.
- Newey, W. and West, K. (1987). *A simple, positive semi-definite, heteroskedasticity and
  autocorrelation consistent covariance matrix.* Econometrica 55.
