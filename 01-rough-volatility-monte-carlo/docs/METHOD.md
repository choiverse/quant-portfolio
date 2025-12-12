# Method — rough Bergomi and the hybrid scheme

This document is the mathematical companion to the code. It states the model,
derives the scheme actually implemented, and records the reasoning behind the
choices a reader would otherwise have to reverse-engineer from `models.py`.

---

## 1. Why "rough" volatility at all

Classical stochastic-volatility models (Heston, Bergomi, SABR) drive variance
with a standard Brownian motion. That choice has a consequence they cannot
escape: the at-the-money implied-volatility skew

$$\psi(T) \;=\; \left|\frac{\partial \sigma_{BS}(k, T)}{\partial k}\right|_{k=0}$$

flattens to a finite constant as maturity $T \to 0$. Market data does the
opposite — the skew *explodes* at the short end, empirically as a power law
$\psi(T) \sim T^{-\alpha}$ with $\alpha \approx 0.4$.

Gatheral, Jaisson and Rosenbaum (2018) observed that realized volatility behaves
like a fractional Brownian motion with Hurst exponent $H \approx 0.1$ — far
rougher than the $H = 1/2$ of a diffusion. Bayer, Friz and Gatheral (2016)
showed that a model built on such a process produces exactly

$$\psi(T) \;\propto\; T^{H - 1/2},$$

which reproduces the observed exponent for $H \approx 0.1$. **Reproducing that
power law from a from-scratch simulator is the goal of this project.**

---

## 2. The model

Under the risk-neutral measure, rough Bergomi is

$$
\begin{aligned}
v_t &= \xi_0 \, \exp\!\left(\eta Y_t - \tfrac{1}{2}\eta^2 t^{2H}\right), \\
\frac{dS_t}{S_t} &= r\,dt + \sqrt{v_t}\,dZ_t, \\
dZ_t &= \rho\, dW_t + \sqrt{1-\rho^2}\, dW_t^{\perp},
\end{aligned}
$$

where $Y$ is the **normalized Riemann–Liouville fractional process**

$$Y_t \;=\; \sqrt{2H}\int_0^t (t-s)^{H-1/2}\, dW_s, \qquad \operatorname{Var}(Y_t) = t^{2H}.$$

| symbol | meaning | reference value here |
|---|---|---|
| $H$ | Hurst exponent — path roughness; $H<1/2$ is rough | 0.10 |
| $\eta$ | vol-of-vol — scales the skew | 1.9 |
| $\rho$ | spot/vol correlation — sets the skew's sign | −0.9 |
| $\xi_0$ | flat forward variance — sets the vol level | 0.04 (≈20% vol) |

Two structural properties are worth naming because they are what make the model
usable, and both are asserted as tests in `tests/test_roughvol.py`:

- **Variance is positive by construction.** It is an exponential, so unlike a
  discretized Heston process it can never go negative and needs no truncation
  hack.
- **The drift term $-\tfrac12\eta^2 t^{2H}$ is not decoration.** It is exactly
  the compensator that makes $\mathbb{E}[v_t] = \xi_0$, so $\xi_0$ really is the
  forward variance rather than an approximate one.

---

## 3. Simulating $Y$ — the hybrid scheme

The kernel $(t-s)^{H-1/2}$ is **singular at $s = t$** when $H < 1/2$. A plain
Euler/Riemann sum badly misprices the contribution of the cell nearest the
evaluation point, and that cell is precisely the one that carries the roughness.
Exact methods (Cholesky of the full covariance matrix) cost $O(n^2)$ memory and
$O(n^3)$ setup, which is unusable at the path counts option pricing needs.

The **hybrid scheme** of Bennedsen, Lunde and Pakkanen (2017) resolves this by
splitting the integral at the first grid cell. With $\kappa = 1$ and
$\alpha = H - \tfrac12$:

$$
\hat{Y}_{t_i} \;=\; \underbrace{\int_{t_{i-1}}^{t_i}(t_i-s)^{\alpha}dW_s}_{\text{exact}}
\;+\; \underbrace{\sum_{k=2}^{i} g\!\left(b_k^{*}\Delta t\right)\, \Delta W_{i-k+1}}_{\text{Riemann sum}}
$$

**Near cell — sampled exactly.** The pair $\left(\Delta W_i,\; W_i^{2}\right)$ where
$W_i^2 = \int_{t_{i-1}}^{t_i}(t_i-s)^{\alpha}dW_s$ is jointly Gaussian with

$$
\Sigma = \begin{pmatrix}
\Delta t & \dfrac{\Delta t^{\alpha+1}}{\alpha+1} \\[8pt]
\dfrac{\Delta t^{\alpha+1}}{\alpha+1} & \dfrac{\Delta t^{2\alpha+1}}{2\alpha+1}
\end{pmatrix}
$$

and is drawn via its Cholesky factor (`_cov_last_cell`). No discretization error
is incurred where the kernel is worst behaved.

**Far cells — optimally-placed Riemann sum.** For $k \ge 2$ the kernel is
smooth, and BLP show the discretization point that minimizes mean-square error is

$$b_k^{*} = \left(\frac{k^{\alpha+1}-(k-1)^{\alpha+1}}{\alpha+1}\right)^{1/\alpha}$$

rather than the naive left or mid point (`_optimal_weights`).

**Normalization.** Multiplying by $\sqrt{2H}$ gives
$\operatorname{Var}(Y_t) = t^{2H}$ exactly — which is validation gate 1.

### The optimization that makes it fast

The far-cell term is a **causal convolution** of the increment sequence with a
fixed weight vector. Written path-by-path (`np.apply_along_axis`) it dominated
runtime. Rewritten as a single `scipy.signal.fftconvolve` over the whole
`(n_paths, n_steps)` array it costs $O(N n \log n)$ for all paths at once:

> **200,000 paths × 100 steps in ≈1.6 s.**

### Predictability of the variance

The price increment over $[t_{i-1}, t_i)$ uses the **left endpoint**
$v_{t_{i-1}}$, with $v_0 = \xi_0$. Using the right endpoint would let the
variance over a step depend on the same Brownian increment that drives the price
over that step — a subtle look-ahead that biases the price and breaks the
martingale property (validation gate 5).

---

## 4. Variance reduction — and a trap

Two techniques are implemented in `mc_pricer.py`:

**Antithetic variates.** For every draw $z$, also use $-z$. Both are valid
samples, and for a payoff monotone in $z$ the pair is negatively correlated, so
the pair *average* has lower variance.

**Control variate.** The discounted terminal price $e^{-rT}S_T$ has a mean known
exactly ($S_0$). Regressing the discounted payoff on it and subtracting
$\beta\,(e^{-rT}S_T - S_0)$ removes the explained component without introducing
bias.

### Two things this project got wrong first, and fixed

**(a) The standard error of an antithetic estimator.** The initial
implementation reported `std(values)/sqrt(N)` over all $N$ draws. That is wrong:
antithetic samples are *deliberately dependent*, and treating them as
independent measures the crude-Monte-Carlo error no matter how much variance the
pairing removed. Antithetic sampling consequently appeared to do **nothing** —
identical standard errors to four significant figures. The correct estimator has
$N/2$ independent observations, each the average of one antithetic pair:

$$\widehat{\operatorname{SE}} = \frac{1}{\sqrt{N/2}}\;\operatorname{sd}\!\left(\frac{f(z_i)+f(-z_i)}{2}\right).$$

The point estimate is unchanged; the error bar was the thing that was lying.
Fixed in `_standard_error`, pinned by
`test_standard_error_uses_antithetic_pairs`.

**(b) Stacking the two techniques is worse than using one.** With the
standard error computed correctly, measured on a 1-year 105-strike call at
400,000 paths:

| estimator | standard error | vs crude MC |
|---|---|---|
| crude Monte Carlo | 0.01931 | 1.00× |
| antithetic only | 0.01613 | 1.20× |
| **control variate only** | **0.00968** | **2.00×** |
| antithetic + control variate | 0.01294 | 1.49× |

The combination is *worse than the control variate alone*, and the reason is
structural rather than numerical. The control variate removes the component of
the payoff that is linear in $S_T$ — which is the odd-in-$z$ component that
antithetic sampling exists to cancel. What survives the regression is close to an
**even** function of $z$, and for an even function the antithetic pair is
*positively* correlated, so pairing adds variance back. Shown in panel (c) of
`reports/figures/03_convergence.png`.

---

## 5. Building the smile

- **Common random numbers.** All strikes at one maturity are priced from a
  single simulated sample, so the smile is smooth rather than jagged from
  independent sampling noise.
- **Out-of-the-money contracts only** — puts below the forward, calls above.
  OTM options carry the informative Monte Carlo signal on each wing; ITM prices
  are dominated by intrinsic value, which swamps the vol information.
- **Points within 3 standard errors of zero are dropped.** Deep OTM at short
  maturity, the price is statistically indistinguishable from zero, and
  inverting such a price yields a meaningless implied vol. These appear as blank
  cells in the surface figure — stated explicitly rather than silently
  interpolated.
- **ATM skew** is a central difference across two strikes bracketing the
  forward, both priced off the same simulation so the difference is not
  contaminated by sampling noise.

---

## 6. Reproducibility

| item | value |
|---|---|
| RNG | `numpy.random.default_rng` (PCG64) |
| Seeds | fixed per script — 0/7/11/21 (`run_experiment.py`), 5/13/17/23 (`make_figures.py`), 1–4 (gates) |
| Determinism | same seed ⇒ bit-identical output (`test_same_seed_gives_identical_output`) |
| Runtime | ≈40 s for `run_experiment.py`, ≈55 s for `make_figures.py`, ≈15 s for `validate.py` on a laptop CPU |
| Memory | < 1 GB at 400,000 paths × 100 steps |
| Dependencies | NumPy, SciPy, pandas, matplotlib — no quant libraries |

No external data is consumed: every number in this project is generated by the
code. See `data/README.md` for the simulated-data card.

---

## 7. References

Ordered by what they contribute here. The rough-volatility literature is active
and **contested** — the last two entries are included because a project that
only cited the papers supporting its premise would be advocacy, not research.

**The model and the empirical claim**
- Gatheral, J., Jaisson, T., Rosenbaum, M. (2018). *Volatility is rough.*
  Quantitative Finance 18(6), 933–949. — the empirical $H \approx 0.1$ finding.
- Bayer, C., Friz, P., Gatheral, J. (2016). *Pricing under rough volatility.*
  Quantitative Finance 16(6), 887–904. — the rough Bergomi model.

**The simulation scheme implemented here**
- Bennedsen, M., Lunde, A., Pakkanen, M. (2017). *Hybrid scheme for Brownian
  semistationary processes.* Finance and Stochastics 21(4), 931–965.
- McCrickerd, R., Pakkanen, M. (2018). *Turbocharging Monte Carlo pricing for
  the rough Bergomi model.* Quantitative Finance 18(11), 1877–1886.

**Surveys and later developments**
- Bayer, C., Friz, P., Gatheral, J., Gulisashvili, A., Horvath, B., Jacquier, A.,
  Muguruza, A. (2023). *Rough Volatility.* SIAM. — the current book-length
  treatment of the field.
- Bayer, C., Breneis, S. (2023). *Markovian approximations of stochastic Volterra
  equations with the fractional kernel.* Quantitative Finance 23(1). — how the
  field made rough models tractable beyond brute-force Monte Carlo.

**The case against — read these before believing the result**
- Rogers, L.C.G. (2019, rev. 2023). *Things we think we know.* — argues the
  evidence for roughness is weaker than claimed.
- Cont, R., Das, P. (2024). *Rough volatility: fact or artefact?* Sankhya B
  86(1). — shows that estimating $H$ from *realized* volatility can produce
  $\hat H \approx 0.1$ even when the underlying spot volatility process is not
  rough, because the estimator picks up microstructure noise and the
  discretization of the realized-variance proxy rather than the true regularity.

**What that means for this project.** The Cont–Das critique is about *estimating*
$H$ from market data. It does not touch what is demonstrated here — that a rough
Bergomi simulator generates the $T^{H-1/2}$ skew law — but it does bear directly
on the motivating sentence "markets have $H \approx 0.1$". The honest statement
of the result is therefore: **the model reproduces the power law it is claimed to
reproduce, and whether real markets are genuinely rough remains open.**
