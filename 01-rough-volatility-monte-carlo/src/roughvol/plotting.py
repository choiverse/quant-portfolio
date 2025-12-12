"""Figures for the rough-volatility study (matplotlib only).

All figures share one theme and one colorblind-validated categorical order, so
the report reads as a single document. Series are assigned hue slots in order and
never recycled; magnitude uses a single-hue sequential ramp.

Public entry points
-------------------
``results_figure``      the headline three-panel summary
``roughness_figure``    reading the Hurst exponent back out of the paths
``convergence_figure``  Monte Carlo error vs path count, per estimator variant
``surface_figure``      the implied-vol surface the model generates
``sensitivity_figure``  which parameter actually controls the short-dated skew
``validation_figure``   the correctness gates as a dashboard
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from .style import (
    AXIS,
    CRITICAL,
    GOOD,
    GRID,
    INK,
    INK_MUTED,
    INK_SECONDARY,
    SEQ_BLUE,
    SERIES,
    SURFACE,
    annotate_source,
    use_style,
)

SOURCE = ("Simulated data — rough Bergomi (Bayer–Friz–Gatheral) via the hybrid scheme "
          "(Bennedsen–Lunde–Pakkanen, κ=1). Seeds fixed; see docs/METHOD.md.")


# --------------------------------------------------------------------------
# 1. Headline results
# --------------------------------------------------------------------------
def results_figure(variance_paths, smiles, skew_df, skew_slope, H, savepath=None):
    """Three-panel summary: rough paths, the smile they generate, the power law.

    Parameters
    ----------
    variance_paths : dict[label -> (t, v_sample)]
    smiles         : dict[label -> DataFrame(index=K, cols log_moneyness, iv)]
    skew_df        : DataFrame(index=T, col 'skew')
    skew_slope     : fitted log-log slope of |skew| vs T
    H              : Hurst exponent for the theoretical reference line
    """
    use_style()
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6))
    fig.suptitle(
        "Rough Bergomi — from a jagged variance path to the empirical skew power law",
        fontsize=13, fontweight="bold", color=INK,
    )

    ax = axes[0]
    for i, (label, (t, v)) in enumerate(variance_paths.items()):
        ax.plot(t, v, linewidth=1.0, label=label, color=SERIES[i])
    ax.set_title("(a) Simulated variance paths")
    ax.set_xlabel("time (years)")
    ax.set_ylabel(r"instantaneous variance $v_t$")
    ax.legend(loc="upper right")

    ax = axes[1]
    for i, (label, df) in enumerate(smiles.items()):
        d = df.dropna(subset=["iv"])
        ax.plot(d["log_moneyness"], d["iv"] * 100, marker="o", ms=4,
                label=label, color=SERIES[i])
    ax.set_title("(b) Implied-vol smile at T = 0.1y")
    ax.set_xlabel(r"log-moneyness  $\ln(K/S_0)$")
    ax.set_ylabel("implied vol (%)")
    ax.legend(loc="upper right")

    ax = axes[2]
    T = skew_df.index.values
    skew = skew_df["skew"].abs().values
    theo_slope = H - 0.5
    anchor = skew[0] / T[0] ** theo_slope
    ax.loglog(T, anchor * T ** theo_slope, "--", color=SERIES[1], linewidth=2.0,
              label=rf"theory  $T^{{H-1/2}}$, slope {theo_slope:.2f}")
    ax.loglog(T, skew, "o", ms=6, color=SERIES[0], label="Monte Carlo ATM skew",
              markeredgecolor="white", markeredgewidth=1.0)
    ax.set_title(f"(c) ATM skew power law — fitted slope {skew_slope:.3f}")
    ax.set_xlabel("maturity T (years)")
    ax.set_ylabel("|ATM skew|")
    ax.legend(loc="lower left")
    ax.grid(True, which="both", color=GRID, linewidth=0.6)

    fig.tight_layout(rect=(0, 0, 1, 0.92))
    annotate_source(fig, SOURCE)
    if savepath:
        fig.savefig(savepath)
    return fig


# --------------------------------------------------------------------------
# 2. Roughness recovered from the paths
# --------------------------------------------------------------------------
def roughness_figure(scalings: dict, paths: dict, savepath=None):
    """Does the simulator actually produce the roughness it was asked for?

    ``scalings`` maps a label to ``(DataFrame[delta, msq], fitted_H)``;
    ``paths`` maps a label to ``(t, v)`` for the illustrative sample paths.
    """
    use_style()
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))
    fig.suptitle(
        "Roughness is measured, not assumed — the Hurst exponent is read back off the paths",
        fontsize=13, fontweight="bold", color=INK,
    )

    # (a) The paths themselves, zoomed so the difference is visible.
    ax = axes[0]
    for i, (label, (t, v)) in enumerate(paths.items()):
        mask = t <= 0.25
        ax.plot(t[mask], v[mask], linewidth=1.0, label=label, color=SERIES[i])
    ax.set_title("(a) First 3 months of a variance path")
    ax.set_xlabel("time (years)")
    ax.set_ylabel(r"$v_t$")
    ax.legend(loc="upper right")

    # (b) Mean squared increment vs lag — slope is 2H.
    ax = axes[1]
    for i, (label, (df, fitted_H)) in enumerate(scalings.items()):
        ax.loglog(df["delta"], df["msq"], "o", ms=5, color=SERIES[i],
                  markeredgecolor="white", markeredgewidth=0.8,
                  label=f"{label} → fitted H = {fitted_H:.3f}")
        fit = np.polyfit(np.log(df["delta"]), np.log(df["msq"]), 1)
        ax.loglog(df["delta"], np.exp(np.polyval(fit, np.log(df["delta"]))),
                  "--", color=SERIES[i], linewidth=1.6)
    ax.set_title(r"(b) $E[(Y_{t+\Delta}-Y_t)^2] \propto \Delta^{2H}$")
    ax.set_xlabel(r"lag $\Delta$ (years)")
    ax.set_ylabel("mean squared increment")
    ax.legend(loc="upper left")
    ax.grid(True, which="both", color=GRID, linewidth=0.6)

    # (c) Recovered vs specified H.
    ax = axes[2]
    specified = [float(lbl.split("=")[-1]) for lbl in scalings]
    recovered = [h for _, h in scalings.values()]
    lim = (0, max(max(specified), max(recovered)) * 1.25)
    ax.plot(lim, lim, "--", color=INK_MUTED, linewidth=1.4, label="perfect recovery")
    ax.plot(specified, recovered, "o", ms=10, color=SERIES[0],
            markeredgecolor="white", markeredgewidth=1.5, label="measured")
    for s, r in zip(specified, recovered):
        ax.annotate(f"  {r:.3f}", (s, r), fontsize=8, color=INK_SECONDARY, va="center")
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_title("(c) Recovered H vs specified H")
    ax.set_xlabel("H given to the simulator")
    ax.set_ylabel("H estimated from the output")
    ax.legend(loc="upper left")

    fig.tight_layout(rect=(0, 0, 1, 0.92))
    annotate_source(fig, SOURCE)
    if savepath:
        fig.savefig(savepath)
    return fig


# --------------------------------------------------------------------------
# 3. Monte Carlo convergence
# --------------------------------------------------------------------------
def convergence_figure(conv: pd.DataFrame, reference_price: float, savepath=None):
    """Error decay, the payoff of variance reduction, and the cost of accuracy."""
    use_style()
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))
    fig.suptitle(
        "Monte Carlo convergence — error falls as N^(−1/2); variance reduction moves the intercept",
        fontsize=13, fontweight="bold", color=INK,
    )
    variants = list(dict.fromkeys(conv["variant"]))

    # (a) Standard error vs path count, with the theoretical slope.
    ax = axes[0]
    for i, variant in enumerate(variants):
        sub = conv[conv["variant"] == variant]
        ax.loglog(sub["n_paths"], sub["std_error"], "o-", ms=5, color=SERIES[i],
                  label=variant, markeredgecolor="white", markeredgewidth=0.8)
    first = conv[conv["variant"] == variants[0]]
    n0, se0 = first["n_paths"].iloc[0], first["std_error"].iloc[0]
    ns = first["n_paths"].values
    ax.loglog(ns, se0 * np.sqrt(n0 / ns), "--", color=INK_MUTED, linewidth=1.4,
              label=r"reference $N^{-1/2}$")
    ax.set_title("(a) Standard error vs paths")
    ax.set_xlabel("paths N (log)")
    ax.set_ylabel("standard error (log)")
    ax.legend(loc="lower left")
    ax.grid(True, which="both", color=GRID, linewidth=0.6)

    # (b) Price estimate with a 2-SE band against the closed form.
    ax = axes[1]
    for i, variant in enumerate(variants):
        sub = conv[conv["variant"] == variant]
        ax.fill_between(sub["n_paths"], sub["price"] - 2 * sub["std_error"],
                        sub["price"] + 2 * sub["std_error"],
                        color=SERIES[i], alpha=0.18, linewidth=0)
        ax.plot(sub["n_paths"], sub["price"], "o-", ms=4, color=SERIES[i], label=variant)
    ax.axhline(reference_price, color=INK, linewidth=1.6, linestyle="--",
               label=f"Black–Scholes = {reference_price:.4f}")
    ax.set_xscale("log")
    ax.set_title("(b) Price estimate ±2 SE vs the closed form")
    ax.set_xlabel("paths N (log)")
    ax.set_ylabel("call price")
    ax.legend(loc="upper right")

    # (c) How many paths each variant needs to match crude MC's best accuracy.
    ax = axes[2]
    target = conv[conv["variant"] == variants[0]]["std_error"].min()
    needed = []
    for variant in variants:
        sub = conv[conv["variant"] == variant]
        n_ref, se_ref = sub["n_paths"].iloc[-1], sub["std_error"].iloc[-1]
        needed.append(n_ref * (se_ref / target) ** 2)  # SE proportional to N^{-1/2}
    bars = ax.bar(range(len(variants)), needed, color=SERIES[: len(variants)],
                  width=0.62, linewidth=0)
    ax.set_yscale("log")
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels([v.replace(" + ", "\n+ ") for v in variants], fontsize=8)
    ax.set_title("(c) Paths needed to match crude MC accuracy")
    ax.set_ylabel("paths required (log)")
    for bar, n in zip(bars, needed):
        speedup = needed[0] / n
        ax.annotate(f"{n:,.0f}\n({speedup:.1f}x cheaper)",
                    xy=(bar.get_x() + bar.get_width() / 2, n), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=8,
                    color=INK_SECONDARY, fontweight="bold")
    ax.set_ylim(top=max(needed) * 12)
    best = variants[int(np.argmin(needed))]
    ax.annotate(
        f"best: {best} alone —\nstacking both is worse",
        xy=(0.02, 0.97), xycoords="axes fraction", va="top",
        fontsize=8.5, color=INK_SECONDARY, fontweight="bold",
    )

    fig.tight_layout(rect=(0, 0, 1, 0.92))
    annotate_source(fig, SOURCE)
    if savepath:
        fig.savefig(savepath)
    return fig


# --------------------------------------------------------------------------
# 4. Implied-vol surface
# --------------------------------------------------------------------------
def surface_figure(surface: pd.DataFrame, savepath=None):
    """The whole surface, plus slices that show the short end steepening."""
    use_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8),
                             gridspec_kw={"width_ratios": [1.25, 1]})
    fig.suptitle(
        "The implied-vol surface rough Bergomi generates",
        fontsize=13, fontweight="bold", color=INK,
    )

    # (a) Heatmap over (moneyness, maturity) — magnitude, so a single hue.
    ax = axes[0]
    im = ax.pcolormesh(
        surface.columns.values.astype(float),
        surface.index.values.astype(float),
        surface.values * 100,
        cmap=SEQ_BLUE, shading="nearest",
    )
    ax.set_yscale("log")
    ax.set_xlabel(r"log-moneyness  $\ln(K/S_0)$")
    ax.set_ylabel("maturity T (years, log)")
    ax.set_title("(a) Implied vol (%)")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("implied vol (%)")
    ax.annotate(
        "blank cells: deep out-of-the-money at short maturity, where the Monte Carlo price sits\n"
        "within 3 standard errors of zero — dropped rather than inverted into a meaningless vol",
        xy=(0.5, -0.42), xycoords="axes fraction", ha="center",
        fontsize=7.5, color=INK_MUTED,
    )

    # (b) Term slices — the steepening at the short end is the whole point.
    ax = axes[1]
    chosen = list(surface.index[:: max(1, len(surface) // 5)])[:5]
    for i, T in enumerate(chosen):
        row = surface.loc[T].dropna()
        ax.plot(row.index.astype(float), row.values * 100, marker="o", ms=4,
                color=SERIES[i], label=f"T = {T:g}y")
    ax.set_xlabel(r"log-moneyness  $\ln(K/S_0)$")
    ax.set_ylabel("implied vol (%)")
    ax.set_title("(b) Smile slices — short maturities are steepest")
    ax.legend(loc="upper right")

    fig.tight_layout(rect=(0, 0.10, 1, 0.91))
    annotate_source(fig, SOURCE)
    if savepath:
        fig.savefig(savepath)
    return fig


# --------------------------------------------------------------------------
# 5. Parameter sensitivity
# --------------------------------------------------------------------------
def sensitivity_figure(sweeps: dict[str, pd.DataFrame], savepath=None):
    """One panel per swept parameter: what it does to the skew and to the level."""
    use_style()
    n = len(sweeps)
    fig, axes = plt.subplots(1, n, figsize=(4.9 * n, 4.3), squeeze=False)
    fig.suptitle(
        "Which parameter sets the short-dated skew? (T = 0.1y, one knob at a time)",
        fontsize=13, fontweight="bold", color=INK,
    )

    labels = {"H": "Hurst exponent H", "eta": "vol-of-vol η", "rho": "spot-vol correlation ρ"}
    for ax, (param, df) in zip(axes[0], sweeps.items()):
        # One series only — the axis label names it, so no legend box is needed.
        ax.plot(df.index.values, df["atm_skew"].values, "o-", ms=6, color=SERIES[0],
                markeredgecolor="white", markeredgewidth=1.0)
        ax.axhline(0, color=AXIS, linewidth=1.0)
        ax.set_xlabel(labels.get(param, param))
        ax.set_ylabel(r"ATM skew  $\partial\sigma_{BS}/\partial\ln K$")
        ax.set_title(f"sweeping {labels.get(param, param)}")
        rng = df["atm_iv"].max() - df["atm_iv"].min()
        # Placed below the axis so it can never collide with the curve.
        ax.annotate(
            f"ATM vol level moves only {rng * 100:.1f} vol pts across this sweep",
            xy=(0.5, -0.26), xycoords="axes fraction", ha="center",
            fontsize=8, color=INK_SECONDARY,
        )

    fig.tight_layout(rect=(0, 0.04, 1, 0.90))
    annotate_source(fig, SOURCE)
    if savepath:
        fig.savefig(savepath)
    return fig


# --------------------------------------------------------------------------
# 6. Validation dashboard
# --------------------------------------------------------------------------
def validation_figure(gates: pd.DataFrame, savepath=None):
    """The correctness gates as a chart: statistic vs its tolerance, per gate."""
    use_style()
    fig, ax = plt.subplots(figsize=(10, 4.2))
    fig.suptitle(
        "Correctness gates — every claim in this project is checked against a known answer",
        fontsize=13, fontweight="bold", color=INK,
    )

    y = np.arange(len(gates))[::-1]
    # Normalize each gate to "fraction of its own budget used", so one axis works
    # for tests measured in |z|, in relative error, and in an SE ratio.
    used = []
    for _, row in gates.iterrows():
        if row["units"] == "SE reduction x":   # bigger is better
            used.append(row["tolerance"] / row["statistic"])
        else:                                  # smaller is better
            used.append(row["statistic"] / row["tolerance"])

    colors = [GOOD if p else CRITICAL for p in gates["passed"]]
    ax.barh(y, used, color=colors, height=0.55, linewidth=0)
    ax.axvline(1.0, color=INK, linewidth=1.6, linestyle="--")
    ax.annotate("tolerance", xy=(1.0, len(gates) - 0.4), xytext=(4, 0),
                textcoords="offset points", fontsize=9, color=INK, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(gates.index, fontsize=9)
    ax.set_xlabel("fraction of the gate's tolerance consumed (< 1 passes)")
    ax.set_xlim(0, max(1.35, max(used) * 1.25))
    ax.grid(axis="y", visible=False)

    for yi, (_, row), u in zip(y, gates.iterrows(), used):
        ax.annotate(
            f"{row['statistic']:.3g} {row['units']}  ({'PASS' if row['passed'] else 'FAIL'})",
            xy=(u, yi), xytext=(6, 0), textcoords="offset points",
            va="center", fontsize=8.5, fontweight="bold",
            color=GOOD if row["passed"] else CRITICAL,
        )

    fig.tight_layout(rect=(0, 0, 1, 0.90))
    annotate_source(fig, SOURCE)
    if savepath:
        fig.savefig(savepath)
    return fig
