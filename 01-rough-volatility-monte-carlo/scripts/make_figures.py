"""Generate the full figure set for the rough-volatility study.

``run_experiment.py`` produces the headline result. This script produces the
evidence *around* it — the roughness measurement, the convergence study, the
surface, the parameter sweeps, and the validation dashboard.

    reports/figures/02_roughness.png
    reports/figures/03_convergence.png
    reports/figures/04_vol_surface.png
    reports/figures/05_parameter_sensitivity.png
    reports/figures/06_validation.png
    reports/tables/*.csv

Usage
-----
    python scripts/make_figures.py            # everything (~3-5 min)
    python scripts/make_figures.py --quick    # smaller path counts
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Console output contains maths symbols; a non-UTF-8 default console (cp949 on a
# Korean Windows install, cp1252 elsewhere) would otherwise raise on print().
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from roughvol import analysis  # noqa: E402
from roughvol.black_scholes import bs_price  # noqa: E402
from roughvol.models import GBM, RoughBergomi  # noqa: E402
from roughvol.plotting import (  # noqa: E402
    convergence_figure,
    roughness_figure,
    sensitivity_figure,
    surface_figure,
    validation_figure,
)

FIGURES = ROOT / "reports" / "figures"
TABLES = ROOT / "reports" / "tables"

# Reference calibration, shared with run_experiment.py.
S0, R = 100.0, 0.0
H_ROUGH, ETA, RHO, XI0 = 0.10, 1.9, -0.9, 0.04


def rb_factory(H=H_ROUGH, eta=ETA, rho=RHO, xi0=XI0):
    def make(T):
        return RoughBergomi(S0=S0, r=R, T=T, H=H, eta=eta, rho=rho, xi0=xi0,
                            n_steps=max(50, int(200 * T)))
    return make


def main() -> None:
    p = argparse.ArgumentParser(description="Figure set for the rough-volatility study.")
    p.add_argument("--quick", action="store_true", help="fewer paths; rougher numbers")
    args = p.parse_args()

    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    scale = 0.25 if args.quick else 1.0
    t0 = time.time()

    # --- 1. Roughness recovered from the simulated paths -------------------
    print("[1/5] Measuring roughness (recovering H from the output) ...")
    scalings, paths = {}, {}
    for H in (0.10, 0.25, 0.45):
        df, fitted = analysis.roughness_scaling(
            H, n_paths=int(20_000 * scale) or 2_000, n_steps=400, seed=5
        )
        scalings[f"H = {H}"] = (df, fitted)
        df.assign(fitted_H=fitted).to_csv(TABLES / f"roughness_scaling_H{H:.2f}.csv")
        print(f"      specified H = {H:.2f}  →  recovered {fitted:.4f}")

        model = RoughBergomi(S0=S0, r=R, T=1.0, H=H, eta=ETA, rho=RHO, xi0=XI0, n_steps=600)
        _, v, t = model.simulate_paths(1, seed=7, antithetic=False)
        paths[f"H = {H}"] = (t, v[0])

    roughness_figure(scalings, paths, savepath=str(FIGURES / "02_roughness.png"))

    # --- 2. Convergence & variance reduction ------------------------------
    print("[2/5] Convergence study (GBM, where the exact answer is known) ...")
    sigma, K, T = 0.2, 105.0, 1.0
    gbm = GBM(S0=S0, r=0.02, T=T, sigma=sigma)
    counts = tuple(int(n * scale) or 1_000
                   for n in (2_000, 5_000, 10_000, 25_000, 50_000, 100_000, 200_000, 400_000))
    conv = analysis.convergence_study(gbm, K, path_counts=counts, seed=13)
    conv.to_csv(TABLES / "convergence.csv", index=False)
    reference = float(bs_price(S0, K, T, 0.02, sigma, "call"))
    convergence_figure(conv, reference, savepath=str(FIGURES / "03_convergence.png"))
    final = conv[conv["n_paths"] == counts[-1]].set_index("variant")["std_error"]
    print(f"      SE at N={counts[-1]:,}: " +
          ", ".join(f"{k} {v:.5f}" for k, v in final.items()))

    # --- 3. Implied-vol surface -------------------------------------------
    print("[3/5] Building the implied-vol surface ...")
    maturities = np.array([0.05, 0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.0])
    moneyness = np.round(np.linspace(-0.25, 0.25, 17), 4)
    surface = analysis.implied_vol_surface(
        rb_factory(), maturities, moneyness, S0=S0,
        n_paths=int(200_000 * scale) or 25_000, seed=17,
    )
    surface.to_csv(TABLES / "implied_vol_surface.csv")
    surface_figure(surface, savepath=str(FIGURES / "04_vol_surface.png"))

    # --- 4. Parameter sensitivity -----------------------------------------
    print("[4/5] Sweeping H, eta, rho ...")
    base = dict(H=H_ROUGH, eta=ETA, rho=RHO, xi0=XI0)
    n_sens = int(150_000 * scale) or 25_000
    sweeps = {
        "H": analysis.skew_sensitivity(base, "H", [0.05, 0.1, 0.2, 0.3, 0.45],
                                       n_paths=n_sens),
        "eta": analysis.skew_sensitivity(base, "eta", [0.5, 1.0, 1.5, 1.9, 2.5],
                                         n_paths=n_sens),
        "rho": analysis.skew_sensitivity(base, "rho", [-0.99, -0.9, -0.7, -0.4, 0.0],
                                         n_paths=n_sens),
    }
    for name, df in sweeps.items():
        df.to_csv(TABLES / f"sensitivity_{name}.csv")
    sensitivity_figure(sweeps, savepath=str(FIGURES / "05_parameter_sensitivity.png"))

    # --- 5. Validation dashboard ------------------------------------------
    print("[5/5] Running the validation gates ...")
    gates = analysis.validation_gates()
    gates.to_csv(TABLES / "validation_gates.csv")
    validation_figure(gates, savepath=str(FIGURES / "06_validation.png"))
    print(gates.to_string())

    print(f"\nDone in {time.time() - t0:.1f}s. Figures in {FIGURES.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
