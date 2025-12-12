"""Main experiment: what does rough volatility do to the implied-vol surface?

Produces reports/rough_vol_results.png and reports/skew_term_structure.csv.

Story
-----
1. Rough (H=0.1) variance paths are visibly jagged vs a smoother (H=0.45) one.
2. At a short maturity, rough Bergomi generates a steep, curved implied-vol
   smile where Black-Scholes is flat by construction.
3. The at-the-money skew decays across maturities as a power law with exponent
   ~ H - 1/2 — the empirical signature that motivated rough volatility.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from roughvol.models import RoughBergomi  # noqa: E402
from roughvol.smile import smile_for_maturity, skew_term_structure  # noqa: E402
from roughvol.plotting import results_figure  # noqa: E402

# Model parameters (typical rough-Bergomi calibration ballpark).
S0, R = 100.0, 0.0
H_ROUGH, ETA, RHO, XI0 = 0.10, 1.9, -0.9, 0.04   # xi0=0.04 -> ~20% vol level


def rb_factory(H):
    def make(T):
        steps = max(50, int(200 * T))  # keep dt roughly constant across maturities
        return RoughBergomi(S0=S0, r=R, T=T, H=H, eta=ETA, rho=RHO, xi0=XI0, n_steps=steps)
    return make


def main():
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)

    print("[1/3] Sampling variance paths (rough vs smooth) ...")
    var_paths = {}
    for label, H in [(f"rough  H={H_ROUGH}", H_ROUGH), ("smooth H=0.45", 0.45)]:
        m = RoughBergomi(S0=S0, r=R, T=1.0, H=H, eta=ETA, rho=RHO, xi0=XI0, n_steps=300)
        _, v, t = m.simulate_paths(1, seed=7, antithetic=False)
        var_paths[label] = (t, v[0])

    print("[2/3] Building implied-vol smiles at T=0.1 ...")
    T_short = 0.1
    strikes = np.round(S0 * np.exp(np.linspace(-0.30, 0.30, 15)), 2)
    smiles = {
        f"rough Bergomi H={H_ROUGH}": smile_for_maturity(
            rb_factory(H_ROUGH), T_short, strikes, n_paths=300_000, seed=11
        ),
        "smooth H=0.45": smile_for_maturity(
            rb_factory(0.45), T_short, strikes, n_paths=300_000, seed=11
        ),
    }

    print("[3/3] Computing ATM skew term structure ...")
    maturities = np.array([0.05, 0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.0])
    skew_df, slope = skew_term_structure(
        rb_factory(H_ROUGH), maturities, n_paths=400_000, seed=21
    )
    skew_df.to_csv(reports / "skew_term_structure.csv")

    results_figure(var_paths, smiles, skew_df, slope, H=H_ROUGH,
                   savepath=str(reports / "rough_vol_results.png"))

    print("\n=== Rough-volatility results ===")
    print(f"Fitted ATM-skew power-law slope : {slope:+.3f}")
    print(f"Theoretical (H - 1/2)           : {H_ROUGH - 0.5:+.3f}")
    print("\nATM skew by maturity:")
    print(skew_df.round(4))
    print(f"\nSaved: {reports/'rough_vol_results.png'}")
    print(f"Saved: {reports/'skew_term_structure.csv'}")


if __name__ == "__main__":
    main()
