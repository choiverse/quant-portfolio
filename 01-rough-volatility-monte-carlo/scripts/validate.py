"""Correctness gates for the Monte Carlo engine — run this before trusting a result.

Each gate compares a simulated quantity against an answer that is known
independently (a closed form, an exact moment, or a theoretical scaling law) and
reports PASS/FAIL against a stated tolerance.

  1. Volterra variance        Var(Y_t) -> t^{2H}              scheme correctness
  2. GBM vs Black-Scholes     MC price within ~3 std errors   pricer correctness
  3. Antithetic sampling      shrinks the SE vs crude MC      efficiency
  4. Control variate          shrinks the SE vs crude MC      efficiency
  5. RB martingale            E[S_T] = S0 e^{rT}              risk-neutral drift
  6. RB flat-vol limit        eta -> 0 collapses to BS        model correctness

The gate definitions live in ``roughvol.analysis.validation_gates`` so that this
script and ``scripts/make_figures.py`` cannot drift apart. Exits non-zero if any
gate fails, which is what makes it usable in CI.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from roughvol.analysis import validation_gates  # noqa: E402


def main() -> None:
    print("Running validation gates...\n")
    gates = validation_gates()

    for name, row in gates.iterrows():
        verdict = "PASS" if row["passed"] else "FAIL"
        comparator = ">" if "reduction" in row["units"] else "<"
        print(f"  [{verdict}] {name}: {row['statistic']:.4g} {row['units']} "
              f"(tolerance {comparator} {row['tolerance']:g})")

    n_pass = int(gates["passed"].sum())
    print(f"\n{n_pass}/{len(gates)} checks passed.")
    sys.exit(0 if n_pass == len(gates) else 1)


if __name__ == "__main__":
    main()
