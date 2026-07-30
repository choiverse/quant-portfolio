"""Correctness gates for the factor model and the learners — run before trusting a result.

Each gate compares an implementation against an answer known independently (an
exact algebraic identity, a limiting distribution from a theorem, a function
with written-down parameters, a panel with a planted signal or a planted null)
and reports PASS/FAIL against a stated tolerance.

   1. PCA spectrum        recovers a constructed covariance   exact algebra
   2. Marchenko-Pastur    empirical eigenvalues vs the law    factor selection
   3. Factor count        0 on noise, k on a planted k        both directions
   4. Residual neutrality manufactures nothing on a null      the subtle one
   5. Ridge identity      shrinkage on an orthonormal design  exact algebra
   6. Ridge recovery      known coefficients, and OLS at a=0  estimator check
   7. Tree split          binned search vs brute force        no accuracy loss
   8. Boosting            learns Friedman #1 nonlinearity     capacity check
   9. Purging             no label overlap, with a control    split integrity
  10. Look-ahead          rewriting the last bar cannot
                          change earlier P&L                  pipeline integrity
  11. Signal recovery     finds a planted cross-sectional
                          signal                              pipeline power

No market data is needed — every gate builds its own — so this runs in CI and
fails the build if the maths breaks. The definitions live in
``mlalpha.validation.validation_gates`` so this script and the write-up cannot
drift apart.

    python scripts/validate.py            # the committed settings
    python scripts/validate.py --quick    # smaller Monte Carlos, seconds
"""

from __future__ import annotations

import argparse
import sys
import time

from _common import ROOT  # noqa: F401  — sets up sys.path for the import below
from mlalpha.validation import validation_gates  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true",
                   help="shrink the Monte Carlo gates for a fast check")
    args = p.parse_args()

    print("Running validation gates...\n")
    t0 = time.perf_counter()
    gates = validation_gates(quick=args.quick)

    width = max(len(str(name)) for name in gates.index)
    for name, row in gates.iterrows():
        verdict = "PASS" if row["passed"] else "FAIL"
        print(f"  [{verdict}] {str(name):<{width}}  "
              f"{row['statistic']:.4g} {row['units']} "
              f"(tolerance {row['comparator']} {row['tolerance']:g})")
        if row["detail"]:
            print(f"         {' ' * width}  {row['detail']}")

    n_pass = int(gates["passed"].sum())
    print(f"\n{n_pass}/{len(gates)} checks passed "
          f"in {time.perf_counter() - t0:.1f}s.")
    sys.exit(0 if n_pass == len(gates) else 1)


if __name__ == "__main__":
    main()
