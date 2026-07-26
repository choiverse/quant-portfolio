"""Correctness gates for the econometrics — run this before trusting a result.

Each gate compares an estimator against an answer that is known independently
(a published critical value, a simulated process with written-down parameters,
or an exact identity) and reports PASS/FAIL against a stated tolerance.

  1. ADF critical values     match MacKinnon (2010)        test calibration
  2. ADF size                rejects at 5% on random walks  no false structure
  3. ADF power               rejects on stationary AR(1)    the test works
  4. Engle-Granger           recovers a known hedge ratio   step 1 correctness
  5. OU half-life            recovers a known AR(1) phi     spread dynamics
  6. GARCH(1,1)              recovers known persistence     MLE correctness
  7. HMM                     recovers simulated states      EM correctness
  8. Look-ahead              rewriting the last bar cannot
                             change earlier P&L             pipeline integrity

No market data is needed — every gate builds its own — so this runs in CI and
fails the build if the maths breaks. The definitions live in
``statarb.validation.validation_gates`` so this script and the write-up cannot
drift apart.

    python scripts/validate.py            # the committed settings
    python scripts/validate.py --quick    # smaller Monte Carlos, seconds
"""

from __future__ import annotations

import argparse
import sys
import time

from _common import ROOT  # noqa: F401  — sets up sys.path for the import below
from statarb.validation import validation_gates  # noqa: E402


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
