"""Shared bootstrap for the scripts: import path, default arguments, data loading."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Report text contains arrows and maths symbols; a non-UTF-8 default console
# (cp949 on a Korean Windows install, cp1252 elsewhere) mangles or rejects them.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
TABLES = REPORTS / "tables"

# The third project to use the same Kaggle panel. Look locally first, then in
# project 02's data directory, so a checkout that already has the 28 MB file
# does not need a third copy of it.
_LOCAL_DATA = ROOT / "data" / "all_stocks_5yr.csv"
_SHARED_DATA = ROOT.parent / "02-systematic-equity-backtest" / "data" / "all_stocks_5yr.csv"
DEFAULT_DATA = _LOCAL_DATA if _LOCAL_DATA.exists() else _SHARED_DATA


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--data", default=str(DEFAULT_DATA),
                   help="path to the long-format OHLCV CSV")
    p.add_argument("--outdir", default=None,
                   help="write reports here instead of reports/ — use this for "
                        "smoke runs on sample data so they cannot overwrite the "
                        "committed results generated from the full dataset")
    p.add_argument("--horizon", type=int, default=5,
                   help="label horizon in trading days (default 5)")
    return p


def ensure_dirs(outdir: str | None = None) -> tuple[Path, Path, Path]:
    """Create the report directories and return ``(reports, figures, tables)``.

    The paths are returned rather than read from module globals so that
    ``--outdir`` actually takes effect: running a stage against the small
    committed sample would otherwise silently overwrite figures produced from
    the full panel, and the README would then be documenting numbers that no
    longer match the images beside them.
    """
    reports = Path(outdir) if outdir is not None else REPORTS
    figures, tables = reports / "figures", reports / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    return reports, figures, tables


def load_matrices(path: str):
    """Load the raw file and return the aligned price/volume/high/low matrices."""
    from mlalpha import data as mdata

    return mdata.load_panel(path)


def saved(path: Path) -> None:
    try:
        shown = path.relative_to(ROOT)
    except ValueError:      # --outdir pointed outside the project tree
        shown = path
    print(f"      saved  {shown}")
