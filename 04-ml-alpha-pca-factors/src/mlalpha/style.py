"""One shared matplotlib style so every figure in the project reads as a set.

The categorical hues are a fixed, colorblind-validated order (checked for
CVD separation and a normal-vision floor against the light chart surface);
series are assigned slots in order and never cycled. Sequential magnitude
uses a single blue hue, and polarity (profit/loss, over/under) uses the
blue-red diverging pair with a neutral gray midpoint.
"""

from __future__ import annotations

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

# --- Categorical slots (assign in order, never cycle) ---------------------
SERIES = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#4a3aa7",  # 6 violet
]

# --- Chart chrome & ink ---------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# --- Status (reserved; never used as a series color) ----------------------
GOOD = "#0ca30c"
CRITICAL = "#d03b3b"

# --- Ramps ----------------------------------------------------------------
# Sequential: one hue, light -> dark. For magnitude only.
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#0d366b"]
)
# Diverging: two poles + neutral gray midpoint. For signed quantities only.
DIV_BLUE_RED = LinearSegmentedColormap.from_list(
    "div_blue_red", ["#0d366b", "#3987e5", "#cde2fb", "#f0efec", "#f5b0b0", "#d03b3b", "#7d1f1f"]
)


def use_style() -> None:
    """Apply the project's rcParams. Call once at the top of a plotting script."""
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
            "font.size": 9,
            "axes.titlesize": 10.5,
            "axes.titleweight": "bold",
            "axes.titlecolor": INK,
            "axes.labelsize": 9,
            "axes.labelcolor": INK_SECONDARY,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.prop_cycle": mpl.cycler(color=SERIES),
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "lines.linewidth": 2.0,
            "lines.markersize": 4.5,
            "figure.dpi": 130,
            "savefig.dpi": 130,
            "savefig.bbox": "tight",
        }
    )


def annotate_source(fig, text: str) -> None:
    """Footer credit line — every figure states its data source and horizon."""
    fig.text(0.005, -0.012, text, fontsize=7.5, color=INK_MUTED, ha="left", va="top")
