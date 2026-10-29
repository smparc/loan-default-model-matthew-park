"""Shared chart style and figure saving.

Colors come from a reference palette checked for colorblind safety: categorical slots 1-3
(blue, orange, aqua) stay distinguishable for any pair, gray marks context, and gridlines
are solid hairlines. Where the rule-based baseline and the model appear together, the
model is blue and the baseline is gray.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, PercentFormatter

from .data import REPO_ROOT

FIGURES_DIR = REPO_ROOT / "figures"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
MODEL_COLOR, RULES_COLOR = BLUE, MUTED


def apply_style() -> None:
    """Quiet chart chrome: light surface, hairline solid grid, no top/right spines."""
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "figure.dpi": 110,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 9.5,
            "text.color": INK,
            "axes.titlesize": 10.5,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlecolor": INK,
            "axes.titlepad": 10,
            "axes.labelcolor": INK_SECONDARY,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "grid.linestyle": "-",
            "xtick.color": AXIS,
            "ytick.color": AXIS,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "legend.frameon": False,
            "lines.linewidth": 2,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
        }
    )


def save(fig: plt.Figure, name: str) -> Path:
    """Save to figures/<name>.png so the README can reference it."""
    FIGURES_DIR.mkdir(exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    return path


def percent_axis(ax: plt.Axes, axis: str = "y") -> None:
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(PercentFormatter(xmax=1, decimals=0))


def dollar_axis(ax: plt.Axes, axis: str = "x") -> None:
    fmt = FuncFormatter(lambda v, _: f"${v / 1000:g}K" if abs(v) >= 1000 else f"${v:,.0f}")
    (ax.xaxis if axis == "x" else ax.yaxis).set_major_formatter(fmt)


def _pct(value: float) -> str:
    return f"{value:.0%}" if value >= 0.1 else f"{value:.1%}"


def rate_bars(ax: plt.Axes, table, labels, color: str = BLUE, width: float = 0.3) -> None:
    """One thin bar per group, its 95% CI as a whisker (if given) and the rate at the tip."""
    x = np.arange(len(table))
    rate = table["rate"].to_numpy(dtype=float)
    ax.bar(x, rate, width=width, color=color, zorder=2)
    tops = rate
    if {"ci_low", "ci_high"} <= set(table.columns):
        low, high = table["ci_low"].to_numpy(dtype=float), table["ci_high"].to_numpy(dtype=float)
        ax.errorbar(x, rate, yerr=[rate - low, high - rate], fmt="none", ecolor=INK_SECONDARY, elinewidth=1, zorder=3)
        tops = high
    for xi, value, top in zip(x, rate, tops):
        ax.text(xi, top + 0.015, _pct(value), ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(x, labels)
    ax.tick_params(axis="x", length=0)
    percent_axis(ax)


def rate_line(ax: plt.Axes, table, x: str = "x_median", color: str = BLUE) -> None:
    """A rate across ordered bins: 2px line, ringed markers, 95% CI as a light wash."""
    ax.fill_between(table[x], table["ci_low"], table["ci_high"], color=color, alpha=0.10, linewidth=0, zorder=1)
    ax.plot(table[x], table["rate"], color=color, marker="o", markersize=6, markeredgecolor=SURFACE,
            markeredgewidth=1.5, zorder=3)
    percent_axis(ax)
