"""Monochrome, institutional-style plots for the risk engine outputs.

Aesthetic (non-negotiable): a grayscale base palette with exactly ONE accent
colour, used only for violation markers. Thin lines, a subtle grid, top/right
spines removed, title + axis labels, and a legend wherever more than one series
is shown. The palette and styling live in ONE place (the constants below and
``_style_axes``) and every plot function applies them.

Layering: plotting functions are PURE — they build and return a ``plt.Axes`` and
never touch the filesystem; ``save_figure`` is the only function that does I/O.
This module is also free of scipy: it imports the single ``violations`` source of
truth from ``var._utils`` (not from ``backtest.var_backtest``, which pulls scipy).

Where two series are drawn together they are aligned leniently (intersect index
+ drop NaN) and the overlap is plotted — plotting is exploratory and never
raises on a mismatch, unlike ``backtest_var`` which is strict. Values are plotted
raw: no smoothing, no clipping, no axis truncation that would hide extreme losses.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.stress.scenarios import _max_drawdown
from src.var._utils import _clean, violations

# --- palette (grayscale + single accent) -----------------------------------
COLOR_LOSS = "#999999"  # realized loss series (light gray)
COLOR_VAR = "#1a1a1a"  # VaR / wealth line (near-black)
COLOR_GRID = "#dddddd"  # very light grid
COLOR_ACCENT = "#c0392b"  # the ONLY accent: violation markers (restrained red)

# Distinguish overlaid series under the grayscale constraint with shades AND
# linestyle, so the comparison plot is readable without colour.
COMPARISON_SHADES = ["#333333", "#777777", "#aaaaaa"]
COMPARISON_STYLES = ["-", "--", ":"]

LINEWIDTH = 0.9
GRID_ALPHA = 0.3
FIGSIZE = (10, 5)


def _new_ax(ax: plt.Axes | None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=FIGSIZE)
    return ax


def _style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color=COLOR_GRID, alpha=GRID_ALPHA, linewidth=0.5)


def _align(a: pd.Series, b: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Lenient alignment for plotting: shared index, NaN pairs dropped, sorted."""
    common = a.index.intersection(b.index).sort_values()
    a2, b2 = a.reindex(common), b.reindex(common)
    mask = a2.notna() & b2.notna()
    return a2[mask], b2[mask]


def _violation_runs(viol: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Split a boolean violation series into (isolated, clustered) boolean masks.

    A maximal run of consecutive-observation violations of length 1 is isolated;
    length >= 2 is clustered (the dependence Christoffersen's test detects).
    """
    v = viol.to_numpy()
    isolated = np.zeros(len(v), dtype=bool)
    clustered = np.zeros(len(v), dtype=bool)
    i, n = 0, len(v)
    while i < n:
        if not v[i]:
            i += 1
            continue
        j = i
        while j < n and v[j]:
            j += 1
        if j - i >= 2:
            clustered[i:j] = True
        else:
            isolated[i] = True
        i = j
    return isolated, clustered


def plot_var_violations(
    returns: pd.Series,
    var_series: pd.Series,
    confidence: float = 0.99,
    title: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot losses (-returns), the VaR line, and mark violations (loss > var)."""
    ax = _new_ax(ax)
    r, v = _align(returns, var_series)
    loss = -r

    ax.plot(
        loss.index,
        loss.to_numpy(),
        color=COLOR_LOSS,
        linewidth=LINEWIDTH,
        alpha=0.8,
        label="loss",
    )
    ax.plot(
        v.index,
        v.to_numpy(),
        color=COLOR_VAR,
        linewidth=LINEWIDTH,
        label=f"VaR {confidence:.0%}",
    )

    viol = violations(r, v)
    vdates = loss.index[viol.to_numpy()]
    ax.scatter(
        vdates,
        loss.loc[vdates].to_numpy(),
        color=COLOR_ACCENT,
        s=18,
        zorder=3,
        label="violation",
    )

    ax.set_title(title or "VaR violations")
    ax.set_xlabel("date")
    ax.set_ylabel("loss (return units)")
    ax.legend(frameon=False)
    _style_axes(ax)
    return ax


def plot_stress_wealth(
    returns: pd.Series,
    name: str,
    title: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Wealth index (1+r).cumprod() with trough and recovery annotations.

    The trough uses the same ``_max_drawdown`` as the stress module, so the plot
    matches ``ScenarioResult``. Recovery means wealth back to >= 1.0 (the
    pre-scenario level); if it never recovers in the window a note is shown.
    """
    ax = _new_ax(ax)
    r = _clean(returns)
    wealth = (1.0 + r).cumprod()

    ax.plot(
        wealth.index,
        wealth.to_numpy(),
        color=COLOR_VAR,
        linewidth=LINEWIDTH,
        label="wealth index",
    )
    ax.axhline(1.0, color=COLOR_LOSS, linewidth=0.8, linestyle="--")

    mdd, trough = _max_drawdown(wealth)
    ax.annotate(
        f"trough -{mdd:.1%}",
        xy=(trough, wealth.loc[trough]),
        xytext=(0, -24),
        textcoords="offset points",
        color=COLOR_VAR,
        fontsize=8,
        ha="center",
        arrowprops=dict(arrowstyle="->", color=COLOR_VAR, lw=0.7),
    )

    post = wealth.loc[wealth.index > trough]
    recovered = post[post >= 1.0]
    if mdd > 0.0 and not recovered.empty:
        rd = recovered.index[0]
        ax.annotate(
            "recovered",
            xy=(rd, wealth.loc[rd]),
            xytext=(0, 16),
            textcoords="offset points",
            color=COLOR_VAR,
            fontsize=8,
            ha="center",
            arrowprops=dict(arrowstyle="->", color=COLOR_VAR, lw=0.7),
        )
    else:
        ax.text(
            0.5,
            0.9,
            "no recovery in window",
            transform=ax.transAxes,
            ha="center",
            color=COLOR_VAR,
            fontsize=8,
        )

    ax.set_title(title or f"Stress wealth index: {name}")
    ax.set_xlabel("date")
    ax.set_ylabel("wealth (start = 1.0)")
    _style_axes(ax)
    return ax


def plot_var_comparison(
    returns: pd.Series,
    var_dict: dict[str, pd.Series],
    title: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Overlay multiple rolling VaR series (grayscale shade + linestyle each),
    over a faint loss line for context. Legend required."""
    ax = _new_ax(ax)
    loss = -_clean(returns)
    ax.plot(
        loss.index,
        loss.to_numpy(),
        color=COLOR_LOSS,
        linewidth=0.7,
        alpha=0.4,
        label="loss",
    )

    for i, (label, series) in enumerate(var_dict.items()):
        shade = COMPARISON_SHADES[i % len(COMPARISON_SHADES)]
        style = COMPARISON_STYLES[i % len(COMPARISON_STYLES)]
        s = series.dropna()
        ax.plot(
            s.index,
            s.to_numpy(),
            color=shade,
            linestyle=style,
            linewidth=LINEWIDTH,
            label=label,
        )

    ax.set_title(title or "VaR method comparison")
    ax.set_xlabel("date")
    ax.set_ylabel("VaR (loss units)")
    ax.legend(frameon=False)
    _style_axes(ax)
    return ax


def plot_violation_clustering(
    returns: pd.Series,
    var_series: pd.Series,
    title: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Rug plot of violation dates, distinguishing consecutive-run clusters
    (length >= 2) from isolated violations."""
    ax = _new_ax(ax)
    r, v = _align(returns, var_series)
    viol = violations(r, v)
    isolated, clustered = _violation_runs(viol)
    idx = viol.index

    iso_dates = idx[isolated]
    clu_dates = idx[clustered]
    ax.scatter(
        iso_dates,
        np.zeros(len(iso_dates)),
        color=COLOR_ACCENT,
        marker="o",
        s=14,
        alpha=0.8,
        label="isolated",
    )
    ax.scatter(
        clu_dates,
        np.zeros(len(clu_dates)),
        color=COLOR_ACCENT,
        marker="s",
        s=36,
        label="clustered (>=2)",
    )

    ax.set_yticks([])
    ax.set_title(title or "VaR violation clustering")
    ax.set_xlabel("date")
    ax.legend(frameon=False)
    _style_axes(ax)
    return ax


def save_figure(fig: plt.Figure, path: str, dpi: int = 150) -> None:
    """Save ``fig`` to ``path`` (creating parent dirs). The only I/O here."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
