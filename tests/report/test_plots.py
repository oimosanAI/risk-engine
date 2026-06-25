"""Tests for report plotting (src/report/plots.py).

Structural tests only (no pixel comparison): assert functions run, return Axes,
and produce the expected artists/colors/annotations. Agg backend forced for
headless runs.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from matplotlib.colors import to_hex  # noqa: E402

from src.report.plots import (  # noqa: E402
    COLOR_ACCENT,
    COLOR_LOSS,
    COLOR_VAR,
    COMPARISON_SHADES,
    plot_stress_wealth,
    plot_var_comparison,
    plot_var_violations,
    plot_violation_clustering,
    save_figure,
)


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


def _dates(n, start="2020-01-01"):
    return pd.date_range(start, periods=n, freq="B")


def _returns_var(values, var_level=0.02):
    idx = _dates(len(values))
    return pd.Series(values, index=idx), pd.Series(var_level, index=idx)


_ALLOWED_LINE_COLORS = {
    to_hex(c).lower() for c in (COLOR_LOSS, COLOR_VAR, *COMPARISON_SHADES)
}


# ---------------------------------------------------------------------------
# 1 & 6. Axes return / ax handling.
# ---------------------------------------------------------------------------
def test_returns_axes():
    ret, var = _returns_var([-0.03, 0.01, -0.05, 0.0])
    ax = plot_var_violations(ret, var)
    assert isinstance(ax, plt.Axes)


def test_ax_none_creates_new():
    ret, var = _returns_var([-0.03, 0.01])
    ax = plot_var_violations(ret, var, ax=None)
    assert ax is not None


def test_passed_ax_is_returned():
    ret, var = _returns_var([-0.03, 0.01])
    fig, my_ax = plt.subplots()
    ax = plot_var_violations(ret, var, ax=my_ax)
    assert ax is my_ax


# ---------------------------------------------------------------------------
# 2. Violation marker count == manual count.
# ---------------------------------------------------------------------------
def test_violation_marker_count():
    # losses 0.03,−0.01,0.05,0,0.04,−0.02 vs var 0.02 -> violations at 0,2,4 = 3
    ret, var = _returns_var([-0.03, 0.01, -0.05, 0.0, -0.04, 0.02])
    ax = plot_var_violations(ret, var)
    offsets = ax.collections[0].get_offsets()
    assert len(offsets) == 3


# ---------------------------------------------------------------------------
# 3. Comparison line count + legend labels.
# ---------------------------------------------------------------------------
def test_comparison_line_count_and_legend():
    idx = _dates(10)
    ret = pd.Series(np.random.default_rng(0).standard_normal(10) * 0.01, index=idx)
    var_dict = {
        "historical": pd.Series(0.02, index=idx),
        "normal": pd.Series(0.025, index=idx),
        "t": pd.Series(0.03, index=idx),
    }
    ax = plot_var_comparison(ret, var_dict)
    assert len(ax.lines) == len(var_dict) + 1  # the faint loss line
    labels = {t.get_text() for t in ax.get_legend().get_texts()}
    assert labels == {"historical", "normal", "t", "loss"}


# ---------------------------------------------------------------------------
# 4. Stress annotations: trough always; recovery only if it recovers.
# ---------------------------------------------------------------------------
def test_stress_recovering_has_recovery_annotation():
    ret = pd.Series([-0.2, 0.1, 0.2], index=_dates(3))  # wealth 0.8,0.88,1.056
    ax = plot_stress_wealth(ret, "x")
    texts = [t.get_text() for t in ax.texts]
    assert any("trough" in t for t in texts)
    assert any("recovered" in t for t in texts)
    assert not any("no recovery" in t for t in texts)


def test_stress_non_recovering_has_no_recovery_note():
    ret = pd.Series([-0.2, -0.1, -0.05], index=_dates(3))  # never back to 1.0
    ax = plot_stress_wealth(ret, "x")
    texts = [t.get_text() for t in ax.texts]
    assert any("trough" in t for t in texts)
    assert any("no recovery" in t for t in texts)
    assert not any("recovered" in t for t in texts)


# ---------------------------------------------------------------------------
# 5. Clustering: total events + isolated vs clustered split.
# ---------------------------------------------------------------------------
def test_violation_clustering_split():
    # violations at 0,1,2 (3-run -> clustered) and 5 (isolated)
    ret, var = _returns_var([-0.03, -0.03, -0.03, 0.0, 0.0, -0.03, 0.0, 0.01])
    ax = plot_violation_clustering(ret, var)
    isolated = ax.collections[0].get_offsets()
    clustered = ax.collections[1].get_offsets()
    assert len(isolated) == 1
    assert len(clustered) == 3
    assert len(isolated) + len(clustered) == 4  # total violations


# ---------------------------------------------------------------------------
# 7. Grayscale enforcement (aesthetic in code).
# ---------------------------------------------------------------------------
def _assert_palette(ax):
    for line in ax.lines:
        assert to_hex(line.get_color()).lower() in _ALLOWED_LINE_COLORS
    accent = to_hex(COLOR_ACCENT).lower()
    for coll in ax.collections:
        if len(coll.get_offsets()) > 0:
            assert to_hex(coll.get_facecolor()[0]).lower() == accent
    # no default matplotlib cycle color (C0 = #1f77b4) on any line
    assert to_hex("#1f77b4").lower() not in _ALLOWED_LINE_COLORS


def test_accent_distinct_from_default_cycle():
    # The accent must be a deliberate choice OUTSIDE the default palette, so a
    # regression like COLOR_ACCENT="#d62728" (matplotlib's default red, C3) fails.
    cycle = {
        to_hex(c).lower()
        for c in matplotlib.rcParams["axes.prop_cycle"].by_key()["color"]
    }
    assert to_hex(COLOR_ACCENT).lower() not in cycle


def test_palette_var_violations():
    ret, var = _returns_var([-0.03, 0.01, -0.05, 0.0])
    _assert_palette(plot_var_violations(ret, var))


def test_palette_comparison():
    idx = _dates(8)
    ret = pd.Series(0.0, index=idx)
    var_dict = {"a": pd.Series(0.02, index=idx), "b": pd.Series(0.03, index=idx)}
    _assert_palette(plot_var_comparison(ret, var_dict))


def test_palette_stress_and_clustering():
    ret = pd.Series([-0.2, 0.1, 0.2], index=_dates(3))
    _assert_palette(plot_stress_wealth(ret, "x"))
    ret2, var2 = _returns_var([-0.03, -0.03, 0.0, -0.03])
    _assert_palette(plot_violation_clustering(ret2, var2))


# ---------------------------------------------------------------------------
# 8. save_figure: writes a non-empty file, creates parent dirs.
# ---------------------------------------------------------------------------
def test_save_figure_creates_dirs_and_file(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3])
    out = tmp_path / "nested" / "dir" / "fig.png"
    save_figure(fig, str(out))
    assert out.exists()
    assert out.stat().st_size > 0


# ---------------------------------------------------------------------------
# 9. Integration: render all four from real data, save four files.
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_integration_render_all_four(tmp_path):
    from src.data.loader import fetch_prices, prices_to_returns
    from src.var.historical import rolling_historical_var
    from src.var.parametric import rolling_parametric_var

    returns = prices_to_returns(
        fetch_prices(["SPY"], "2005-01-01", "2022-12-31"), method="simple"
    )["SPY"]
    rh = rolling_historical_var(returns, window=500, confidence=0.99)["var"]
    rn = rolling_parametric_var(returns, method="normal", window=500)["var"]
    rt = rolling_parametric_var(returns, method="t", window=500)["var"]

    axes = {
        "violations": plot_var_violations(returns.loc[rh.index], rh),
        "wealth": plot_stress_wealth(
            returns.loc["2008-09-01":"2009-03-31"], "gfc_2008"
        ),
        "comparison": plot_var_comparison(
            returns, {"historical": rh, "normal": rn, "t": rt}
        ),
        "clustering": plot_violation_clustering(returns.loc[rh.index], rh),
    }
    for name, ax in axes.items():
        out = tmp_path / f"{name}.png"
        save_figure(ax.figure, str(out))
        assert out.exists() and out.stat().st_size > 0
