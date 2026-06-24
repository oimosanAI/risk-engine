"""Historical stress scenarios: replay a market crisis window and report what it
did to a return series.

Per scenario we report the cumulative return, the maximum drawdown, the recovery
period, and (optionally) the VaR violation rate over the window.

Conventions (see CLAUDE.md):
  * Cumulative return is ``prod(1+r) - 1`` over the window (NOT a sum).
  * Max drawdown is a POSITIVE magnitude on a wealth index that starts at 1.0
    (the pre-scenario capital): the running peak is floored at 1.0, so a purely
    declining window measures its drawdown from the pre-scenario level.
  * Recovery means the wealth index returns to >= 1.0 (the pre-scenario level),
    NOT back to a within-scenario high. ``recovery_days`` is CALENDAR days from
    the drawdown trough to the recovery date (differs from trading days: a ~5
    month recovery is ~150 calendar but ~105 trading days). It is None if the
    window never recovers, and None when there is no drawdown at all (MDD=0):
    recovery is not applicable, so None is the honest signal (not 0).
  * VaR violation rate reuses ``var_backtest.violations`` over the overlapping,
    non-NaN subset of the window; its denominator may be SMALLER than ``n_obs``
    because a rolling VaR series does not cover the earliest window dates.
  * NaNs in the window's returns are dropped (count logged); ``n_obs`` reflects
    the reduction. Empirical values are reported as-is — never rounded or clipped.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.backtest.var_backtest import violations
from src.var._utils import _clean

logger = logging.getLogger(__name__)

SCENARIOS: dict[str, tuple[str, str]] = {
    "gfc_2008": ("2008-09-01", "2009-03-31"),
    "covid_2020": ("2020-02-01", "2020-06-30"),
}


@dataclass
class ScenarioResult:
    """Result of replaying one stress window.

    ``start`` and ``end`` are the actual first/last data dates within the
    requested window (e.g. a Monday if the requested start fell on a weekend),
    not the requested bounds.
    """

    name: str
    start: pd.Timestamp
    end: pd.Timestamp
    n_obs: int
    cumulative_return: float  # prod(1+r) - 1 over the scenario window
    max_drawdown: float  # positive magnitude, e.g. 0.55 = 55% drawdown
    recovery_days: int | None  # None if not recovered within the window
    var_violation_rate: float | None  # None if no var_series provided / no overlap
    var_method: str | None  # e.g. "historical"; None if no var_series / no overlap


def _max_drawdown(wealth: pd.Series) -> tuple[float, pd.Timestamp]:
    """Return (max drawdown magnitude, trough date) on a wealth index whose
    running peak is floored at 1.0 (the pre-scenario capital)."""
    values = wealth.to_numpy()
    peak = np.maximum(np.maximum.accumulate(values), 1.0)
    drawdown = pd.Series((peak - values) / peak, index=wealth.index)
    return float(drawdown.max()), drawdown.idxmax()


def _recovery_days(
    wealth: pd.Series, trough_date: pd.Timestamp, mdd: float
) -> int | None:
    """Calendar days from the trough to the first later date where the wealth
    index is back to >= 1.0. None if it never recovers, or if there is no
    drawdown (MDD=0 -> recovery not applicable)."""
    if mdd == 0.0:
        return None
    post = wealth.loc[wealth.index > trough_date]
    recovered = post[post >= 1.0]
    if recovered.empty:
        return None
    return int((recovered.index[0] - trough_date).days)


def _violation_rate(
    window_returns: pd.Series, var_series: pd.Series, name: str
) -> float | None:
    """VaR violation rate over the overlapping, non-NaN subset of the window.

    Returns None (and logs a warning) when ``var_series`` has no overlap with the
    window after alignment.
    """
    var_window = var_series.reindex(window_returns.index)
    pair = pd.DataFrame({"r": window_returns, "v": var_window}).dropna()
    if pair.empty:
        logger.warning(
            "scenario %s: var_series has no overlap with the window; "
            "var_violation_rate=None",
            name,
        )
        return None
    return float(violations(pair["r"], pair["v"]).mean())


def run_scenario(
    returns: pd.Series,
    name: str,
    start: str,
    end: str,
    var_series: pd.Series | None = None,
    var_method: str | None = None,
) -> ScenarioResult:
    """Replay the window ``[start, end]`` (inclusive) of ``returns``.

    Raises ValueError if the window contains no data. NaN returns in the window
    are dropped (logged). See the module docstring for the recovery and VaR
    violation-rate conventions.
    """
    window = returns.loc[start:end]
    if window.empty:
        raise ValueError(f"scenario {name!r}: no return data in window {start}..{end}")

    window = _clean(window)
    if window.empty:
        raise ValueError(
            f"scenario {name!r}: window {start}..{end} is all-NaN after cleaning"
        )

    n_obs = len(window)
    cumulative_return = float((1.0 + window).prod() - 1.0)

    wealth = (1.0 + window).cumprod()
    mdd, trough_date = _max_drawdown(wealth)
    recovery_days = _recovery_days(wealth, trough_date, mdd)

    if var_series is None:
        var_rate: float | None = None
        resolved_method: str | None = None
    else:
        var_rate = _violation_rate(window, var_series, name)
        resolved_method = var_method if var_rate is not None else None

    return ScenarioResult(
        name=name,
        start=window.index[0],
        end=window.index[-1],
        n_obs=n_obs,
        cumulative_return=cumulative_return,
        max_drawdown=mdd,
        recovery_days=recovery_days,
        var_violation_rate=var_rate,
        var_method=resolved_method,
    )


def run_preset_scenarios(
    returns: pd.Series,
    var_series: pd.Series | None = None,
    var_method: str | None = None,
    scenarios: dict[str, tuple[str, str]] | None = None,
) -> dict[str, ScenarioResult]:
    """Run every scenario in ``scenarios`` (default ``SCENARIOS``).

    Uniformly lenient: a scenario whose window has no usable data — empty, or
    all-NaN after cleaning — is skipped (logged), not raised. This is the inverse
    of a direct ``run_scenario`` call, which is strict and raises.
    """
    scenarios = scenarios if scenarios is not None else SCENARIOS
    results: dict[str, ScenarioResult] = {}
    for name, (start, end) in scenarios.items():
        try:
            results[name] = run_scenario(
                returns, name, start, end, var_series=var_series, var_method=var_method
            )
        except ValueError as exc:
            logger.warning("scenario %s: skipping (%s)", name, exc)
    return results
