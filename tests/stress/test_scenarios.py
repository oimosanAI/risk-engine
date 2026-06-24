"""Tests for historical stress scenarios (src/stress/scenarios.py).

Unit tests use synthetic returns with explicit dates so calendar-day recovery
counts are hand-verifiable. The integration test (real SPY data) is marked and
skipped by default.
"""

import logging

import numpy as np
import pandas as pd
import pytest

from src.stress.scenarios import (
    SCENARIOS,
    ScenarioResult,
    run_preset_scenarios,
    run_scenario,
)


def _series(values, dates):
    return pd.Series(np.asarray(values, dtype=float), index=pd.DatetimeIndex(dates))


_D3 = ["2020-01-01", "2020-01-02", "2020-01-05"]  # note the weekend gap


# ---------------------------------------------------------------------------
# 1. Cumulative return.
# ---------------------------------------------------------------------------
def test_cumulative_return_hand():
    ret = _series([0.01, -0.02, 0.03], _D3)
    res = run_scenario(ret, "x", "2020-01-01", "2020-01-05")
    # prod(1.01, 0.98, 1.03) - 1 = 0.019494
    assert res.cumulative_return == pytest.approx(0.019494, abs=1e-10)
    assert res.n_obs == 3


# ---------------------------------------------------------------------------
# 2. Max drawdown (incl. MDD=0 -> recovery_days None per override).
# ---------------------------------------------------------------------------
def test_max_drawdown_hand():
    ret = _series([0.10, -0.20, 0.05], _D3)
    res = run_scenario(ret, "x", "2020-01-01", "2020-01-05")
    # wealth [1.1, 0.88, 0.924]; peak floored at 1.1 -> MDD = (1.1-0.88)/1.1 = 0.2
    assert res.max_drawdown == pytest.approx(0.20, abs=1e-10)


def test_max_drawdown_zero_when_all_positive():
    ret = _series([0.01, 0.02, 0.03], _D3)
    res = run_scenario(ret, "x", "2020-01-01", "2020-01-05")
    assert res.max_drawdown == 0.0
    assert res.recovery_days is None  # no drawdown -> recovery not applicable


# ---------------------------------------------------------------------------
# 3. Recovery period.
# ---------------------------------------------------------------------------
def test_recovery_within_window_calendar_days():
    ret = _series([-0.2, 0.1, 0.2], _D3)
    # wealth [0.8, 0.88, 1.056]; trough at 2020-01-01; recovers (>=1.0) at 2020-01-05
    res = run_scenario(ret, "x", "2020-01-01", "2020-01-05")
    assert res.recovery_days == 4  # calendar days 01-01 -> 01-05


def test_no_recovery_returns_none():
    ret = _series([-0.2, -0.1, -0.05], _D3)
    res = run_scenario(ret, "x", "2020-01-01", "2020-01-05")
    assert res.recovery_days is None


def test_recovery_to_one_not_within_scenario_peak():
    # wealth [1.2, 0.72, 1.152]: recovers to >=1.0 but never back to the 1.2 high.
    ret = _series([0.2, -0.4, 0.6], _D3)
    res = run_scenario(ret, "x", "2020-01-01", "2020-01-05")
    assert res.max_drawdown == pytest.approx(0.40, abs=1e-10)  # (1.2-0.72)/1.2
    assert res.recovery_days == 3  # trough 2020-01-02 -> recovery 2020-01-05


# ---------------------------------------------------------------------------
# 4. VaR violation rate (reuses var_backtest.violations).
# ---------------------------------------------------------------------------
def test_var_violation_rate():
    dates = pd.date_range("2020-01-01", periods=4, freq="B")
    ret = pd.Series([-0.03, 0.01, -0.05, 0.0], index=dates)
    var = pd.Series(0.02, index=dates)
    res = run_scenario(
        ret, "x", "2020-01-01", "2020-01-10", var_series=var, var_method="historical"
    )
    # losses 0.03>0.02 T, -0.01 F, 0.05>0.02 T, 0 F -> 2/4 = 0.5
    assert res.var_violation_rate == pytest.approx(0.5)
    assert res.var_method == "historical"


def test_var_none_when_no_var_series():
    ret = _series([0.01, -0.02, 0.03], _D3)
    res = run_scenario(ret, "x", "2020-01-01", "2020-01-05")
    assert res.var_violation_rate is None
    assert res.var_method is None


# ---------------------------------------------------------------------------
# 5. NaN handling.
# ---------------------------------------------------------------------------
def test_nan_returns_dropped_and_logged(caplog):
    dates = pd.date_range("2020-01-01", periods=4, freq="B")
    ret = pd.Series([0.01, np.nan, -0.02, 0.03], index=dates)
    with caplog.at_level(logging.WARNING):
        res = run_scenario(ret, "x", "2020-01-01", "2020-01-10")
    assert res.n_obs == 3  # one NaN dropped
    assert "nan" in caplog.text.lower()


# ---------------------------------------------------------------------------
# 6. Empty window raises.
# ---------------------------------------------------------------------------
def test_empty_window_raises():
    ret = _series([0.01, -0.02, 0.03], _D3)  # all in 2020-01
    with pytest.raises(ValueError):
        run_scenario(ret, "future", "2021-01-01", "2021-12-31")


def test_all_nan_window_raises():
    # Window has rows but they are all NaN -> empty after cleaning -> raise.
    dates = pd.date_range("2020-01-01", periods=3, freq="B")
    ret = pd.Series([np.nan, np.nan, np.nan], index=dates)
    with pytest.raises(ValueError):
        run_scenario(ret, "x", "2020-01-01", "2020-01-31")


# ---------------------------------------------------------------------------
# 7. No-overlap var_series -> None + warning.
# ---------------------------------------------------------------------------
def test_no_overlap_var_series_warns_and_none(caplog):
    ret = pd.Series(
        [0.01, -0.02, 0.03], index=pd.date_range("2020-01-01", periods=3, freq="B")
    )
    var = pd.Series(
        0.02, index=pd.date_range("2019-01-01", periods=3, freq="B")  # disjoint
    )
    with caplog.at_level(logging.WARNING):
        res = run_scenario(
            ret,
            "x",
            "2020-01-01",
            "2020-01-31",
            var_series=var,
            var_method="historical",
        )
    assert res.var_violation_rate is None
    assert res.var_method is None
    assert "overlap" in caplog.text.lower()


# ---------------------------------------------------------------------------
# 8. run_preset_scenarios.
# ---------------------------------------------------------------------------
def _long_returns(start, end):
    idx = pd.date_range(start, end, freq="B")
    return pd.Series(0.001, index=idx)


def test_preset_both_scenarios_run():
    ret = _long_returns("2008-01-01", "2020-12-31")
    out = run_preset_scenarios(ret)
    assert set(out.keys()) == set(SCENARIOS.keys())
    assert all(isinstance(v, ScenarioResult) for v in out.values())


def test_preset_skips_scenario_without_data(caplog):
    ret = _long_returns("2020-01-01", "2020-12-31")  # no 2008 data
    with caplog.at_level(logging.WARNING):
        out = run_preset_scenarios(ret)
    assert "covid_2020" in out
    assert "gfc_2008" not in out  # silently skipped
    assert "gfc_2008" in caplog.text


def test_preset_custom_scenarios_override():
    ret = _long_returns("2020-01-01", "2020-12-31")
    custom = {"mini": ("2020-03-01", "2020-03-31")}
    out = run_preset_scenarios(ret, scenarios=custom)
    assert set(out.keys()) == {"mini"}


def test_preset_skips_all_nan_window(caplog):
    # Uniform leniency: an all-NaN window (survives the empty check) is skipped,
    # not raised, and the other scenario still returns its result.
    ret = _long_returns("2020-01-01", "2020-12-31")
    ret.loc["2020-03-01":"2020-03-31"] = np.nan
    custom = {
        "bad": ("2020-03-01", "2020-03-31"),  # all-NaN -> skipped
        "good": ("2020-05-01", "2020-05-31"),  # has data -> result
    }
    with caplog.at_level(logging.WARNING):
        out = run_preset_scenarios(ret, scenarios=custom)
    assert "good" in out
    assert "bad" not in out
    assert "bad" in caplog.text


# ---------------------------------------------------------------------------
# 9. Integration (real SPY data; skipped by default).
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_integration_spy_scenarios():
    from src.data.loader import fetch_prices, prices_to_returns
    from src.var.historical import rolling_historical_var

    prices = fetch_prices(["SPY"], "2000-01-01", "2023-12-31")
    returns = prices_to_returns(prices, method="simple")["SPY"]
    roll = rolling_historical_var(returns, window=500, confidence=0.99)

    out = run_preset_scenarios(returns, var_series=roll["var"], var_method="historical")
    for name, r in out.items():
        print(
            f"\n[{name}] n_obs={r.n_obs} cum={r.cumulative_return:.4f} "
            f"mdd={r.max_drawdown:.4f} recovery_days={r.recovery_days} "
            f"var_viol={r.var_violation_rate}"
        )

    gfc = out["gfc_2008"]
    covid = out["covid_2020"]
    # Robust, data-confirmed bounds (observed: GFC cum=-0.370 mdd=0.464 viol=0.110
    # recovery=None; COVID cum=-0.032 mdd=0.337 viol=0.106 recovery=77 days).
    # NOTE: the spec's literal "GFC cum < -0.40" and "COVID cum < -0.20" are wrong
    # (cum=-0.37 and -0.03); the latter conflated max_drawdown with window return.
    # COVID cum is reported, not gated. COVID recovery is now asserted (confirmed).
    assert gfc.cumulative_return < -0.20
    assert gfc.max_drawdown > 0.40
    assert gfc.var_violation_rate is not None and gfc.var_violation_rate > 0.05
    assert covid.max_drawdown > 0.20
    assert covid.recovery_days is not None  # SPY recovered to Feb-1 level by June
