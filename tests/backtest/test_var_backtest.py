"""Tests for VaR backtesting (src/backtest/var_backtest.py).

Hand fixtures recompute the LR statistics from the formula directly. The
integration test (real market data) is marked and skipped by default.
"""

import logging
import math

import numpy as np
import pandas as pd
import pytest

from src.backtest.var_backtest import (
    BacktestResult,
    _basel_zone,
    _lr_ind,
    _lr_pof,
    _transition_counts,
    backtest_var,
    violations,
)


def _dates(n: int, start: str = "2010-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="B")


def _make(n: int, viol_positions, var_level: float = 0.02):
    """Constant positive VaR; a violation is a -0.03 loss (> var) at a position."""
    idx = _dates(n)
    var = pd.Series(var_level, index=idx)
    ret = pd.Series(0.0, index=idx)  # loss 0 < var -> no violation
    for p in viol_positions:
        ret.iloc[p] = -0.03  # loss 0.03 > 0.02 -> violation
    return ret, var


def _ref_lr_pof(x: int, T: int, p: float) -> float:
    def xlogx(count, prob):
        return 0.0 if count == 0 else count * math.log(prob)

    pi = x / T
    ll0 = xlogx(x, p) + xlogx(T - x, 1 - p)
    lla = xlogx(x, pi) + xlogx(T - x, 1 - pi)
    return -2.0 * (ll0 - lla)


# ---------------------------------------------------------------------------
# 1. Kupiec POF hand fixtures.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("x,T,p", [(10, 250, 0.01), (5, 500, 0.05), (25, 1000, 0.01)])
def test_lr_pof_matches_formula(x, T, p):
    assert _lr_pof(x, T, p) == pytest.approx(_ref_lr_pof(x, T, p), abs=1e-10)


def test_lr_pof_general_absolute_anchor():
    # x=10, T=250, p=0.01 hand-computed -> 12.95549
    assert _lr_pof(10, 250, 0.01) == pytest.approx(12.95549, abs=1e-4)


def test_lr_pof_x_zero_is_not_zero():
    # x=0 -> LR = -2T ln(1-p); NOT zero. T=250, c=0.99 -> ~5.0252
    lr = _lr_pof(0, 250, 0.01)
    assert lr == pytest.approx(-2 * 250 * math.log(0.99), abs=1e-10)
    assert lr == pytest.approx(5.025168, abs=1e-6)  # ~5.03, precise value pinned
    assert lr > 0


def test_lr_pof_x_equals_T_is_not_zero():
    # x=T -> LR = -2T ln(p); NOT zero, no log(0) crash.
    lr = _lr_pof(250, 250, 0.01)
    assert lr == pytest.approx(-2 * 250 * math.log(0.01), abs=1e-10)
    assert lr > 0


# ---------------------------------------------------------------------------
# 2. Christoffersen independence + CC hand fixtures.
# ---------------------------------------------------------------------------
def test_transition_counts():
    seq = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    viol = pd.Series(np.array(seq, dtype=bool), index=_dates(len(seq)))
    assert _transition_counts(viol) == (6, 0, 1, 2)  # n00, n01, n10, n11


def test_lr_ind_hand_fixture():
    # counts (n00,n01,n10,n11) = (6,0,1,2) -> LR_ind = 5.715626
    assert _lr_ind((6, 0, 1, 2)) == pytest.approx(5.715626, abs=1e-6)


def test_lr_cc_is_pof_plus_ind():
    counts = (6, 0, 1, 2)
    lr_ind = _lr_ind(counts)
    lr_pof = _lr_pof(3, 10, 0.01)
    assert lr_pof + lr_ind == pytest.approx(15.55444 + 5.715626, abs=1e-4)


def test_lr_ind_degenerate_no_violation_transition_is_zero():
    # x=0 -> n10+n11=0 -> clustering undetectable -> LR_ind = 0
    assert _lr_ind((9, 0, 0, 0)) == 0.0
    # all violations -> n00+n01=0 -> also degenerate
    assert _lr_ind((0, 0, 0, 9)) == 0.0


def test_lr_ind_degenerate_pi_zero_or_one_is_zero():
    # Single isolated leading violation: seq=[1,0,0,...] -> (5,0,1,0); transition
    # rate pi=0 even though both denominators are positive -> LR_ind = 0.
    assert _lr_ind((5, 0, 1, 0)) == 0.0
    # Symmetric pi=1 case.
    assert _lr_ind((0, 1, 0, 5)) == 0.0


# ---------------------------------------------------------------------------
# 3-5. Perfect / bad / clustered models through backtest_var.
# ---------------------------------------------------------------------------
def test_perfect_model_pof_and_ind_pass():
    # c=0.95 -> p=0.05; exactly 25/500 violations, uniformly spaced.
    ret, var = _make(500, list(range(0, 500, 20)))
    res = backtest_var(ret, var, confidence=0.95, method="test")
    assert res.n_violations == 25
    assert res.violation_rate == pytest.approx(0.05)
    assert res.lr_pof == pytest.approx(0.0, abs=1e-9)  # pi == p
    assert res.pvalue_pof > 0.05  # not rejected
    assert res.pvalue_ind > 0.05  # uniform -> independence not rejected


def test_too_many_violations_pof_rejects():
    # 3x expected: 75/500 violations.
    rng = np.random.default_rng(0)
    positions = sorted(rng.choice(500, size=75, replace=False).tolist())
    ret, var = _make(500, positions)
    res = backtest_var(ret, var, confidence=0.95, method="test")
    assert res.n_violations == 75
    assert res.pvalue_pof < 0.05  # too many -> rejected


def test_clustered_violations_ind_rejects_pof_passes():
    # Same count as the perfect model (25) but clustered at the start.
    ret, var = _make(500, list(range(25)))
    res = backtest_var(ret, var, confidence=0.95, method="test")
    assert res.n_violations == 25
    assert res.pvalue_pof > 0.05  # correct count -> POF still passes
    assert res.pvalue_ind < 0.05  # clustering detected


# ---------------------------------------------------------------------------
# 6. Edge cases.
# ---------------------------------------------------------------------------
def test_zero_violations_no_crash():
    ret, var = _make(50, [])
    res = backtest_var(ret, var, confidence=0.99, method="test")
    assert res.n_violations == 0
    assert res.lr_pof == pytest.approx(-2 * 50 * math.log(0.99), abs=1e-10)
    assert res.pvalue_pof > 0.05


def test_all_violations_no_crash():
    ret, var = _make(50, list(range(50)))
    res = backtest_var(ret, var, confidence=0.99, method="test")
    assert res.n_violations == 50
    assert res.lr_pof == pytest.approx(-2 * 50 * math.log(0.01), abs=1e-10)


def test_insufficient_obs_raises():
    ret, var = _make(20, [1, 5])
    with pytest.raises(ValueError):
        backtest_var(ret, var, confidence=0.99)


def test_index_mismatch_raises():
    ret, var = _make(50, [1])
    var2 = var.copy()
    var2.index = _dates(50, start="2011-01-01")  # different dates
    with pytest.raises(ValueError):
        backtest_var(ret, var2, confidence=0.99)
    with pytest.raises(ValueError):
        violations(ret, var2)


def test_nan_pairs_dropped_and_logged(caplog):
    ret, var = _make(60, [1, 10, 20])
    ret.iloc[5] = np.nan
    var.iloc[6] = np.nan
    with caplog.at_level(logging.WARNING):
        res = backtest_var(ret, var, confidence=0.99, method="test")
    assert res.n_obs == 58  # 60 - 2 NaN pairs
    assert "nan" in caplog.text.lower()


def test_negative_var_warns_no_raise(caplog):
    ret, var = _make(50, [1, 2])
    var.iloc[3] = -0.01  # sign error in caller
    with caplog.at_level(logging.WARNING):
        res = backtest_var(ret, var, confidence=0.99, method="test")
    assert isinstance(res, BacktestResult)
    assert "negative" in caplog.text.lower() or "sign" in caplog.text.lower()


def test_confidence_out_of_range_raises():
    ret, var = _make(50, [1])
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            backtest_var(ret, var, confidence=bad)


# ---------------------------------------------------------------------------
# 7. Basel traffic-light zones.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "n_viol,zone",
    [
        (0, "green"),
        (4, "green"),
        (5, "yellow"),
        (9, "yellow"),
        (10, "red"),
        (25, "red"),
    ],
)
def test_basel_zone_250(n_viol, zone):
    assert _basel_zone(n_viol, 250) == zone


@pytest.mark.parametrize("n_obs", [249, 251, 500, 100])
def test_basel_zone_na_when_not_250(n_obs):
    assert _basel_zone(5, n_obs) == "n/a"


def test_basel_zone_via_backtest_250():
    ret, var = _make(250, list(range(7)))  # 7 violations -> yellow
    res = backtest_var(ret, var, confidence=0.99, method="test")
    assert res.n_obs == 250
    assert res.basel_zone == "yellow"


def test_basel_zone_na_via_backtest_non_250():
    ret, var = _make(300, list(range(7)))
    res = backtest_var(ret, var, confidence=0.99, method="test")
    assert res.basel_zone == "n/a"


# ---------------------------------------------------------------------------
# 8. violations() boolean series.
# ---------------------------------------------------------------------------
def test_violations_strict_and_aligned():
    idx = _dates(4)
    ret = pd.Series([-0.03, 0.01, -0.02, -0.05], index=idx)
    var = pd.Series([0.02, 0.02, 0.02, 0.02], index=idx)
    # losses: 0.03 > 0.02 T ; -0.01 not ; 0.02 == 0.02 -> NOT (strict) ; 0.05 > 0.02 T
    out = violations(ret, var)
    assert out.tolist() == [True, False, False, True]
    assert out.index.equals(idx)


# ---------------------------------------------------------------------------
# 9. Cross-module integration (real data; skipped by default).
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_integration_spy_pipeline():
    from src.data.loader import fetch_prices, prices_to_returns
    from src.var.historical import rolling_historical_var
    from src.var.parametric import rolling_parametric_var

    prices = fetch_prices(["SPY"], "2005-01-01", "2022-12-31")
    returns = prices_to_returns(prices, method="simple")["SPY"]

    configs = [
        ("historical", rolling_historical_var(returns, window=500, confidence=0.99)),
        (
            "parametric-normal",
            rolling_parametric_var(
                returns, method="normal", window=500, confidence=0.99
            ),
        ),
        (
            "parametric-t",
            rolling_parametric_var(returns, method="t", window=500, confidence=0.99),
        ),
    ]
    results = []
    for method, roll in configs:
        aligned = returns.loc[roll.index]
        res = backtest_var(aligned, roll["var"], confidence=0.99, method=method)
        results.append(res)
        print(
            f"\n[{res.method}] n_obs={res.n_obs} n_viol={res.n_violations} "
            f"rate={res.violation_rate:.4f} (expected {res.expected_rate:.4f}) "
            f"pof_p={res.pvalue_pof:.4f} ind_p={res.pvalue_ind:.4f} "
            f"cc_p={res.pvalue_cc:.4f} basel={res.basel_zone}"
        )

    # Assert AFTER printing so all three rates are visible even if one is out of range.
    # This is a PIPELINE smoke test, not a model-quality gate: the band is wide
    # enough to catch a broken pipeline (zero violations, or a sign error at ~50%)
    # without penalizing a model that legitimately over-violates. On SPY 2005-2022
    # (2008 + 2020 crashes) parametric VaR genuinely over-violates at ~3x nominal
    # (normal ~3.2%, t ~3.6% vs 1% expected) and is correctly rejected by POF/ind;
    # that is reported, not asserted away. See CLAUDE.md "report what the data gives".
    for res in results:
        assert res.n_obs > 1000
        assert 0.001 <= res.violation_rate <= 0.10, (res.method, res.violation_rate)
