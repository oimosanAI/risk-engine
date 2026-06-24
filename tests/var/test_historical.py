"""Tests for historical-simulation VaR + ES (src/var/historical.py).

Synthetic returns only: seeded numpy normal and Student-t series. No market data.
Conventions under test: VaR/ES as POSITIVE loss magnitudes, ES >= VaR, no lookahead
in rolling estimates, empirical numbers reported without inflation.
"""

import logging
import math

import numpy as np
import pandas as pd
import pytest

from src.var.historical import (
    VaRResult,
    _aggregate_horizon,
    _min_obs,
    _var_es_from_returns,
    historical_var,
    portfolio_returns,
    rolling_historical_var,
)


def _dates(n: int, start: str = "2020-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="B")


def _series(values, start: str = "2020-01-01") -> pd.Series:
    arr = np.asarray(values, dtype=float)
    return pd.Series(arr, index=_dates(len(arr), start))


# ---------------------------------------------------------------------------
# 1. Hand-computable fixture, asserted against the private numeric core.
# ---------------------------------------------------------------------------
def test_core_var_es_hand_computed_lower():
    # Arrange: returns -> losses = [-0.01, 0.02, -0.03, 0.04, -0.05]
    #          sorted losses     = [-0.05, -0.03, -0.01, 0.02, 0.04]
    returns = np.array([0.01, -0.02, 0.03, -0.04, 0.05])

    # Act: quantile 0.8 with method "lower" -> virtual index 3.2 -> sorted[3] = 0.02
    var, es = _var_es_from_returns(returns, confidence=0.8, interpolation="lower")

    # Assert: VaR = 0.02; ES = mean(losses >= 0.02) = mean(0.02, 0.04) = 0.03
    assert var == pytest.approx(0.02)
    assert es == pytest.approx(0.03)
    assert es >= var


# ---------------------------------------------------------------------------
# 2. Property: ES >= VaR for arbitrary seeded inputs.
# ---------------------------------------------------------------------------
def test_es_ge_var_property():
    rng = np.random.default_rng(0)
    for _ in range(50):
        returns = rng.standard_normal(200) * 0.01
        for confidence in (0.90, 0.95, 0.99):
            var, es = _var_es_from_returns(returns, confidence, "linear")
            assert es >= var - 1e-12


# ---------------------------------------------------------------------------
# 3. Monotonicity: VaR(0.99) >= VaR(0.95).
# ---------------------------------------------------------------------------
def test_var_monotonic_in_confidence():
    rng = np.random.default_rng(1)
    s = _series(rng.standard_normal(2000) * 0.01)

    v99 = historical_var(s, confidence=0.99).var
    v95 = historical_var(s, confidence=0.95).var

    assert v99 >= v95


# ---------------------------------------------------------------------------
# 4. Large-sample sanity: N=100k standard normal ~ analytic normal VaR.
# ---------------------------------------------------------------------------
def test_large_sample_matches_analytic_normal():
    rng = np.random.default_rng(42)
    sigma = 0.01
    s = _series(rng.standard_normal(100_000) * sigma)

    res = historical_var(s, confidence=0.99, interpolation="linear")

    assert res.var == pytest.approx(
        2.326 * sigma, rel=0.1
    )  # loose: tail sampling error
    assert res.method == "historical"
    assert res.n_obs == 100_000
    assert isinstance(res, VaRResult)


# ---------------------------------------------------------------------------
# 5. Fat tails: Student-t ES exceeds normal ES at matched scale.
# ---------------------------------------------------------------------------
def test_fat_tail_es_exceeds_normal():
    rng = np.random.default_rng(7)
    n = 100_000
    sigma = 0.01
    df = 3
    normal = rng.standard_normal(n) * sigma
    # scale t to unit variance (Var(t_df) = df/(df-2)) then to sigma
    student_t = rng.standard_t(df, n) / math.sqrt(df / (df - 2)) * sigma

    es_normal = _var_es_from_returns(normal, 0.99, "linear")[1]
    es_t = _var_es_from_returns(student_t, 0.99, "linear")[1]

    assert es_t > es_normal


# ---------------------------------------------------------------------------
# 6a. LOOKAHEAD GUARD - mutation AT and AFTER t (not just after).
# ---------------------------------------------------------------------------
def test_lookahead_mutation_at_and_after_t():
    """Mutating only dates strictly after t would pass even a `.rolling(window)`
    bug that reads the current row. We mutate at-and-after t to catch that."""
    rng = np.random.default_rng(3)
    window = 50
    s = _series(rng.standard_normal(200) * 0.01)

    base = rolling_historical_var(
        s, window=window, confidence=0.9, interpolation="lower"
    )

    i = 120
    t = s.index[i]
    mutated = s.copy()
    mutated.iloc[i:] += 5.0  # shock at t itself and every later date

    after = rolling_historical_var(
        mutated, window=window, confidence=0.9, interpolation="lower"
    )

    assert after.loc[t, "var"] == base.loc[t, "var"]
    assert after.loc[t, "es"] == base.loc[t, "es"]


# ---------------------------------------------------------------------------
# 6b. LOOKAHEAD GUARD - rolling row equals historical_var on the exact prior window.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("confidence", [0.9, 0.8])
def test_rolling_equals_historical_on_prior_window(confidence):
    rng = np.random.default_rng(11)
    window = 50
    s = _series(rng.standard_normal(220) * 0.01)

    roll = rolling_historical_var(
        s, window=window, confidence=confidence, interpolation="lower"
    )

    for i in (window, 100, 219):
        t = s.index[i]
        prior = s.iloc[i - window : i]  # left-inclusive, right-EXCLUSIVE of t
        direct = historical_var(
            prior, confidence=confidence, horizon=1, interpolation="lower"
        )
        assert roll.loc[t, "var"] == pytest.approx(direct.var)
        assert roll.loc[t, "es"] == pytest.approx(direct.es)


# ---------------------------------------------------------------------------
# 7. Rolling shape: first row at index[window], length == n_obs - window.
# ---------------------------------------------------------------------------
def test_rolling_first_row_and_length():
    window = 30
    s = _series(np.random.default_rng(2).standard_normal(100) * 0.01)

    roll = rolling_historical_var(
        s, window=window, confidence=0.9, interpolation="lower"
    )

    assert roll.index[0] == s.index[window]
    assert len(roll) == len(s) - window
    assert list(roll.columns) == ["var", "es"]


# ---------------------------------------------------------------------------
# 8. Edge case: rolling raises when n_obs is not strictly greater than window.
# ---------------------------------------------------------------------------
def test_rolling_raises_when_obs_not_greater_than_window():
    # confidence=0.5 -> floor=2, so the effective-obs floor passes and the
    # n <= window guard is what fires (n == window, no strictly-prior window).
    s = _series(np.random.default_rng(0).standard_normal(50) * 0.01)
    with pytest.raises(ValueError):
        rolling_historical_var(s, window=50, confidence=0.5)


# ---------------------------------------------------------------------------
# 9. Edge case: confidence outside (0, 1) raises.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_confidence_out_of_range_raises(bad):
    s = _series(np.random.default_rng(0).standard_normal(200) * 0.01)
    with pytest.raises(ValueError):
        historical_var(s, confidence=bad)


# ---------------------------------------------------------------------------
# 10. Edge case: NaNs dropped explicitly and the count is logged.
# ---------------------------------------------------------------------------
def test_nan_dropped_and_logged(caplog):
    s = _series([0.01, 0.0, -0.02, 0.03, 0.0, -0.05])
    s.iloc[1] = np.nan
    s.iloc[4] = np.nan

    with caplog.at_level(logging.WARNING):
        res = historical_var(s, confidence=0.5, interpolation="lower")

    assert res.n_obs == 4  # 6 - 2 dropped
    assert "nan" in caplog.text.lower()


# ---------------------------------------------------------------------------
# 11. Edge case: all-positive returns -> VaR not clamped to zero.
# ---------------------------------------------------------------------------
def test_all_positive_returns_var_not_clamped():
    s = _series([0.01, 0.02, 0.03, 0.04, 0.05, 0.06])

    res = historical_var(s, confidence=0.5, interpolation="lower")

    assert res.var < 0  # no real loss -> negative loss magnitude, never clamped to 0
    assert res.es >= res.var


# ---------------------------------------------------------------------------
# 12. interpolation is a real parameter: changing it changes the result.
# ---------------------------------------------------------------------------
def test_interpolation_changes_result():
    s = _series(np.random.default_rng(5).standard_normal(500) * 0.01)

    low = historical_var(s, confidence=0.99, interpolation="lower").var
    high = historical_var(s, confidence=0.99, interpolation="higher").var

    assert low != high


# ---------------------------------------------------------------------------
# 13. MIN_OBS = ceil(1/(1-c)); historical_var raises below it.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("confidence,floor", [(0.95, 20), (0.99, 100), (0.5, 2)])
def test_historical_var_requires_min_obs(confidence, floor):
    assert _min_obs(confidence) == floor

    short = _series(np.random.default_rng(0).standard_normal(floor - 1) * 0.01)
    with pytest.raises(ValueError):
        historical_var(short, confidence=confidence)

    ok = _series(np.random.default_rng(0).standard_normal(floor) * 0.01)
    res = historical_var(ok, confidence=confidence, interpolation="lower")
    assert res.n_obs == floor


# ---------------------------------------------------------------------------
# 14. horizon>1 uses non-overlapping block compounding, NOT sqrt(h) scaling.
# ---------------------------------------------------------------------------
def test_horizon_uses_nonoverlapping_blocks_not_sqrt_scaling():
    returns = [-0.02, -0.03, 0.01, -0.05, 0.02, -0.01, -0.04, 0.03]
    s = _series(returns)

    # manual non-overlapping 2-day block returns: prod(1+r) - 1
    blocks = [
        (1 + returns[j]) * (1 + returns[j + 1]) - 1 for j in range(0, len(returns), 2)
    ]
    expected_var, expected_es = _var_es_from_returns(np.array(blocks), 0.5, "lower")

    res2 = historical_var(s, confidence=0.5, horizon=2, interpolation="lower")
    assert res2.var == pytest.approx(expected_var)
    assert res2.es == pytest.approx(expected_es)
    assert res2.horizon == 2

    # explicitly NOT the iid sqrt(h) approximation of 1-day VaR
    var1 = historical_var(s, confidence=0.5, horizon=1, interpolation="lower").var
    assert res2.var != pytest.approx(math.sqrt(2) * var1)


# ---------------------------------------------------------------------------
# 15. portfolio_returns: dict weights == ndarray weights.
# ---------------------------------------------------------------------------
def test_portfolio_weights_dict_matches_array():
    rng = np.random.default_rng(9)
    df = pd.DataFrame(
        {"A": rng.standard_normal(100) * 0.01, "B": rng.standard_normal(100) * 0.01},
        index=_dates(100),
    )

    by_array = portfolio_returns(df, np.array([0.6, 0.4]))
    by_dict = portfolio_returns(df, {"A": 0.6, "B": 0.4})

    pd.testing.assert_series_equal(by_array, by_dict)


# ---------------------------------------------------------------------------
# 16. portfolio_returns: any-NaN day is dropped (matmul propagates NaN).
# ---------------------------------------------------------------------------
def test_portfolio_drops_rows_with_any_nan_asset(caplog):
    df = pd.DataFrame(
        {"A": [0.01, 0.02, np.nan, 0.04], "B": [0.0, -0.01, 0.02, 0.03]},
        index=_dates(4),
    )

    with caplog.at_level(logging.WARNING):
        out = portfolio_returns(df, {"A": 0.5, "B": 0.5})

    assert len(out) == 3  # day with NaN in asset A is dropped, not skipna-masked
    assert df.index[2] not in out.index


# ---------------------------------------------------------------------------
# 17. portfolio_returns: warns on non-unit weights, does not re-normalize.
# ---------------------------------------------------------------------------
def test_portfolio_warns_and_does_not_renormalize(caplog):
    df = pd.DataFrame({"A": [0.01, 0.02], "B": [0.0, 0.01]}, index=_dates(2))

    with caplog.at_level(logging.WARNING):
        out = portfolio_returns(df, [0.6, 0.6])

    assert "sum" in caplog.text.lower()
    # day 0 = 0.6*0.01 + 0.6*0.0 = 0.006 (NOT re-normalized to weights summing to 1)
    assert out.iloc[0] == pytest.approx(0.006)


# ---------------------------------------------------------------------------
# 18. portfolio_returns: mismatched dict keys raise.
# ---------------------------------------------------------------------------
def test_portfolio_unknown_weight_key_raises():
    df = pd.DataFrame({"A": [0.01], "B": [0.0]}, index=_dates(1))
    with pytest.raises(ValueError):
        portfolio_returns(df, {"A": 0.5, "C": 0.5})


# ---------------------------------------------------------------------------
# 19. Boundary validations (cover every guard branch).
# ---------------------------------------------------------------------------
def test_invalid_interpolation_raises():
    s = _series(np.random.default_rng(0).standard_normal(200) * 0.01)
    with pytest.raises(ValueError):
        historical_var(s, interpolation="quadratic")


def test_non_positive_horizon_raises():
    s = _series(np.random.default_rng(0).standard_normal(200) * 0.01)
    with pytest.raises(ValueError):
        historical_var(s, horizon=0)


def test_non_positive_window_raises():
    s = _series(np.random.default_rng(0).standard_normal(200) * 0.01)
    with pytest.raises(ValueError):
        rolling_historical_var(s, window=0)


def test_aggregate_horizon_empty_when_fewer_than_horizon_obs():
    # 3 observations cannot form a single 5-day non-overlapping block.
    s = _series([0.01, -0.02, 0.03])
    out = _aggregate_horizon(s, horizon=5)
    assert len(out) == 0
    assert isinstance(out, pd.Series)


def test_portfolio_array_wrong_length_raises():
    df = pd.DataFrame({"A": [0.01, 0.02], "B": [0.0, 0.01]}, index=_dates(2))
    with pytest.raises(ValueError):
        portfolio_returns(df, np.array([0.5, 0.3, 0.2]))  # 3 weights, 2 assets


# ---------------------------------------------------------------------------
# 20. rolling enforces the same unresolvable-tail floor as historical_var,
#     applied to the EFFECTIVE per-window count (window // horizon).
# ---------------------------------------------------------------------------
def test_rolling_enforces_unresolvable_tail_floor():
    # window=50, horizon=1, c=0.99: effective 50 < floor 100 -> raises.
    s = _series(np.random.default_rng(0).standard_normal(300) * 0.01)
    with pytest.raises(ValueError):
        rolling_historical_var(s, window=50, confidence=0.99)


def test_rolling_floor_exact_boundary_with_horizon():
    assert _min_obs(0.99) == 100  # anchor the floor
    s = _series(np.random.default_rng(1).standard_normal(600) * 0.01)

    # effective == floor: window=500, horizon=5 -> 500 // 5 == 100 -> PASSES
    roll = rolling_historical_var(s, window=500, horizon=5, confidence=0.99)
    assert isinstance(roll, pd.DataFrame)
    assert list(roll.columns) == ["var", "es"]

    # effective == floor - 1: window=495, horizon=5 -> 495 // 5 == 99 -> RAISES
    with pytest.raises(ValueError):
        rolling_historical_var(s, window=495, horizon=5, confidence=0.99)


def test_rolling_default_window_horizon1_c99_still_works():
    # default window=500, horizon=1, c=0.99: effective 500 >= floor 100.
    s = _series(np.random.default_rng(2).standard_normal(600) * 0.01)
    roll = rolling_historical_var(s)  # all defaults
    assert isinstance(roll, pd.DataFrame)
    assert len(roll) == 600 - 500


# ---------------------------------------------------------------------------
# 21. portfolio_returns: non-finite weights raise before the matmul,
#     with a message distinct from the dropped-rows warning.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad_weights", [[0.5, np.nan], [0.5, np.inf], [0.5, -np.inf]])
def test_portfolio_non_finite_weights_raise(bad_weights):
    df = pd.DataFrame({"A": [0.01, 0.02], "B": [0.0, 0.01]}, index=_dates(2))
    with pytest.raises(ValueError, match="finite"):
        portfolio_returns(df, np.array(bad_weights))


def test_portfolio_nan_weight_in_dict_raises():
    df = pd.DataFrame({"A": [0.01, 0.02], "B": [0.0, 0.01]}, index=_dates(2))
    with pytest.raises(ValueError, match="finite"):
        portfolio_returns(df, {"A": 0.5, "B": np.nan})
