"""Tests for parametric VaR + ES (src/var/parametric.py).

Synthetic returns only: seeded numpy normal and Student-t series. No market data.
Conventions under test: positive loss magnitudes, ES >= VaR, no-lookahead rolling,
MIN_OBS floor, NaN drop+log, do-not-inflate. Closed forms are asserted against
scipy directly so the math is pinned, not just self-consistent.
"""

import logging
import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm
from scipy.stats import t as student_t

from src.var.historical import historical_var
from src.var.parametric import (
    ParamVaRResult,
    _normal_var_es,
    _t_var_es,
    parametric_normal_var,
    parametric_t_var,
    rolling_parametric_var,
)


def _dates(n: int, start: str = "2020-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="B")


def _series(values, start: str = "2020-01-01") -> pd.Series:
    arr = np.asarray(values, dtype=float)
    return pd.Series(arr, index=_dates(len(arr), start))


# ---------------------------------------------------------------------------
# 1. Closed-form fixture (normal), asserted against scipy directly.
# ---------------------------------------------------------------------------
def test_normal_core_matches_scipy_closed_form():
    mu, sigma, c = 0.0005, 0.012, 0.99

    var, es = _normal_var_es(mu, sigma, c)

    z = norm.ppf(1 - c)
    expected_var = -(mu + sigma * z)
    expected_es = -(mu - sigma * norm.pdf(z) / (1 - c))
    assert var == pytest.approx(expected_var, abs=1e-10)
    assert es == pytest.approx(expected_es, abs=1e-10)
    assert es >= var


# ---------------------------------------------------------------------------
# 2. Closed-form fixture (Student-t), asserted against scipy directly.
# ---------------------------------------------------------------------------
def test_t_core_matches_scipy_closed_form():
    mu, sigma, nu, c = 0.0005, 0.012, 6.0, 0.99

    var, es = _t_var_es(mu, sigma, nu, c)

    # Standardized-t: sigma is the actual volatility; convert to the t scale so
    # the distribution's SD equals sigma. MINUS on the ES tail term (corrected
    # from the spec's sign typo). See _t_var_es docstring.
    scale = sigma * math.sqrt((nu - 2) / nu)
    q = student_t.ppf(1 - c, df=nu)
    expected_var = -(mu + scale * q)
    expected_es = -(
        mu - scale * student_t.pdf(q, df=nu) / (1 - c) * (nu + q**2) / (nu - 1)
    )
    assert var == pytest.approx(expected_var, abs=1e-10)
    assert es == pytest.approx(expected_es, abs=1e-10)
    assert es >= var


# ---------------------------------------------------------------------------
# 3. Property: ES >= VaR for both methods.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("method", ["normal", "t"])
def test_es_ge_var_property(method):
    rng = np.random.default_rng(0)
    for _ in range(30):
        s = _series(rng.standard_normal(300) * 0.01)
        if method == "normal":
            res = parametric_normal_var(s, confidence=0.99)
        else:
            res = parametric_t_var(s, confidence=0.99, nu=6.0)
        assert res.es >= res.var


# ---------------------------------------------------------------------------
# 4. Monotonicity: VaR(0.99) >= VaR(0.95) for both methods.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("method", ["normal", "t"])
def test_var_monotonic_in_confidence(method):
    rng = np.random.default_rng(1)
    s = _series(rng.standard_normal(2000) * 0.01)
    if method == "normal":
        v99 = parametric_normal_var(s, confidence=0.99).var
        v95 = parametric_normal_var(s, confidence=0.95).var
    else:
        v99 = parametric_t_var(s, confidence=0.99, nu=6.0).var
        v95 = parametric_t_var(s, confidence=0.95, nu=6.0).var
    assert v99 >= v95


# ---------------------------------------------------------------------------
# 5. Cross-module: normal parametric converges to historical on normal data.
# ---------------------------------------------------------------------------
def test_normal_converges_to_historical():
    rng = np.random.default_rng(42)
    sigma = 0.01
    s = _series(rng.standard_normal(100_000) * sigma)

    p = parametric_normal_var(s, confidence=0.99).var
    h = historical_var(s, confidence=0.99).var

    assert p == pytest.approx(h, rel=0.03)  # loose: tail sampling noise


# ---------------------------------------------------------------------------
# 6. Fat tails: parametric-t ES exceeds parametric-normal ES (direction only).
# ---------------------------------------------------------------------------
def test_t_es_exceeds_normal_es_on_fat_tails():
    rng = np.random.default_rng(7)
    s = _series(rng.standard_t(3, 100_000) * 0.01)  # df=3, heavy tails

    es_normal = parametric_normal_var(s, confidence=0.99).es
    es_t = parametric_t_var(s, confidence=0.99).es  # nu estimated via MLE

    assert es_t > es_normal


# ---------------------------------------------------------------------------
# 7. nu MLE lands in a plausible range on df=5 data (loose, estimation variance).
# ---------------------------------------------------------------------------
def test_nu_estimation_in_plausible_range():
    rng = np.random.default_rng(11)
    s = _series(rng.standard_t(5, 50_000) * 0.01)

    res = parametric_t_var(s, confidence=0.99)  # estimates nu

    assert res.nu is not None
    assert 3.0 <= res.nu <= 10.0


# ---------------------------------------------------------------------------
# 8a. LOOKAHEAD GUARD - mutation AT and AFTER t leaves row t unchanged.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("method", ["normal", "t"])
def test_lookahead_mutation_at_and_after_t(method):
    # window=120 keeps the per-window Student-t MLE well-conditioned (min nu ~14
    # on normal data); a +0.05 shock (~5 sigma) is large enough that an erroneous
    # read of date t would change the row, but small enough not to drive a window
    # to the nu<=2 infinite-variance edge. This isolates the lookahead property.
    rng = np.random.default_rng(3)
    window, n = 120, 200
    s = _series(rng.standard_normal(n) * 0.01)

    base = rolling_parametric_var(s, method=method, window=window, confidence=0.95)

    i = 160
    t = s.index[i]
    mutated = s.copy()
    mutated.iloc[i:] += 0.05  # shock at t itself and every later date

    after = rolling_parametric_var(
        mutated, method=method, window=window, confidence=0.95
    )

    assert after.loc[t, "var"] == base.loc[t, "var"]
    assert after.loc[t, "es"] == base.loc[t, "es"]


# ---------------------------------------------------------------------------
# 8b. LOOKAHEAD GUARD - rolling row equals the point-in-time function on the
#     exact prior window.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("method", ["normal", "t"])
def test_rolling_equals_pointwise_on_prior_window(method):
    rng = np.random.default_rng(11)
    window, n = 120, 200  # large enough to keep the per-window t-MLE stable
    s = _series(rng.standard_normal(n) * 0.01)

    roll = rolling_parametric_var(s, method=method, window=window, confidence=0.95)
    func = parametric_normal_var if method == "normal" else parametric_t_var

    for i in (window, 160, 199):
        t = s.index[i]
        prior = s.iloc[i - window : i]  # left-inclusive, right-EXCLUSIVE of t
        direct = func(prior, confidence=0.95)
        assert roll.loc[t, "var"] == pytest.approx(direct.var)
        assert roll.loc[t, "es"] == pytest.approx(direct.es)


# ---------------------------------------------------------------------------
# 9. Edge cases.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad_nu", [2.0, 1.5, 0.5])
def test_explicit_nu_le_2_raises(bad_nu):
    s = _series(np.random.default_rng(0).standard_normal(300) * 0.01)
    with pytest.raises(ValueError):
        parametric_t_var(s, confidence=0.99, nu=bad_nu)


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_confidence_out_of_range_raises(bad):
    s = _series(np.random.default_rng(0).standard_normal(300) * 0.01)
    with pytest.raises(ValueError):
        parametric_normal_var(s, confidence=bad)
    with pytest.raises(ValueError):
        parametric_t_var(s, confidence=bad, nu=6.0)


@pytest.mark.parametrize("bad_horizon", [0, -1])
def test_non_positive_horizon_raises(bad_horizon):
    s = _series(np.random.default_rng(0).standard_normal(300) * 0.01)
    with pytest.raises(ValueError):
        parametric_normal_var(s, horizon=bad_horizon)
    with pytest.raises(ValueError):
        parametric_t_var(s, horizon=bad_horizon, nu=6.0)


def test_below_min_obs_raises():
    # c=0.99 -> MIN_OBS=100; 99 observations is one short.
    short = _series(np.random.default_rng(0).standard_normal(99) * 0.01)
    with pytest.raises(ValueError):
        parametric_normal_var(short, confidence=0.99)
    with pytest.raises(ValueError):
        parametric_t_var(short, confidence=0.99, nu=6.0)


def test_nan_dropped_and_logged(caplog):
    s = _series(np.random.default_rng(0).standard_normal(120) * 0.01)
    s.iloc[5] = np.nan
    s.iloc[50] = np.nan

    with caplog.at_level(logging.WARNING):
        res = parametric_normal_var(s, confidence=0.95)  # MIN_OBS=20, n=118

    assert res.n_obs == 118
    assert "nan" in caplog.text.lower()


def test_horizon_scaling_approx_sqrt_h_zero_mean():
    rng = np.random.default_rng(5)
    s = _series(rng.standard_normal(5000) * 0.01)
    s = s - s.mean()  # force (near) zero mean so VaR(h) = sqrt(h) * VaR(1) exactly

    v1 = parametric_normal_var(s, confidence=0.99, horizon=1).var
    v2 = parametric_normal_var(s, confidence=0.99, horizon=2).var

    assert v2 != v1
    assert v2 == pytest.approx(math.sqrt(2) * v1, rel=0.05)


def test_low_nu_warns(caplog):
    rng = np.random.default_rng(7)
    s = _series(rng.standard_t(3, 20_000) * 0.01)

    with caplog.at_level(logging.WARNING):
        res = parametric_t_var(s, confidence=0.99)

    assert res.nu <= 5
    assert "nu" in caplog.text.lower()


# ---------------------------------------------------------------------------
# 10. Rolling shape: first row at index[window], length == n_obs - window.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("method", ["normal", "t"])
def test_rolling_shape(method):
    window, n = 40, 100
    s = _series(np.random.default_rng(2).standard_normal(n) * 0.01)

    roll = rolling_parametric_var(s, method=method, window=window, confidence=0.95)

    assert roll.index[0] == s.index[window]
    assert len(roll) == n - window
    assert list(roll.columns) == ["var", "es"]


# ---------------------------------------------------------------------------
# 11. Decision B: parametric rolling floor is on `window`, NOT window // horizon
#     (no block compression — unlike historical).
# ---------------------------------------------------------------------------
def test_rolling_floor_is_window_not_divided_by_horizon():
    # c=0.99 -> MIN_OBS=100. window=120, horizon=10.
    # Historical would compress to 120//10=12 < 100 and raise; parametric uses
    # all 120 observations (sqrt scaling), so effective=120 >= 100 -> no raise.
    s = _series(np.random.default_rng(0).standard_normal(300) * 0.01)
    roll = rolling_parametric_var(
        s, method="normal", window=120, horizon=10, confidence=0.99
    )
    assert isinstance(roll, pd.DataFrame)
    assert len(roll) == 300 - 120

    # window below MIN_OBS still raises regardless of horizon.
    with pytest.raises(ValueError):
        rolling_parametric_var(s, method="normal", window=50, confidence=0.99)


# ---------------------------------------------------------------------------
# 12. Rolling guard rails.
# ---------------------------------------------------------------------------
def test_rolling_invalid_method_raises():
    s = _series(np.random.default_rng(0).standard_normal(300) * 0.01)
    with pytest.raises(ValueError):
        rolling_parametric_var(s, method="lognormal", window=40, confidence=0.95)


def test_rolling_raises_when_obs_not_greater_than_window():
    # c=0.95 -> floor 20, window=40 passes the floor; n == window fires the
    # "need more than window" guard.
    s = _series(np.random.default_rng(0).standard_normal(40) * 0.01)
    with pytest.raises(ValueError):
        rolling_parametric_var(s, method="normal", window=40, confidence=0.95)


def test_rolling_non_positive_window_raises():
    s = _series(np.random.default_rng(0).standard_normal(300) * 0.01)
    with pytest.raises(ValueError):
        rolling_parametric_var(s, method="normal", window=0, confidence=0.95)


# ---------------------------------------------------------------------------
# 12b. rolling-t degrades to NaN on a window whose MLE gives nu<=2, rather than
#      aborting the whole series. Point-in-time parametric_t_var still raises.
# ---------------------------------------------------------------------------
def _bimodal_series(n=300):
    # A contiguous block shifted by a large constant makes windows straddling the
    # block boundary bimodal -> free-scale t-MLE returns nu<=2 for some windows.
    rng = np.random.default_rng(0)
    x = rng.standard_normal(n) * 0.01
    s = pd.Series(x, index=_dates(n))
    s.iloc[150:165] += 5.0
    return s


def test_rolling_t_nan_on_pathological_window_does_not_abort():
    s = _bimodal_series()
    n, window = 300, 120

    roll = rolling_parametric_var(s, method="t", window=window, confidence=0.95)

    # the series is NOT aborted: full length, all rows present
    assert len(roll) == n - window
    # at least one row is NaN (a window the t-fit could not resolve, nu<=2)
    nan_mask = roll["var"].isna()
    assert nan_mask.any()
    # a NaN row is NaN in BOTH columns
    assert roll.loc[nan_mask, "es"].isna().all()


def test_rolling_t_continues_finite_around_nan_rows():
    s = _bimodal_series()
    roll = rolling_parametric_var(s, method="t", window=120, confidence=0.95)

    nan_mask = roll["var"].isna()
    assert nan_mask.any()
    first_nan = int(nan_mask.values.argmax())
    last_nan = len(nan_mask) - 1 - int(nan_mask.values[::-1].argmax())

    before = roll.iloc[:first_nan]
    after = roll.iloc[last_nan + 1 :]

    # finite rows exist on BOTH sides of the NaN region...
    assert len(before) > 0 and before["var"].notna().all()
    assert len(after) > 0 and after["var"].notna().all()
    # ...and they still satisfy ES >= VaR (the series continues correctly).
    assert (before["es"] >= before["var"]).all()
    assert (after["es"] >= after["var"]).all()


# ---------------------------------------------------------------------------
# 13. Result metadata is populated correctly.
# ---------------------------------------------------------------------------
def test_result_metadata():
    s = _series(np.random.default_rng(0).standard_normal(300) * 0.01)

    rn = parametric_normal_var(s, confidence=0.99)
    assert isinstance(rn, ParamVaRResult)
    assert rn.method == "parametric-normal"
    assert rn.nu is None
    assert rn.n_obs == 300
    assert rn.sigma > 0
    assert rn.horizon == 1

    rt = parametric_t_var(s, confidence=0.99, nu=6.0)
    assert rt.method == "parametric-t"
    assert rt.nu == 6.0
