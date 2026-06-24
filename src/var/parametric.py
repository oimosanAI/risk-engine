"""Parametric Value-at-Risk (VaR) and Expected Shortfall (ES): normal & Student-t.

Unlike historical simulation, the parametric approach **assumes a distribution**
and reads VaR/ES from its closed form.

  Advantages: smooth (no quantile sampling jitter), extrapolates beyond the worst
  loss in the sample, and is cheap to evaluate.
  Disadvantages: the answer is only as good as the assumed distribution. The
  normal model systematically **underestimates** fat tails; the Student-t model
  captures them but depends on a stable estimate of the degrees of freedom (nu),
  which is noisy in short windows. A normal ES below the historical ES on
  fat-tailed data is the *correct* output of a misspecified model, not a bug.

Conventions (shared with historical.py; see CLAUDE.md):
  * ``loss = -return``. VaR and ES are POSITIVE loss magnitudes in return units.
  * ES >= VaR always (asserted).
  * Rolling estimates use ONLY observations strictly before the labelled date
    (no lookahead): the row at ``t`` reads ``iloc[i-window:i]``.
  * MIN_OBS = ceil(1/(1-confidence)) is enforced; below it the tail is
    unresolvable and a ValueError is raised.
  * NaNs are dropped and the count logged (shared ``_clean``).
  * Empirical estimates are reported as-is; never inflated, clamped, or padded.

Horizon scaling (decision, contrast with historical.py):
  For ``horizon = h`` we scale ``mu -> mu * h`` and ``sigma -> sigma * sqrt(h)``.
  This sqrt(h) rule assumes i.i.d. returns and breaks under autocorrelation.
  It IS appropriate here (and NOT in historical.py, which compounds
  non-overlapping blocks instead) because the parametric form already commits to
  a distributional model — the closed form scales analytically, so no resampling
  is needed.

Scale-parameter note:
  ``sigma`` is the **actual volatility** (``returns.std(ddof=1)``) for BOTH
  methods, so normal and t results are directly comparable. For the normal that
  is also the distribution scale. For the Student-t it is NOT: the t is
  standardized to this volatility via ``scale = sigma * sqrt((nu-2)/nu)`` inside
  ``_t_var_es``, so the t and normal share the same variance and differ only in
  tail shape. nu is estimated by free-scale MLE (``t.fit(returns, floc=mu)``);
  fixing the fit's scale to the sample std would bias nu high. See ``_t_var_es``
  and ``_estimate_nu``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd
from scipy.stats import norm
from scipy.stats import t as student_t

from src.var._utils import _clean, _min_obs

logger = logging.getLogger(__name__)

METHOD_NORMAL = "parametric-normal"
METHOD_T = "parametric-t"
_VALID_METHODS = frozenset({"normal", "t"})
_MIN_NU = 2.0  # below this the Student-t has infinite variance
_NU_WARN_THRESHOLD = 5.0  # heavy-tail estimation is unreliable in short windows


@dataclass
class ParamVaRResult:
    """Parametric VaR/ES result.

    ``mu`` and ``sigma`` are 1-day estimates (not h-day scaled); horizon scaling
    is applied only internally to the closed-form inputs, so they are NOT h-day
    values when ``horizon > 1``.
    """

    var: float  # positive loss magnitude
    es: float  # ES >= var
    confidence: float
    horizon: int
    method: str  # "parametric-normal" or "parametric-t"
    n_obs: int
    mu: float  # estimated mean (1-day)
    sigma: float  # estimated volatility (1-day, actual std; see _t_var_es)
    nu: float | None  # Student-t degrees of freedom; None for normal
    start: pd.Timestamp
    end: pd.Timestamp


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _validate(confidence: float, horizon: int) -> None:
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    if horizon < 1:
        raise ValueError(f"horizon must be a positive integer, got {horizon}")


def _validate_nu(nu: float) -> None:
    if nu <= _MIN_NU:
        raise ValueError(
            f"Student-t degrees of freedom must be > {_MIN_NU} for finite "
            f"variance, got nu={nu}"
        )


def _require_min_obs(n: int, confidence: float, *, label: str) -> None:
    floor = _min_obs(confidence)
    if n < floor:
        raise ValueError(
            f"{label}: need at least {floor} observation(s) for "
            f"confidence={confidence}, got {n} (the (1-confidence) tail is "
            f"otherwise unresolvable)"
        )


# ---------------------------------------------------------------------------
# Estimation
# ---------------------------------------------------------------------------
def _estimate_mu_sigma(returns: pd.Series) -> tuple[float, float]:
    return float(returns.mean()), float(returns.std(ddof=1))


def _scale_horizon(mu: float, sigma: float, horizon: int) -> tuple[float, float]:
    """Scale mu linearly and sigma by sqrt(horizon) (i.i.d. assumption)."""
    return mu * horizon, sigma * (horizon**0.5)


def _estimate_nu(returns: pd.Series, mu: float) -> float:
    """MLE of the Student-t degrees of freedom with only the location fixed.

    Fits on the RETURNS (not losses) with ``floc=mu`` and the scale estimated
    FREELY. Fixing the scale to the sample std would bias nu high by a factor
    that grows with tail heaviness: the sample std overstates the t scale
    parameter by ``sqrt(nu/(nu-2))``, so a fixed-scale fit inflates nu to
    compensate (empirically df=3 -> nu~9, df=5 -> nu~12). Freeing the scale
    recovers the true df (df=3 -> ~2.94). nu is sign-invariant, so fitting
    returns vs losses is equivalent.

    Only the df is returned; the closed forms use the actual volatility (sample
    std) as the standardizing scale, not this fit's scale estimate.
    """
    df, _loc, _scale = student_t.fit(returns.to_numpy(), floc=mu)
    return float(df)


# ---------------------------------------------------------------------------
# Closed forms (the hand-tested numeric cores)
# ---------------------------------------------------------------------------
def _normal_var_es(mu: float, sigma: float, confidence: float) -> tuple[float, float]:
    """Closed-form normal VaR and ES as positive loss magnitudes.

    VaR = -(mu + sigma * Phi^{-1}(1-c));
    ES  = -(mu - sigma * phi(Phi^{-1}(1-c)) / (1-c)).
    """
    z = float(norm.ppf(1 - confidence))
    var = -(mu + sigma * z)
    es = -(mu - sigma * float(norm.pdf(z)) / (1 - confidence))
    return float(var), float(es)


def _t_var_es(
    mu: float, sigma: float, nu: float, confidence: float
) -> tuple[float, float]:
    """Closed-form Student-t VaR and ES as positive loss magnitudes.

    Standardized-t parameterization: ``sigma`` is the **actual volatility**
    (sample ``std(ddof=1)``), the same quantity the normal path uses, so the two
    methods are directly comparable. The t distribution is standardized to that
    volatility by converting to its scale parameter::

        scale = sigma * sqrt((nu - 2) / nu)

    so that ``Std(t(loc=mu, scale=scale, df=nu)) == sigma``. Textbook formulas
    that pass the volatility straight in as the t scale omit this factor and
    thereby OVERSTATE dispersion by ``sqrt(nu/(nu-2))``; we apply it explicitly so
    the t and normal models share the same variance and differ only in tail
    shape.

    With ``q = t.ppf(1-c, nu)`` (negative, in the left tail):
      VaR = -(mu + scale * q);
      ES  = -(mu - scale * t.pdf(q,nu)/(1-c) * (nu + q^2)/(nu - 1)).
    Requires ``nu > 2`` for finite variance (and for the scale conversion).

    Sign note: the tail term carries a MINUS (matching the normal ES form). The
    conditional tail mean E[T | T <= q] of a standard t is negative, so
    subtracting ``scale * (positive)`` yields a positive loss ES with ES >= VaR.
    A plus sign here would make ES negative and violate the ES >= VaR invariant.
    """
    _validate_nu(nu)
    scale = sigma * ((nu - 2.0) / nu) ** 0.5
    q = float(student_t.ppf(1 - confidence, df=nu))
    var = -(mu + scale * q)
    es = -(
        mu
        - scale
        * float(student_t.pdf(q, df=nu))
        / (1 - confidence)
        * (nu + q**2)
        / (nu - 1)
    )
    return float(var), float(es)


# ---------------------------------------------------------------------------
# Window-level compute helpers (no logging; reused by rolling without spam)
# ---------------------------------------------------------------------------
def _normal_from_window(
    window: pd.Series, confidence: float, horizon: int
) -> tuple[float, float, float, float]:
    mu, sigma = _estimate_mu_sigma(window)
    mu_h, sigma_h = _scale_horizon(mu, sigma, horizon)
    var, es = _normal_var_es(mu_h, sigma_h, confidence)
    return var, es, mu, sigma


def _t_from_window(
    window: pd.Series, confidence: float, horizon: int, nu: float | None
) -> tuple[float, float, float, float, float]:
    mu, sigma = _estimate_mu_sigma(window)
    nu_used = _estimate_nu(window, mu) if nu is None else float(nu)
    _validate_nu(nu_used)  # raises for explicit or estimated nu <= 2
    mu_h, sigma_h = _scale_horizon(mu, sigma, horizon)
    var, es = _t_var_es(mu_h, sigma_h, nu_used, confidence)
    return var, es, mu, sigma, nu_used


# ---------------------------------------------------------------------------
# Public point-in-time functions
# ---------------------------------------------------------------------------
def parametric_normal_var(
    returns: pd.Series,
    confidence: float = 0.99,
    horizon: int = 1,
) -> ParamVaRResult:
    """Normal-distribution parametric VaR + ES (positive loss magnitudes)."""
    _validate(confidence, horizon)
    cleaned = _clean(returns)
    _require_min_obs(len(cleaned), confidence, label="parametric_normal_var")

    var, es, mu, sigma = _normal_from_window(cleaned, confidence, horizon)
    assert es >= var, "ES must be >= VaR"

    return ParamVaRResult(
        var=var,
        es=es,
        confidence=confidence,
        horizon=horizon,
        method=METHOD_NORMAL,
        n_obs=int(len(cleaned)),
        mu=mu,
        sigma=sigma,
        nu=None,
        start=cleaned.index[0],
        end=cleaned.index[-1],
    )


def parametric_t_var(
    returns: pd.Series,
    confidence: float = 0.99,
    horizon: int = 1,
    nu: float | None = None,
) -> ParamVaRResult:
    """Student-t parametric VaR + ES (positive loss magnitudes).

    If ``nu`` is None it is estimated by MLE (location/scale fixed to the sample
    mean/std). A heavy-tail estimate (``nu <= 5``) is logged as a reliability
    warning. ``nu <= 2`` (explicit or estimated) raises — infinite variance.
    """
    _validate(confidence, horizon)
    if nu is not None:
        _validate_nu(nu)
    cleaned = _clean(returns)
    _require_min_obs(len(cleaned), confidence, label="parametric_t_var")

    var, es, mu, sigma, nu_used = _t_from_window(cleaned, confidence, horizon, nu)
    if nu is None and nu_used <= _NU_WARN_THRESHOLD:
        logger.warning(
            "estimated nu=%.3f <= %.1f (very heavy tail; estimation is "
            "unreliable in short windows)",
            nu_used,
            _NU_WARN_THRESHOLD,
        )
    assert es >= var, "ES must be >= VaR"

    return ParamVaRResult(
        var=var,
        es=es,
        confidence=confidence,
        horizon=horizon,
        method=METHOD_T,
        n_obs=int(len(cleaned)),
        mu=mu,
        sigma=sigma,
        nu=nu_used,
        start=cleaned.index[0],
        end=cleaned.index[-1],
    )


# ---------------------------------------------------------------------------
# Rolling
# ---------------------------------------------------------------------------
def rolling_parametric_var(
    returns: pd.Series,
    method: str = "normal",
    window: int = 500,
    confidence: float = 0.99,
    horizon: int = 1,
) -> pd.DataFrame:
    """Rolling parametric VaR + ES with NO lookahead.

    The row at date ``t = index[i]`` is computed from ``iloc[i-window:i]`` only
    (dates ``t-window .. t-1``); ``t`` is excluded. Columns ``['var', 'es']``,
    first row at ``index[window]``, length ``n_obs - window``.

    MIN_OBS floor (decision): the floor is applied to ``window`` itself, NOT to
    ``window // horizon``. Parametric estimation uses all ``window`` observations
    regardless of horizon (sqrt scaling, no block compression), so the effective
    sample size is ``window``. (Contrast rolling_historical_var, which compresses
    into ``window // horizon`` non-overlapping blocks.)

    For ``method="t"`` the per-window nu<=5 heavy-tail warning is suppressed
    (one log line per window otherwise). A window whose free-scale MLE returns
    nu<=2 (infinite variance) is undefined for the t closed form: that row's
    ``var`` and ``es`` are set to NaN and the series continues, rather than
    aborting or silently clamping. (Point-in-time ``parametric_t_var`` still
    raises for nu<=2; only rolling degrades to NaN.)

    Minimum safe window (empirical, P(nu<=2) over the windows of a series):
      * Gaussian returns: ~0% even at window=30 — normal small-sample noise does
        NOT trigger it.
      * Heavy-tailed returns (Student-t df=3): ~9% at window=30, ~4% at
        window=100, and ~0% at window>=250.
    Use **window >= 250** for robust Student-t rolling estimates; the default
    window=500 showed zero aborts even on df=3 data. The real triggers are short
    windows on heavy-tailed data, or a single extreme outlier dominating a window.
    """
    if method not in _VALID_METHODS:
        raise ValueError(
            f"method must be one of {sorted(_VALID_METHODS)}, got {method!r}"
        )
    _validate(confidence, horizon)
    if window < 1:
        raise ValueError(f"window must be a positive integer, got {window}")
    _require_min_obs(window, confidence, label="rolling_parametric_var(window)")

    cleaned = _clean(returns)
    n = len(cleaned)
    if n <= window:
        raise ValueError(f"need more than window={window} observations; got {n}")

    dates: list[pd.Timestamp] = []
    var_values: list[float] = []
    es_values: list[float] = []
    for i in range(window, n):
        prior = cleaned.iloc[i - window : i]  # strictly before t = index[i]
        if method == "normal":
            var, es, _mu, _sigma = _normal_from_window(prior, confidence, horizon)
        else:
            # A window whose MLE yields nu<=2 (infinite variance) is undefined
            # for the t closed form. Rather than aborting the whole series, mark
            # that row NaN and continue. This is honest (no clamping/inflation)
            # and keeps the rest of the series usable. Point-in-time
            # parametric_t_var still raises — only rolling degrades gracefully.
            try:
                var, es, _mu, _sigma, _nu = _t_from_window(
                    prior, confidence, horizon, None
                )
            except ValueError:
                var, es = float("nan"), float("nan")
        dates.append(cleaned.index[i])
        var_values.append(var)
        es_values.append(es)

    index = pd.Index(dates, name=cleaned.index.name)
    return pd.DataFrame({"var": var_values, "es": es_values}, index=index)
