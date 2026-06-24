"""VaR backtesting: Kupiec POF, Christoffersen independence + conditional
coverage, and the Basel traffic-light zone classifier.

Conventions (see CLAUDE.md):
  * A violation is a loss strictly exceeding VaR: ``-return > var`` (strict).
  * ``var_series`` holds POSITIVE loss magnitudes. A negative value is a likely
    caller sign error: it is logged (warning) but not raised.
  * LR statistics are exact log-likelihood ratios (no approximations); p-values
    use ``scipy.stats.chi2.sf`` (survival function), df=1 for POF/independence,
    df=2 for conditional coverage.
  * Results are reported raw — empirical violation rate and raw LR statistics,
    never adjusted, smoothed, or rounded.
  * Aligned NaN pairs are dropped (count logged) before any statistic; the
    reduced count is reflected in ``n_obs``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import pandas as pd
from scipy.stats import chi2

logger = logging.getLogger(__name__)

_MIN_OBS = 30
_BASEL_WINDOW = 250
_BASEL_GREEN_MAX = 4  # 0-4 violations: green
_BASEL_YELLOW_MAX = 9  # 5-9: yellow; 10+: red


@dataclass
class BacktestResult:
    n_obs: int
    n_violations: int
    violation_rate: float  # x / T
    expected_rate: float  # 1 - confidence
    lr_pof: float  # Kupiec POF LR statistic
    pvalue_pof: float
    lr_ind: float  # Christoffersen independence LR
    pvalue_ind: float
    lr_cc: float  # lr_pof + lr_ind
    pvalue_cc: float
    confidence: float
    method: str
    basel_zone: str  # "green"/"yellow"/"red" (250-day only); "n/a" otherwise


def _xlogx_term(count: int, prob: float) -> float:
    """Return ``count * log(prob)`` with the convention 0*log(0) = 0.

    Only ``count == 0`` can pair with ``prob == 0`` for valid (MLE/null)
    probabilities, so returning 0 whenever ``count == 0`` is sufficient to avoid
    log(0); a positive count always has a positive probability here.
    """
    if count == 0:
        return 0.0
    return count * math.log(prob)


def _lr_pof(x: int, T: int, p: float) -> float:
    """Kupiec proportion-of-failures LR statistic (~chi2, df=1).

    ``x`` violations in ``T`` observations against null violation rate ``p``.
    At x=0 this is ``-2T*ln(1-p)`` and at x=T it is ``-2T*ln(p)`` — both strictly
    positive (NOT zero); the 0*log(0) convention only zeroes the empty count's
    own term, it does not zero the statistic.
    """
    pi = x / T
    ll_null = _xlogx_term(x, p) + _xlogx_term(T - x, 1 - p)
    ll_alt = _xlogx_term(x, pi) + _xlogx_term(T - x, 1 - pi)
    return -2.0 * (ll_null - ll_alt)


def _transition_counts(viol: pd.Series) -> tuple[int, int, int, int]:
    """Count consecutive (prev, cur) state transitions: (n00, n01, n10, n11)."""
    v = viol.to_numpy().astype(int)
    prev, cur = v[:-1], v[1:]
    n00 = int(((prev == 0) & (cur == 0)).sum())
    n01 = int(((prev == 0) & (cur == 1)).sum())
    n10 = int(((prev == 1) & (cur == 0)).sum())
    n11 = int(((prev == 1) & (cur == 1)).sum())
    return n00, n01, n10, n11


def _lr_ind(counts: tuple[int, int, int, int]) -> float:
    """Christoffersen independence LR statistic (~chi2, df=1).

    Degenerate case: when ``n10 + n11 == 0`` (no observation follows a violation,
    e.g. zero violations) the first-order Markov alternative has no estimable
    transition probability out of the violation state, so it is indistinguishable
    from the null and the test contributes no information — LR_ind = 0. The
    symmetric all-violations case (``n00 + n01 == 0``), and a degenerate overall
    rate (``pi`` of 0 or 1), are handled the same way.
    """
    n00, n01, n10, n11 = counts
    denom0 = n00 + n01
    denom1 = n10 + n11
    total = denom0 + denom1
    if denom1 == 0 or denom0 == 0 or total == 0:
        return 0.0

    pi01 = n01 / denom0
    pi11 = n11 / denom1
    pi = (n01 + n11) / total
    # Exact float equality is safe here: pi is an integer ratio that equals 0.0
    # or 1.0 only when the numerator is 0 or == total, both exact in IEEE-754.
    # Near-degenerate values (e.g. 8/9) are correctly passed through.
    if pi in (0.0, 1.0):
        return 0.0

    ll_restricted = _xlogx_term(n00 + n10, 1 - pi) + _xlogx_term(n01 + n11, pi)
    ll_unrestricted = (
        _xlogx_term(n00, 1 - pi01)
        + _xlogx_term(n01, pi01)
        + _xlogx_term(n10, 1 - pi11)
        + _xlogx_term(n11, pi11)
    )
    return -2.0 * (ll_restricted - ll_unrestricted)


def _basel_zone(n_violations: int, n_obs: int) -> str:
    """Basel traffic-light zone; only defined for a 250-day window."""
    if n_obs != _BASEL_WINDOW:
        return "n/a"
    if n_violations <= _BASEL_GREEN_MAX:
        return "green"
    if n_violations <= _BASEL_YELLOW_MAX:
        return "yellow"
    return "red"


def _drop_aligned_nan(
    returns: pd.Series, var_series: pd.Series
) -> tuple[pd.Series, pd.Series]:
    mask = returns.isna() | var_series.isna()
    n_drop = int(mask.sum())
    if n_drop:
        logger.warning("Dropping %d aligned NaN pair(s) before backtest", n_drop)
    keep = ~mask
    return returns[keep], var_series[keep]


def violations(returns: pd.Series, var_series: pd.Series) -> pd.Series:
    """Boolean series, True where the loss strictly exceeds VaR (-return > var).

    Raises ValueError if the two indices are not identical. A negative value in
    ``var_series`` (VaR should be a positive loss magnitude) is logged as a
    possible caller sign error but does not raise.
    """
    if not returns.index.equals(var_series.index):
        raise ValueError(
            "returns and var_series must be date-aligned (identical index)"
        )
    if (var_series < 0).any():
        logger.warning(
            "var_series contains negative values; VaR should be positive loss "
            "magnitudes (possible caller sign error)"
        )
    return -returns > var_series


def backtest_var(
    returns: pd.Series,
    var_series: pd.Series,
    confidence: float = 0.99,
    method: str = "unknown",
) -> BacktestResult:
    """Backtest a rolling VaR series against realized returns.

    ``returns`` and ``var_series`` must share an identical index (no auto-
    intersection). Aligned NaN pairs are dropped (logged). Raises ValueError for
    an index mismatch, confidence outside (0, 1), or fewer than 30 observations
    after cleaning.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    if not returns.index.equals(var_series.index):
        raise ValueError(
            "returns and var_series must be date-aligned (identical index)"
        )

    returns, var_series = _drop_aligned_nan(returns, var_series)
    T = len(returns)
    if T < _MIN_OBS:
        raise ValueError(
            f"insufficient observations for reliable LR test: "
            f"need >= {_MIN_OBS}, got {T}"
        )

    viol = violations(returns, var_series)
    x = int(viol.sum())
    p = 1.0 - confidence

    lr_pof = _lr_pof(x, T, p)
    lr_ind = _lr_ind(_transition_counts(viol))
    lr_cc = lr_pof + lr_ind

    return BacktestResult(
        n_obs=T,
        n_violations=x,
        violation_rate=x / T,
        expected_rate=p,
        lr_pof=lr_pof,
        pvalue_pof=float(chi2.sf(lr_pof, df=1)),
        lr_ind=lr_ind,
        pvalue_ind=float(chi2.sf(lr_ind, df=1)),
        lr_cc=lr_cc,
        pvalue_cc=float(chi2.sf(lr_cc, df=2)),
        confidence=confidence,
        method=method,
        basel_zone=_basel_zone(x, T),
    )
