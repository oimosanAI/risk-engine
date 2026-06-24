"""Historical-simulation Value-at-Risk (VaR) and Expected Shortfall (ES).

Conventions (do not violate):
  * ``loss = -return``. VaR and ES are reported as POSITIVE loss magnitudes in
    return units.
  * VaR at confidence ``c`` is the ``c``-quantile of the loss distribution,
    i.e. the ``(1 - c)`` lower-tail quantile of returns, sign-flipped.
  * ES is the mean loss in the tail at or beyond VaR; ``es >= var`` always.
  * Rolling estimates use ONLY observations strictly before the labelled date
    (no lookahead): the row at date ``t`` reads ``t-window .. t-1``.
  * Empirical numbers are reported as-is. We do not inflate, pad, or apply
    conservative fudge factors. With no losses in the sample, VaR may be <= 0;
    it is never clamped to zero.
  * NaNs are dropped explicitly and the dropped count is logged.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

# _clean and _min_obs are shared, distribution-agnostic helpers; they live in
# _utils so sibling estimator modules depend on a neutral module, not on each
# other. Re-exported here so existing `src.var.historical._clean/_min_obs`
# references keep resolving.
from src.var._utils import _clean, _min_obs

logger = logging.getLogger(__name__)

METHOD = "historical"
_VALID_INTERP = frozenset({"linear", "lower", "higher", "nearest", "midpoint"})
_WEIGHT_SUM_TOL = 1e-6


@dataclass
class VaRResult:
    var: float  # positive loss magnitude, return units
    es: float  # expected shortfall, >= var
    confidence: float
    horizon: int
    method: str  # "historical"
    n_obs: int
    start: pd.Timestamp
    end: pd.Timestamp


def _validate_params(confidence: float, interpolation: str, horizon: int) -> None:
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    if interpolation not in _VALID_INTERP:
        raise ValueError(
            f"interpolation must be one of {sorted(_VALID_INTERP)}, "
            f"got {interpolation!r}"
        )
    if horizon < 1:
        raise ValueError(f"horizon must be a positive integer, got {horizon}")


def _aggregate_horizon(returns: pd.Series, horizon: int) -> pd.Series:
    """Build NON-OVERLAPPING ``horizon``-day compounded returns ``prod(1+r) - 1``.

    A trailing partial block (fewer than ``horizon`` observations) is dropped.
    No sqrt(h) scaling is used: that iid approximation breaks under fat tails
    and autocorrelation.
    """
    if horizon == 1:
        return returns
    n_blocks = len(returns) // horizon
    if n_blocks == 0:
        return returns.iloc[0:0]
    trimmed = returns.iloc[: n_blocks * horizon]
    blocks = trimmed.to_numpy().reshape(n_blocks, horizon)
    compounded = np.prod(1.0 + blocks, axis=1) - 1.0
    block_index = trimmed.index[horizon - 1 :: horizon]  # last date of each block
    return pd.Series(compounded, index=block_index)


def _var_es_from_returns(
    returns: np.ndarray, confidence: float, interpolation: str
) -> tuple[float, float]:
    """Pure numeric core: empirical VaR and ES as positive loss magnitudes.

    ``var`` is the ``confidence``-quantile of losses (``-returns``). ``es`` is the
    mean of losses at or beyond ``var``.

    The ``else var`` fallback is an unreachable defensive guard: ``np.quantile``
    never exceeds ``max(losses)``, so ``losses >= var`` always selects at least
    the maximum loss and ``tail`` is never empty. It is marked ``no cover`` so the
    coverage number stays honest rather than counting an unreachable branch.
    """
    losses = -np.asarray(returns, dtype=float)
    var = float(np.quantile(losses, confidence, method=interpolation))
    tail = losses[losses >= var]
    if tail.size:
        es = float(tail.mean())
    else:  # pragma: no cover - unreachable: np.quantile never exceeds max(losses)
        es = var
    return var, es


def historical_var(
    returns: pd.Series,
    confidence: float = 0.99,
    horizon: int = 1,
    interpolation: str = "linear",
) -> VaRResult:
    """Single-sample historical VaR + ES.

    VaR/ES are positive loss magnitudes; ``es >= var``. NaNs are dropped (logged).
    Raises ``ValueError`` for confidence outside (0, 1), an unknown interpolation
    method, a non-positive horizon, or a sample smaller than ``_min_obs(confidence)``.
    """
    _validate_params(confidence, interpolation, horizon)
    cleaned = _clean(returns)
    sample = _aggregate_horizon(cleaned, horizon)

    floor = _min_obs(confidence)
    if len(sample) < floor:
        raise ValueError(
            f"need at least {floor} observation(s) for confidence={confidence} "
            f"(horizon={horizon}); got {len(sample)}"
        )

    var, es = _var_es_from_returns(sample.to_numpy(), confidence, interpolation)
    # Invariant guaranteed by construction; assert documents intent.
    assert es >= var, "ES must be >= VaR"

    return VaRResult(
        var=var,
        es=es,
        confidence=confidence,
        horizon=horizon,
        method=METHOD,
        n_obs=int(len(sample)),
        start=cleaned.index[0],
        end=cleaned.index[-1],
    )


def rolling_historical_var(
    returns: pd.Series,
    window: int = 500,
    confidence: float = 0.99,
    horizon: int = 1,
    interpolation: str = "linear",
) -> pd.DataFrame:
    """Rolling historical VaR + ES with NO lookahead.

    The row labelled at date ``t = index[i]`` is computed from ``iloc[i-window:i]``
    only (dates ``t-window .. t-1``); ``t`` itself is excluded. Returns a DataFrame
    with columns ``['var', 'es']`` indexed by date, the first row at ``index[window]``
    and length ``n_obs - window``.
    """
    _validate_params(confidence, interpolation, horizon)
    if window < 1:
        raise ValueError(f"window must be a positive integer, got {window}")

    # Same unresolvable-tail rule as historical_var, applied to the EFFECTIVE
    # per-window sample after horizon compression (non-overlapping blocks).
    floor = _min_obs(confidence)
    effective = window // horizon
    if effective < floor:
        raise ValueError(
            f"window={window} with horizon={horizon} yields only "
            f"{effective} effective observation(s) per window, but "
            f"confidence={confidence} needs at least {floor} "
            f"(the (1-confidence) tail is otherwise unresolvable)"
        )

    cleaned = _clean(returns)
    n = len(cleaned)
    if n <= window:
        raise ValueError(f"need more than window={window} observations; got {n}")

    dates: list[pd.Timestamp] = []
    var_values: list[float] = []
    es_values: list[float] = []
    for i in range(window, n):
        prior = cleaned.iloc[i - window : i]  # strictly before t = index[i]
        sample = _aggregate_horizon(prior, horizon)
        var, es = _var_es_from_returns(sample.to_numpy(), confidence, interpolation)
        dates.append(cleaned.index[i])
        var_values.append(var)
        es_values.append(es)

    index = pd.Index(dates, name=cleaned.index.name)
    return pd.DataFrame({"var": var_values, "es": es_values}, index=index)


def portfolio_returns(asset_returns: pd.DataFrame, weights) -> pd.Series:
    """Daily-rebalanced portfolio return series from per-asset returns and weights.

    ``weights`` may be an ``np.ndarray`` (aligned to column order) or a
    ``dict[str, float]`` (keys must match columns exactly). Weights are used as
    given: a non-unit sum is warned about, never re-normalized. The portfolio
    return is the matrix product ``asset_returns @ weights``, so a NaN in ANY
    asset propagates to that day; such days are dropped (count logged), mirroring
    the single-series NaN policy rather than masking them via skipna.
    """
    columns = list(asset_returns.columns)
    if isinstance(weights, dict):
        unknown = set(weights) - set(columns)
        missing = set(columns) - set(weights)
        if unknown or missing:
            raise ValueError(
                "weight keys must match columns exactly; "
                f"unknown={sorted(unknown)}, missing={sorted(missing)}"
            )
        w = np.array([float(weights[col]) for col in columns], dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
        if w.ndim != 1 or w.shape[0] != asset_returns.shape[1]:
            raise ValueError(
                f"weights must be 1-D with one entry per asset "
                f"({asset_returns.shape[1]}), got shape {w.shape}"
            )

    if not np.all(np.isfinite(w)):
        raise ValueError(f"weights must all be finite (no NaN/inf), got {w}")

    total = float(w.sum())
    if abs(total - 1.0) > _WEIGHT_SUM_TOL:
        logger.warning(
            "portfolio weights sum to %.6f, not 1.0; using as-is (no re-normalization)",
            total,
        )

    raw = asset_returns.to_numpy(dtype=float) @ w  # NaN in any asset -> NaN day
    series = pd.Series(raw, index=asset_returns.index)

    n_nan = int(series.isna().sum())
    if n_nan:
        logger.warning("Dropping %d row(s) with missing asset returns", n_nan)
        series = series.dropna()
    return series
