"""Shared, distribution-agnostic helpers for the VaR modules.

These live here (rather than inside any one estimator module) so that sibling
estimators — historical, parametric, monte_carlo — depend on a neutral utility
module instead of importing each other's private names. Keeps the modules peers
with low coupling.
"""

from __future__ import annotations

import logging
import math

import pandas as pd

# Logger name is src.var._utils; NaN-drop and violation sign warnings from
# historical, parametric, var_backtest and report all appear under this name
# (these shared helpers live here so callers don't import each other's modules).
logger = logging.getLogger(__name__)


def _min_obs(confidence: float) -> int:
    """Minimum sample size for the ``(1 - c)`` tail to contain >= 1 observation."""
    return math.ceil(1.0 / (1.0 - confidence))


def violations(returns: pd.Series, var_series: pd.Series) -> pd.Series:
    """Boolean series, True where the loss strictly exceeds VaR (-return > var).

    The single source of truth for the violation definition, shared by
    var_backtest (statistics) and report.plots (visualization) so the two can
    never diverge. Pure pandas — no scipy — so the report layer can import it
    without pulling the backtest module's scipy dependency.

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


def _clean(returns: pd.Series) -> pd.Series:
    """Drop NaNs, logging the dropped count. Never silently swallow them."""
    n_nan = int(returns.isna().sum())
    if n_nan:
        logger.warning("Dropping %d NaN observation(s) from return series", n_nan)
        returns = returns.dropna()
    return returns
