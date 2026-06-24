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

# Logger name is src.var._utils; NaN-drop warnings from both historical and
# parametric appear under this name (these helpers moved here from historical).
logger = logging.getLogger(__name__)


def _min_obs(confidence: float) -> int:
    """Minimum sample size for the ``(1 - c)`` tail to contain >= 1 observation."""
    return math.ceil(1.0 / (1.0 - confidence))


def _clean(returns: pd.Series) -> pd.Series:
    """Drop NaNs, logging the dropped count. Never silently swallow them."""
    n_nan = int(returns.isna().sum())
    if n_nan:
        logger.warning("Dropping %d NaN observation(s) from return series", n_nan)
        returns = returns.dropna()
    return returns
