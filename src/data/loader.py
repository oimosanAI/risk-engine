"""Pure data layer: download prices and convert to returns.

No VaR / risk logic lives here. The only responsibilities are fetching daily
prices (via yfinance) and turning them into a clean returns DataFrame.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

_MISSING_WARN_FRACTION = 0.05  # warn if a ticker is >5% missing after alignment


def fetch_prices(
    tickers: list[str],
    start: str,
    end: str,
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """Download daily close prices via yfinance.

    With ``auto_adjust=True`` the returned close is split/dividend adjusted.
    A bare string ticker is accepted and treated as a single-element list.

    Raises ValueError if the download is empty or any requested ticker returns
    no data. Logs a warning for any ticker with more than 5% missing rows after
    aligning to the common date index.
    """
    if isinstance(tickers, str):
        tickers = [tickers]

    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=auto_adjust,
        group_by="column",  # field on column level 0, so raw["Close"] is reliable
        progress=False,
    )
    if raw.empty:
        raise ValueError(f"no data returned for tickers={tickers} ({start}..{end})")

    # Normalize the two documented yfinance shapes to a ticker-keyed close frame.
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].copy()
    else:
        close = raw[["Close"]].copy()
        close.columns = [tickers[0]]

    for ticker in tickers:
        if ticker not in close.columns or close[ticker].dropna().empty:
            raise ValueError(f"ticker {ticker!r} returned empty data")

    close = close.reindex(columns=tickers)
    aligned = close.dropna(how="all")

    n = len(aligned)
    for ticker in tickers:
        missing = int(aligned[ticker].isna().sum())
        if n and missing / n > _MISSING_WARN_FRACTION:
            logger.warning(
                "ticker %s has %.1f%% missing rows after alignment",
                ticker,
                100.0 * missing / n,
            )

    return aligned


def prices_to_returns(
    prices: pd.DataFrame,
    method: str = "simple",
) -> pd.DataFrame:
    """Compute daily returns from a price DataFrame.

    ``simple`` uses ``pct_change()``; ``log`` uses ``diff(log(price))``. The
    leading NaN row (no prior price) is dropped. Raises ValueError for an
    unrecognized method.
    """
    if method == "simple":
        returns = prices.pct_change()
    elif method == "log":
        returns = np.log(prices).diff()
    else:
        raise ValueError(f"method must be 'simple' or 'log', got {method!r}")

    return returns.iloc[1:]  # drop the leading NaN row
