"""Tests for the data layer (src/data/loader.py).

All tests are OFFLINE: fetch_prices' network call (yfinance.download) is
monkeypatched. The real network path is exercised only by the integration test
in tests/backtest/test_var_backtest.py.
"""

import logging

import numpy as np
import pandas as pd
import pytest

from src.data import loader
from src.data.loader import fetch_prices, prices_to_returns


def _dates(n: int, start: str = "2020-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="B")


# ---------------------------------------------------------------------------
# prices_to_returns
# ---------------------------------------------------------------------------
def test_prices_to_returns_simple():
    prices = pd.DataFrame({"A": [100.0, 101.0, 99.0]}, index=_dates(3))

    out = prices_to_returns(prices, method="simple")

    assert len(out) == 2  # first NaN row dropped
    assert out["A"].iloc[0] == pytest.approx(0.01)
    assert out["A"].iloc[1] == pytest.approx(99.0 / 101.0 - 1.0)


def test_prices_to_returns_log():
    prices = pd.DataFrame({"A": [100.0, 110.0]}, index=_dates(2))

    out = prices_to_returns(prices, method="log")

    assert len(out) == 1
    assert out["A"].iloc[0] == pytest.approx(np.log(110.0 / 100.0))


def test_prices_to_returns_first_nan_dropped():
    prices = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0]}, index=_dates(4))
    out = prices_to_returns(prices, method="simple")
    assert not out.isna().any().any()
    assert out.index[0] == prices.index[1]


def test_prices_to_returns_unknown_method_raises():
    prices = pd.DataFrame({"A": [1.0, 2.0]}, index=_dates(2))
    with pytest.raises(ValueError):
        prices_to_returns(prices, method="geometric")


# ---------------------------------------------------------------------------
# fetch_prices (yfinance.download monkeypatched)
# ---------------------------------------------------------------------------
def _multi_frame(tickers, n=10):
    idx = _dates(n)
    cols = pd.MultiIndex.from_product([["Close", "Volume"], tickers])
    data = np.random.default_rng(0).uniform(50, 150, size=(n, len(cols)))
    return pd.DataFrame(data, index=idx, columns=cols)


def _single_frame(n=10):
    idx = _dates(n)
    return pd.DataFrame(
        {
            "Open": np.linspace(100, 110, n),
            "Close": np.linspace(101, 111, n),
            "Volume": np.ones(n),
        },
        index=idx,
    )


def test_fetch_prices_multi_ticker_shape(monkeypatch):
    monkeypatch.setattr(
        loader.yf, "download", lambda *a, **k: _multi_frame(["SPY", "QQQ"])
    )
    out = fetch_prices(["SPY", "QQQ"], "2020-01-01", "2020-02-01")
    assert list(out.columns) == ["SPY", "QQQ"]
    assert len(out) == 10


def test_fetch_prices_single_ticker_shape(monkeypatch):
    monkeypatch.setattr(loader.yf, "download", lambda *a, **k: _single_frame())
    out = fetch_prices("SPY", "2020-01-01", "2020-02-01")  # bare string accepted
    assert list(out.columns) == ["SPY"]
    assert out["SPY"].iloc[0] == pytest.approx(101.0)


def test_fetch_prices_empty_raises(monkeypatch):
    monkeypatch.setattr(loader.yf, "download", lambda *a, **k: pd.DataFrame())
    with pytest.raises(ValueError):
        fetch_prices(["SPY"], "2020-01-01", "2020-02-01")


def test_fetch_prices_ticker_all_nan_raises(monkeypatch):
    # Download is non-empty (passes the raw.empty guard) but one requested
    # ticker has no usable data -> raise.
    frame = _multi_frame(["SPY", "QQQ"], n=10)
    frame[("Close", "QQQ")] = np.nan
    monkeypatch.setattr(loader.yf, "download", lambda *a, **k: frame)
    with pytest.raises(ValueError):
        fetch_prices(["SPY", "QQQ"], "2020-01-01", "2020-02-01")


def test_fetch_prices_missing_rows_warns(monkeypatch, caplog):
    frame = _multi_frame(["SPY", "QQQ"], n=20)
    frame.loc[frame.index[:3], ("Close", "QQQ")] = np.nan  # 15% missing for QQQ

    monkeypatch.setattr(loader.yf, "download", lambda *a, **k: frame)
    with caplog.at_level(logging.WARNING):
        out = fetch_prices(["SPY", "QQQ"], "2020-01-01", "2020-02-01")

    assert "QQQ" in caplog.text
    assert "missing" in caplog.text.lower()
    assert list(out.columns) == ["SPY", "QQQ"]
