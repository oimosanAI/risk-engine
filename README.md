# Risk Engine

A portfolio risk engine computing Value-at-Risk (VaR) and Expected Shortfall (ES)
by historical simulation. The guiding principles are: **verifiable over impressive**
(every number reproducible from raw data), **honest about limitations** (assumptions
and failure modes stated, not hidden), **out-of-sample discipline** (any estimate at
time $t$ is computable from data strictly before $t$, never after), and **do not
inflate** (the empirical result is reported as-is; no conservative fudge, no clamping,
no padding — a number that looks too good is a bug until proven otherwise).

---

## Implemented

`src/var/historical.py` — historical-simulation VaR + ES, rolling estimates,
portfolio return aggregation.

All other modules (`parametric.py`, `monte_carlo.py`, `stress/`, `backtest/`,
`report/`) are scaffolded but not yet implemented.

---

## Mathematical definitions

### Sign convention

Daily returns are arithmetic (simple). Losses are defined as negated returns:

$$L_t = -r_t$$

VaR and ES are reported as **positive loss magnitudes**. A return of $-5\%$ becomes
a loss of $+0.05$.

### Value-at-Risk

$$\text{VaR}_c = Q_c(L)$$

where $Q_c$ is the $c$-quantile of the empirical loss distribution. Equivalently,
$\text{VaR}_c$ is the $(1-c)$ lower-tail quantile of returns, sign-flipped.  
At $c = 0.99$, the loss is exceeded on at most $1\%$ of historical days.

When the sample contains no losses (all returns positive), $\text{VaR}_c \leq 0$.
This is not clamped to zero — it faithfully reports that the historical tail contains
no real loss at that confidence level.

### Expected Shortfall

$$\text{ES}_c = \mathbb{E}[L \mid L \geq \text{VaR}_c]$$

the mean loss in the tail at and beyond VaR. The boundary is $\geq$, not $>$, so
the VaR observation itself belongs to the tail. The invariant $\text{ES}_c \geq
\text{VaR}_c$ holds unconditionally.

### Historical simulation vs. parametric

Historical simulation uses the **empirical quantile** of realised losses directly,
making no distributional assumption. Fat tails, skew, and regime structure are
captured to the extent they appear in the sample. The forthcoming parametric module
(`parametric.py`) will fit a normal or Student-$t$ distribution and compute quantiles
analytically — a faster estimate, but one whose accuracy depends on how well the
assumed distribution matches reality. Which is more appropriate depends on the tail
behaviour of the asset and the length of the available history.

### Multi-day horizon

For horizon $h > 1$, returns are compounded over **non-overlapping** blocks:

$$r^{(h)}_k = \prod_{j=0}^{h-1}(1 + r_{kh+j}) - 1$$

Trailing observations that do not fill a complete block are discarded.

The common alternative, $\text{VaR}_{c,h} \approx \sqrt{h}\,\text{VaR}_{c,1}$, is
**not used**. The square-root-of-time scaling assumes i.i.d. returns. It breaks under
fat tails (where tail mass decays more slowly than $\sqrt{h}$ implies) and under
autocorrelation (common in volatility-clustered series). Non-overlapping compounding
is slower but honest.

### Quantile interpolation

The interpolation method passed to `np.quantile` is an explicit parameter
(default `"linear"`). It is never hardcoded. At extreme confidences on small samples,
`"lower"` and `"higher"` give the same result as adjacent order statistics;
`"linear"` interpolates between them. The choice affects the reported number — use
the same method consistently when comparing results.

---

## No-lookahead guarantee

The rolling estimate indexed at date $t$ is computed from
`returns.iloc[i-window:i]` — the `window` observations at positions
$i{-}\text{window}, \ldots, i{-}1$. The upper bound is exclusive: position $i$
(date $t$ itself) is never read.

This is tested by two complementary guards that run in the test suite:

**Mutation test** — after computing the rolling result, returns at all dates
$\geq t$ (including $t$ itself) are shocked by a large constant. The rolling row
at $t$ must be byte-identical before and after. Mutating only dates strictly *after*
$t$ is not sufficient: a bare `.rolling(window)` aggregate — a common mistake —
reads the current row and would pass that weaker check but fail this one.

**Equivalence test** — `rolling_historical_var` and `historical_var` route through
the same private numeric core (`_var_es_from_returns`). For each tested date $t$,
`historical_var(returns.iloc[i-window:i], ...)` is called independently on the
exact prior window and asserted to be numerically identical to the rolling row at
$t$ for both `var` and `es`, across multiple dates and confidence levels.

---

## Minimum observations and the unresolvable-tail rule

$$\text{MIN\_OBS}(c) = \left\lceil \frac{1}{1-c} \right\rceil$$

At $c = 0.99$, the tail probability is $1\%$, requiring at least 100 observations
for the empirical distribution to place a single point in the tail.  Below this
threshold the $(1-c)$ quantile is unresolvable from the data — the reported number
would be pure extrapolation dressed as an empirical estimate. Both `historical_var`
and `rolling_historical_var` raise `ValueError` when the available (or
effective, after horizon compression) sample falls below this floor.

For rolling estimates with horizon $h > 1$, the floor applies to the
**effective** per-window count after compression:

$$\text{effective} = \lfloor \text{window} / h \rfloor \geq \text{MIN\_OBS}(c)$$

---

## Limitations

**Empirical VaR cannot extrapolate beyond the historical worst loss.** The
$(1-c)$ quantile of $n$ observations is bounded by the minimum realised loss.
At $c = 0.999$ on a 500-day window, the estimate rests on fewer than one
expected tail observation and is highly unstable.

**Tail estimates have high sampling error.** Quantiles in the far tail converge
slowly. At $c = 0.99$ with 250 daily observations, the standard error of the
historical VaR estimate is of the same order as the estimate itself. This is not
a defect of the implementation; it is a property of fat-tailed data and small
samples. The parametric and Monte Carlo modules (forthcoming) trade this
sampling error for distributional assumption error.

**ES has higher estimation variance than VaR** by construction — it averages
over a subset of the tail rather than reading a single quantile. ES is the more
coherent risk measure (sub-additive, convex), but expect wider confidence
intervals in backtesting.

**Synthetic data only.** `loader.py` is not yet implemented. All tests use
seeded numpy normal and Student-$t$ series. No market data has been validated.
Once `loader.py` exists, note that the data source (yfinance) carries its own
limitations: no survivorship-bias correction, dividend and split adjustment is
provider-dependent, and holiday/trading-gap handling affects return continuity.
Every downstream risk estimate inherits these data-quality bounds.

**Window choice is not prescribed.** Basel III uses 250 trading days; the
implementation default is 500. Neither is objectively correct. A longer window
stabilises estimates but makes the model slower to respond to changing volatility
regimes. A shorter window is more responsive but noisier. The caller is
responsible for this choice.

---

## Usage

`loader.py` is not yet implemented; generate synthetic returns directly.

```python
import numpy as np
import pandas as pd
from src.var.historical import (
    VaRResult,
    historical_var,
    rolling_historical_var,
    portfolio_returns,
)

rng = np.random.default_rng(42)
dates = pd.date_range("2020-01-01", periods=1000, freq="B")
returns = pd.Series(rng.standard_normal(1000) * 0.01, index=dates)

# Single-window VaR + ES
result: VaRResult = historical_var(returns, confidence=0.99, horizon=1)
print(f"VaR(99%): {result.var:.4f}   ES(99%): {result.es:.4f}")
print(f"n_obs={result.n_obs}  {result.start.date()} to {result.end.date()}")

# Rolling estimates (500-day window, no lookahead)
roll = rolling_historical_var(returns, window=500, confidence=0.99)
print(roll.tail())          # columns: ['var', 'es'], date index

# Multi-asset portfolio
asset_returns = pd.DataFrame(
    {"SPY": rng.standard_normal(1000) * 0.01,
     "TLT": rng.standard_normal(1000) * 0.005},
    index=dates,
)
port = portfolio_returns(asset_returns, {"SPY": 0.6, "TLT": 0.4})
port_var = historical_var(port, confidence=0.99)
```

VaR and ES are **positive loss magnitudes** in return units. A `var` of `0.0231`
means a 2.31% loss is exceeded on at most 1% of historical days in the sample.

---

## Development

```bash
# Install runtime deps
pip install -r requirements.txt

# Install dev/test tooling (separate from runtime)
pip install -r requirements-dev.txt

# Run tests
pytest tests/var/test_historical.py -v

# Coverage (must remain ≥ 80%; currently 100%)
pytest tests/var/test_historical.py --cov=src.var.historical --cov-report=term-missing

# Lint and format
ruff check src/
black src/ tests/
```
