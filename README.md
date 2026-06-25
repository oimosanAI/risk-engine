# Risk Engine

![Python](https://img.shields.io/badge/python-3.11+-blue.svg) ![License: MIT](https://img.shields.io/badge/license-MIT-green.svg) ![tests](https://img.shields.io/badge/tests-144%20passing-brightgreen.svg) ![ruff](https://img.shields.io/badge/lint-ruff-blue.svg) ![code style](https://img.shields.io/badge/code%20style-black-black.svg)

A portfolio risk engine computing Value-at-Risk (VaR) and Expected Shortfall (ES)
by historical simulation, parametric (normal / Student-$t$) closed forms, and
historical stress replay — with statistical backtesting and a monochrome reporting
layer. The guiding principles are **verifiable over impressive** (every number is
reproducible from raw data), **honest about limitations** (assumptions and failure
modes are stated, not hidden), **out-of-sample discipline** (any estimate at time
$t$ is computable from data strictly before $t$, never after), and **do not inflate**
(the empirical result is reported as-is — no conservative fudge, no clamping, no
padding; a number that looks too good is a bug until proven otherwise).

*日本語版: [README.ja.md](README.ja.md)*

---

## Modules

- **`src/data/loader.py`** — a pure data layer. `fetch_prices` downloads daily
  adjusted close via yfinance (normalising single- and multi-ticker shapes, raising
  on empty data, warning on >5% missing rows); `prices_to_returns` computes simple
  or log returns. No risk logic lives here.
- **`src/var/historical.py`** — historical-simulation VaR + ES, rolling estimates
  with a strict no-lookahead guarantee, and daily-rebalanced portfolio aggregation.
- **`src/var/parametric.py`** — parametric VaR + ES under the normal and Student-$t$
  distributions, with closed-form ES and free-scale maximum-likelihood estimation of
  the $t$ degrees of freedom.
- **`src/backtest/var_backtest.py`** — VaR backtesting: the Kupiec proportion-of-
  failures test, the Christoffersen independence test, their conditional-coverage
  combination, and the Basel traffic-light zone classifier.
- **`src/stress/scenarios.py`** — historical stress replay (2008 GFC, 2020 COVID
  presets) reporting cumulative return, maximum drawdown, recovery period, and the
  VaR violation rate over each window.
- **`src/report/plots.py`** — a monochrome, institutional reporting layer: four
  plot functions plus a single save helper, with a grayscale palette and exactly one
  accent colour reserved for violation markers.

The single violation definition (`-return > VaR`, strict) lives in `src/var/_utils.py`
and is reused by the backtest, stress, and report layers — one source of truth, so
the statistics and the visualisations can never diverge.

---

## Mathematical definitions

### Sign convention

Daily returns are arithmetic (simple). Losses are negated returns:

$$L_t = -r_t$$

VaR and ES are reported as **positive loss magnitudes**: a return of $-5\%$ is a loss
of $+0.05$.

### Value-at-Risk and Expected Shortfall

$$\text{VaR}_c = Q_c(L), \qquad \text{ES}_c = \mathbb{E}[L \mid L \geq \text{VaR}_c]$$

$Q_c$ is the $c$-quantile of the loss distribution; equivalently $\text{VaR}_c$ is the
$(1-c)$ lower-tail quantile of returns, sign-flipped. The ES boundary is $\geq$, not
$>$, so the VaR observation itself belongs to the tail; if the tail is degenerate
(empty) ES falls back to VaR. The invariant $\text{ES}_c \geq \text{VaR}_c$ holds
unconditionally. When the sample contains no losses, $\text{VaR}_c \leq 0$ is reported
as-is, never clamped to zero.

### Historical vs. parametric — the tradeoff

Historical simulation uses the **empirical quantile** of realised losses, making no
distributional assumption: it cannot extrapolate beyond the worst loss in the sample,
but it also invents nothing. Parametric VaR fits a distribution and reads the quantile
from its **closed form** — it is smooth and extrapolates beyond the worst historical
loss, but it is only as good as the assumed distribution. The **normal** model
systematically underestimates fat tails; the **Student-$t$** model captures them but
depends on a stable estimate of the degrees of freedom $\nu$, which is noisy in short
windows. A normal ES below the historical ES on fat-tailed data is the *correct* output
of a misspecified model, not a bug.

For the Student-$t$, $\sigma$ is the **actual volatility** (sample standard deviation),
and the distribution is standardized to it via scale $= \sigma\sqrt{(\nu-2)/\nu}$ so the
$t$ and normal models share variance and differ only in tail shape; $\nu$ is estimated
by free-scale MLE. Multi-day horizons scale $\mu \to \mu h$ and $\sigma \to \sigma\sqrt{h}$
(the i.i.d. closed-form scaling), in contrast to the historical module, which compounds
non-overlapping $h$-day blocks rather than scaling.

### Backtesting

With $x$ violations in $T$ observations against a null rate $p = 1-c$, the Kupiec
proportion-of-failures statistic is

$$\text{LR}_{\text{POF}} = -2\ln\frac{(1-p)^{T-x}\,p^{x}}{(1-\hat\pi)^{T-x}\,\hat\pi^{x}}, \qquad \hat\pi = \tfrac{x}{T},$$

distributed $\chi^2_1$. At $x=0$ this equals $-2T\ln(1-p) > 0$ (not zero — zero
violations when you expect some is itself informative). The Christoffersen independence
statistic $\text{LR}_{\text{ind}}$ ($\chi^2_1$) compares a first-order Markov model of
the violation sequence against an i.i.d. one, detecting clustering. Conditional coverage
combines them:

$$\text{LR}_{\text{cc}} = \text{LR}_{\text{POF}} + \text{LR}_{\text{ind}} \sim \chi^2_2.$$

P-values use the $\chi^2$ survival function.

### Drawdown and recovery

On a wealth index $W_t = \prod_{i \le t}(1+r_i)$ whose running peak is floored at the
pre-scenario capital $1.0$,

$$\text{MDD} = \max_t \frac{\text{peak}_t - W_t}{\text{peak}_t},$$

a positive magnitude. Recovery is the first date after the trough where $W_t \geq 1.0$
— back to the **pre-scenario level**, not to a within-window high — measured in calendar
days; `None` if the window never recovers.

---

## No-lookahead guarantee

This is the credibility centerpiece. The rolling estimate indexed at date $t$ is
computed from `returns.iloc[i-window:i]` — the `window` observations at positions
$i{-}\text{window}, \ldots, i{-}1$. The upper bound is **exclusive**: position $i$
(date $t$ itself) is never read. The first row lands at `index[window]` and the result
has length `n_obs - window`.

Two complementary guards enforce this in the test suite:

**Mutation test** — after computing the rolling result, returns at all dates $\geq t$
(*including $t$ itself*) are shocked by a large constant; the row at $t$ must be
unchanged. Mutating only dates strictly *after* $t$ is insufficient: a bare
`.rolling(window)` aggregate — a common mistake — reads the current row and would pass
that weaker check but fail this one.

**Equivalence test** — the rolling row at $t$ is asserted numerically identical to the
point-in-time estimator run on exactly `returns.iloc[i-window:i]`, across multiple dates
and confidence levels, because both route through the same numeric core.

The unresolvable-tail floor (`MIN_OBS` in the code) is enforced everywhere:

$$n_{\min}(c) = \left\lceil \frac{1}{1-c} \right\rceil$$

At $c = 0.99$ this is 100. Below it the $(1-c)$ quantile is pure extrapolation dressed as
an empirical estimate, and a `ValueError` is raised.

---

## Empirical results (SPY, 2005–2022)

Regenerate with `python scripts/generate_readme_figures.py`. The figures and numbers
below are from that exact run (rolling window 500, $c = 0.99$).

### VaR violations

![VaR violations](docs/images/var_violations.png)

Daily losses against the rolling 99% historical VaR; violations (loss > VaR) are marked
in the accent colour. They cluster sharply in 2008 and 2020 — precisely where an i.i.d.
model assumes they should not.

### VaR method comparison

![VaR method comparison](docs/images/var_comparison.png)

Historical, parametric-normal, and parametric-$t$ rolling VaR overlaid (grayscale shade
plus linestyle, since colour is reserved for violations). Historical VaR steps with the
empirical tail; the parametric lines are smoother but sit lower through the calm periods,
which is exactly why they over-violate.

### Stress: 2008 GFC wealth index

![GFC wealth index](docs/images/stress_wealth_gfc.png)

The wealth index over the GFC window (2008-09-01 to 2009-03-31), annotated with the
trough. The drawdown reaches **46.4%**; there is **no recovery within the window** — the
market bottomed in March 2009 and recovered later.

### Violation clustering

![Violation clustering](docs/images/violation_clustering.png)

A rug plot of violation dates, distinguishing isolated violations from consecutive-day
runs (length $\geq 2$). The clusters are what the Christoffersen independence test
detects, and why conditional coverage rejects so decisively.

### The numbers, honestly

| Method | Observations | Violation rate (expected 1.00%) | POF $p$ | Independence $p$ | Conditional coverage $p$ |
|---|---|---|---|---|---|
| Historical | 4030 | **1.94%** | $1.2\times10^{-7}$ | $1.5\times10^{-5}$ | $7.4\times10^{-11}$ |
| Parametric-normal | 4030 | **3.23%** | $1.7\times10^{-29}$ | $3.4\times10^{-6}$ | $5.1\times10^{-33}$ |
| Parametric-$t$ | 3700 | **3.59%** | $1.1\times10^{-34}$ | $8.5\times10^{-10}$ | $1.2\times10^{-41}$ |

Stress windows (VaR violation rate against the historical rolling VaR):

| Scenario | Cumulative return | Max drawdown | Recovery | VaR violation rate |
|---|---|---|---|---|
| GFC 2008 | $-36.95\%$ | $46.38\%$ | none in window | $10.96\%$ |
| COVID 2020 | $-3.17\%$ | $33.72\%$ | 77 calendar days | $10.58\%$ |

**Every model is rejected by the backtest** ($p \approx 0$ on POF, independence, and
conditional coverage), and the GFC/COVID windows violate VaR roughly 10× the nominal
rate. This is the finding, not a failure: the engine correctly demonstrates that simple
99% VaR models — historical and parametric alike — under-state risk and break down under
clustering during crises. The COVID cumulative return is small ($-3.2\%$) precisely
because the window includes the recovery; the **drawdown** ($33.7\%$) is the crisis
signal, and conflating the two is a common error this layer is built to avoid.

---

## Limitations

- **Empirical VaR cannot extrapolate beyond the worst historical loss.** The $(1-c)$
  quantile of $n$ observations is bounded by the minimum realised loss; at high $c$ on
  short windows it rests on fewer than one expected tail observation and is unstable.
- **Tail estimates have high sampling error.** Far-tail quantiles converge slowly; at
  $c=0.99$ with a few hundred observations the standard error is of the same order as the
  estimate. Parametric methods trade this sampling error for distributional-assumption
  error.
- **Data caveats (yfinance).** No survivorship-bias correction; dividend/split adjustment
  is provider-dependent; holiday and trading-gap handling affects return continuity.
  Every downstream estimate inherits these bounds.
- **Free-scale $t$ MLE can yield $\nu \leq 2$ on short or outlier-dominated windows**
  (infinite variance). Point-in-time `parametric_t_var` raises; rolling degrades that row
  to `NaN` rather than aborting the series. Use a window $\geq 250$ for stable $t$ rolling
  estimates (empirically $\sim 0\%$ failures even on heavy tails).

---

## Usage

```python
import numpy as np
import pandas as pd

from src.data.loader import fetch_prices, prices_to_returns
from src.var.historical import rolling_historical_var
from src.var.parametric import rolling_parametric_var
from src.backtest.var_backtest import backtest_var
from src.stress.scenarios import run_preset_scenarios

# Real data
prices = fetch_prices(["SPY"], "2005-01-01", "2022-12-31")
returns = prices_to_returns(prices, method="simple")["SPY"]

# Rolling 99% VaR + ES, no lookahead
roll = rolling_historical_var(returns, window=500, confidence=0.99)  # cols ['var','es']

# Backtest the rolling VaR against realised returns
aligned = returns.loc[roll.index]
result = backtest_var(aligned, roll["var"], confidence=0.99, method="historical")
print(result.violation_rate, result.pvalue_cc, result.basel_zone)

# Parametric comparison + historical stress replay
rt = rolling_parametric_var(returns, method="t", window=500, confidence=0.99)
scenarios = run_preset_scenarios(returns, var_series=roll["var"], var_method="historical")
print(scenarios["gfc_2008"].max_drawdown, scenarios["covid_2020"].recovery_days)
```

VaR and ES are **positive loss magnitudes** in return units: a `var` of `0.0231` means a
2.31% loss is exceeded on at most 1% of days in the sample.

---

## Testing

The engine is built test-first (TDD) under an ECC plan → tests → implement → review →
gated-commit workflow. **144 unit tests** plus **4 integration tests** (real market data,
behind `pytest -m integration` and skipped by default) cover every module, with **100%
line coverage** on the engine modules and `ruff` / `black` clean.

```bash
pip install -r requirements.txt           # runtime deps
pip install -r requirements-dev.txt       # pytest, pytest-cov, ruff, black

pytest tests/                             # full suite (integration auto-skipped)
pytest -m integration tests/             # network integration tests, explicit
pytest tests/ --cov=src --cov-report=term-missing
ruff check src/ tests/ && black --check src/ tests/
```

The no-lookahead dual guards (mutation at-and-after $t$; equivalence to the prior window)
run on every rolling estimator. Statistical claims are pinned by hand-computed fixtures
(Kupiec/Christoffersen LR statistics, normal/$t$ closed forms) asserted to $10^{-10}$.

---

## Tech stack

Python (NumPy, pandas, SciPy, Matplotlib, yfinance). Dev tooling: pytest, pytest-cov,
ruff, black. Runtime dependencies are in `requirements.txt`; dev/test tooling is kept
separate in `requirements-dev.txt`.
