# Risk Engine — Design Decisions

This document records the non-obvious design choices made during implementation,
and the reasoning behind each. It exists so that a reader (or a future session
of Claude Code) can understand not just *what* was built, but *why*.

## 1. Sign Convention

Loss is defined as L = −r, where r is the arithmetic daily return.
VaR and ES are reported as **positive loss magnitudes**.

    VaR_c = quantile(L, c)       # e.g. c=0.99
    ES_c  = E[L | L ≥ VaR_c]

This means a 1% daily loss is reported as VaR = 0.01, not −0.01.
The choice is deliberate: risk numbers that are positive loss magnitudes
are harder to misread (a larger number always means more risk).
All three VaR methods (historical, parametric, Monte Carlo) and the
backtest share this convention. Any caller that passes negative var_series
receives a warning, not a raise, because plotting is exploratory — but
the backtest is strict.

## 2. No-Lookahead Guarantee

The rolling estimate indexed at date t is computed from
`returns.iloc[i-window:i]` — the window observations at positions
i−window, …, i−1. The upper bound is **exclusive**: position i (date t
itself) is never read.

A naive `.rolling(window)` aggregate includes the current row and would
pass a weaker test. Two complementary guards enforce the stricter rule:

- **Mutation test** — returns at all dates ≥ t (including t itself) are
  shocked after computing the rolling result; the row at t must be unchanged.
  Mutating only dates strictly *after* t is insufficient and would pass the
  naive implementation.
- **Equivalence test** — the rolling row at t is asserted numerically
  identical to the point-in-time estimator run on exactly
  `returns.iloc[i-window:i]`, across multiple dates and confidence levels.

Both guards must pass. The equivalence test works because both the rolling
loop and the point-in-time function route through the same private numeric
core (`_var_es_from_returns` / `_normal_var_es` / `_t_var_es`).

## 3. Minimum Observations Floor

$$n_{\min}(c) = \lceil 1/(1-c) \rceil$$

At c = 0.99 this is 100. Below this floor, the (1−c) tail contains fewer
than one expected observation; the quantile is pure extrapolation dressed
as an empirical estimate. Both the point-in-time functions and the rolling
window enforce this floor and raise `ValueError` rather than returning a
silently unreliable number.

For `rolling_historical_var`, the floor applies to the raw window (no
compression). For `rolling_parametric_var`, the same applies — parametric
uses sqrt(h) scaling and feeds all window observations into moment/MLE
estimation (no block compression). This differs from `rolling_historical_var`
with horizon > 1, where the floor is checked on `window // horizon`
(the number of non-overlapping blocks).

## 4. Horizon Scaling: Historical vs Parametric

The two methods use **different** horizon scaling, for mathematical reasons:

| Method      | Horizon scaling               | Why |
|-------------|-------------------------------|-----|
| Historical  | Non-overlapping blocks: `prod(1+r)−1` | The empirical distribution is built from the actual h-day returns; no distributional assumption is made, so sqrt(h) would introduce an iid assumption that the method deliberately avoids. |
| Parametric  | `mu × h`, `sigma × sqrt(h)`  | The closed form already assumes a parametric distribution (normal or t). Under that assumption, scaling the moments is exact for iid returns. sqrt(h) is labeled in docstrings as an iid approximation that breaks under autocorrelation. |

Mixing these up (using sqrt(h) for historical, or blocks for parametric)
would be a category error.

## 5. Student-t ES: Formula and Scale Parameterisation

The closed-form ES for the Student-t required a correction during
implementation. The original spec had a sign error. The correct form
(Acerbi–Tasche / McNeil-Frey-Embrechts) is:

$$\text{ES} = -\left(\mu - \text{scale} \cdot
\frac{f(q)}{1-c} \cdot \frac{\nu + q^2}{\nu - 1}\right)$$

where q = t.ppf(1−c, ν) and f is the t pdf. The minus sign on the
second term (not plus) is critical: ES ≥ VaR follows analytically.

The scale parameter is **not** sigma. The t distribution with df=ν has
variance `scale² × ν/(ν−2)`, so to match the normal path (where sigma
IS the standard deviation), we standardise:

    scale = sigma × sqrt((nu − 2) / nu)

This ensures the t and normal paths share the same variance and differ
only in tail shape. The result stored in `ParamVaRResult.sigma` is the
actual volatility (sample std), not the internal t-scale.

ν is estimated via free-scale MLE: `scipy.stats.t.fit(returns, floc=mu)`.
Fixing `fscale=sigma` inflates ν by 2–3× on fat-tailed data (confirmed
empirically on Student-t(df=3) synthetic returns), making the t-tails
too thin and defeating the purpose of using the t distribution.

## 6. Kupiec POF at x = 0 and x = T

The original spec stated "x = 0 → LR_POF = 0 trivially." This is wrong.

At x = 0 (zero violations over T days):

$$LR_{\text{POF}} = -2T \ln(1-p_0)$$

For c = 0.99, T = 250: LR ≈ 5.03, p-value ≈ 0.025. Zero violations is
*mildly surprising* — it signals an over-conservative model, not a perfect
one. The hand-computable fixture pins this value at 1e-10 tolerance.

The `0 × ln(0) = 0` convention is applied via an explicit guard function
(`_xlogx_term`) that returns 0.0 before reaching `math.log`, preventing
numpy's silent −inf path.

## 7. Christoffersen Degenerate Cases

When `n10 + n11 = 0` (no transition out of the violation state — all
violations are consecutive with no intervening non-violation), LR_ind = 0.
The Markov alternative model cannot be distinguished from the null because
the transition probability π₁₁ has no denominator. This is not an error;
it means clustering is undetectable from this data, and the test contributes
no information. Documented in `_lr_ind` docstring.

## 8. Violation Definition: Strict Inequality

A violation is defined as `loss > VaR` (strict), not `loss ≥ VaR`.
This is consistent with VaR = "the loss **not exceeded** with probability c."
A loss exactly equal to VaR is not a violation by this definition.
The backtest, the plots, and the stress module all share this definition
via `var_backtest.violations()`, which lives in `_utils.py` as the single
source of truth.

## 9. Dependency Architecture

The engine is layered so that shared logic has exactly one home and the
visualization layer stays lightweight.

### Module map

```
src/
├── data/loader.py           fetch_prices / prices_to_returns
├── var/
│   ├── _utils.py            _clean / _min_obs / violations (shared, scipy-free)
│   ├── historical.py        historical-simulation VaR/ES
│   └── parametric.py        normal + Student-t VaR/ES (closed form)
├── backtest/var_backtest.py Kupiec POF / Christoffersen / Basel traffic-light
├── stress/scenarios.py      GFC 2008 / COVID 2020 stress scenarios
└── report/plots.py          monochrome visualization (scipy-free)
```

### Dependency structure

```
var._utils          (scipy-free leaf: _clean, _min_obs, violations)
    ↑                   ↑               ↑
var.historical   var.parametric    stress.scenarios    report.plots
                      ↑
               var_backtest  (only scipy importer in the engine layer)
```

`var._utils` is the single source of truth for the shared logic used across
modules — `_clean`, `_min_obs`, and the strict `violations` definition (§8).
Every layer routes through it rather than re-deriving these, so the statistics
and the visualizations can never diverge.

`report.plots` is intentionally scipy-free: importing it does not pull scipy,
keeping the visualization layer lightweight. `var_backtest` is the only module
in the engine layer that imports scipy (for the chi-square survival function
used by the Kupiec, Christoffersen, and conditional-coverage p-values).