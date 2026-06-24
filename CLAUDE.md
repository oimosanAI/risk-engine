# Risk Engine — Project Context

Portfolio risk engine: Value-at-Risk, Expected Shortfall, and historical
stress testing. This is a portfolio piece for risk/quant roles. It is judged on
correctness and honesty, not on impressive numbers.

Generic engineering standards (PEP 8, black/isort/ruff, full type hints, TDD,
security) are governed by the global ECC rules in ~/.claude/rules/. This file
does NOT repeat them — it captures only what ECC cannot know: the domain
conventions and the decisions locked while building this repo.

## Design philosophy (non-negotiable)
- Verifiable over impressive. Every metric reproducible from raw data.
- Honest about limitations. State assumptions and failure modes in docstrings
  and README. Never hide them.
- Out-of-sample discipline. Any rolling/backtested estimate at time t must be
  computable using ONLY data strictly before t. No lookahead, ever.
- Do not inflate. Never tune toward a target number. Report the empirical
  result. No conservative fudge, no clamping, no padding. A too-good result is a
  bug until proven otherwise.

## Domain conventions
- Returns: simple (arithmetic) daily unless a module states otherwise (log).
- loss = -return. VaR and ES are reported as POSITIVE loss magnitudes.
- VaR at confidence c (e.g. 0.99) = quantile(losses, c), losses = -returns
  = the (1-c) lower-tail of returns, sign-flipped to a positive loss.
- ES (CVaR) = mean(losses[losses >= VaR]); empty-tail fallback returns VaR.
  Invariant: ES >= VaR, always.
- Quantile interpolation is a PARAMETER (maps to np.quantile method=). Default
  "linear". Never hardcoded.
- Horizon default 1. For h>1, compound NON-OVERLAPPING blocks: prod(1+r)-1,
  dropping the trailing partial block. sqrt(h) scaling is NOT used; if ever
  introduced, it must be docstring-labeled as an iid approximation that breaks
  under fat tails / autocorrelation.

## Locked decisions (source of truth — propagate to other modules)
- MIN_OBS = ceil(1 / (1 - confidence)). historical_var raises ValueError below
  it (the tail is unresolvable). rolling raises unless n_obs > window.
- NaN policy: drop + logging.warning(dropped_count). No silent skipna.
- portfolio_returns: weighted sum via matmul (asset_returns.values @ w) so a
  NaN in any asset PROPAGATES (never masked by skipna), then any-NaN rows are
  dropped + logged. Weights: dict aligned to columns, raise on unknown/missing;
  warn if |sum - 1| > tol but do NOT auto-normalize. Daily rebalance assumed.
- All-positive returns: VaR may be <= 0 (no losses). No zero-clamp.
- Shared numeric core (_var_es_from_returns) is the single path for both
  historical_var and each rolling row, so lookahead-equivalence is exact.
- No-lookahead mechanics: rolling row at index[i] uses iloc[i-window:i]
  (left-inclusive, right-EXCLUSIVE of t). First valid row at index[window];
  length == n_obs - window. Never a bare .rolling(window) — it includes t.

## Architecture
- src/data/loader.py      price fetch + return computation (no VaR logic here)
- src/var/historical.py   historical simulation VaR + ES        [implemented]
- src/var/parametric.py   variance-covariance (normal, Student-t)
- src/var/monte_carlo.py  Monte Carlo VaR (seeded)
- src/stress/scenarios.py historical stress windows (2008 GFC, 2020 COVID)
- src/backtest/var_backtest.py  Kupiec POF + Christoffersen independence tests
- src/report/plots.py     monochrome/grayscale visualization only

## Testing
- pytest, AAA, seeded numpy fixtures. Synthetic returns (normal + Student-t) for
  unit tests; do not fetch market data in tests.
- Every VaR method needs: a hand-computable fixture asserted against the private
  numeric core (not the public guarded function), plus property tests
  (ES >= VaR, monotonicity in confidence), plus a dual lookahead guard for any
  rolling estimate: (a) mutation at-and-after t leaves row t unchanged;
  (b) row t exactly equals the method run on the prior window alone.
- Coverage: run pytest-cov with term-missing. Report the REAL number and the
  missing lines. Do not chase 100% by deleting code or loosening asserts;
  justify any intentionally unreachable line.

## Dependencies
- Runtime deps in requirements.txt. Dev/test tooling (pytest, pytest-cov, ruff,
  black) in requirements-dev.txt — keep them separate.

## Data limitations (state in README once loader exists)
- Source: yfinance (free). No survivorship-bias correction; dividend/split
  adjustment is provider-dependent; holiday/gap handling matters. These bound
  the credibility of every downstream result — say so explicitly.

## Workflow with ECC
- New module: /plan → review → /tdd → /code-review → /verify.
- Tests are written FIRST under /tdd. Lookahead guards go at the TOP of the
  rolling test, not appended later.
- Commits: conventional commits (feat/fix/chore/docs/test). Author MUST be the
  human (oimosanAI), not the agent. Verify author before push. AgentShield hooks
  blocking unsafe git flags are expected behavior — fix the cause, don't disable.

## Known tech debt
- Import resolution currently relies on a root conftest.py injecting sys.path.
  Migrate to pyproject.toml [tool.pytest.ini_options] pythonpath (or an editable
  install) when convenient. Not blocking.