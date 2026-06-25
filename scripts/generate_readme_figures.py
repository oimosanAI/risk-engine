"""Generate the four README figures from real SPY data.

Run from anywhere:  python scripts/generate_readme_figures.py

Fetches SPY, computes rolling VaR (historical + parametric normal/t), renders the
four report plots to docs/images/, and prints the exact metrics so the README
numbers match the committed figures. Uses save_figure (the only I/O path).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.backtest.var_backtest import backtest_var  # noqa: E402
from src.data.loader import fetch_prices, prices_to_returns  # noqa: E402
from src.report.plots import (  # noqa: E402
    plot_stress_wealth,
    plot_var_comparison,
    plot_var_violations,
    plot_violation_clustering,
    save_figure,
)
from src.stress.scenarios import run_preset_scenarios  # noqa: E402
from src.var.historical import rolling_historical_var  # noqa: E402
from src.var.parametric import rolling_parametric_var  # noqa: E402

START, END = "2005-01-01", "2022-12-31"
WINDOW, CONFIDENCE = 500, 0.99
GFC_START, GFC_END = "2008-09-01", "2009-03-31"
OUT_DIR = _ROOT / "docs" / "images"


def main() -> None:
    returns = prices_to_returns(fetch_prices(["SPY"], START, END), method="simple")[
        "SPY"
    ]

    rh = rolling_historical_var(returns, window=WINDOW, confidence=CONFIDENCE)["var"]
    rn = rolling_parametric_var(
        returns, method="normal", window=WINDOW, confidence=CONFIDENCE
    )["var"]
    rt = rolling_parametric_var(
        returns, method="t", window=WINDOW, confidence=CONFIDENCE
    )["var"]

    figures = {
        "var_violations": plot_var_violations(returns.loc[rh.index], rh),
        "stress_wealth_gfc": plot_stress_wealth(
            returns.loc[GFC_START:GFC_END], "gfc_2008"
        ),
        "var_comparison": plot_var_comparison(
            returns, {"historical": rh, "normal": rn, "t": rt}
        ),
        "violation_clustering": plot_violation_clustering(returns.loc[rh.index], rh),
    }
    for name, ax in figures.items():
        path = OUT_DIR / f"{name}.png"
        save_figure(ax.figure, str(path), dpi=150)
        print(f"saved {path}  ({path.stat().st_size} bytes)")

    print("\n--- backtest violation rates (c=0.99) ---")
    for method, var_series in (("historical", rh), ("normal", rn), ("t", rt)):
        aligned = returns.loc[var_series.index]
        res = backtest_var(aligned, var_series, confidence=CONFIDENCE, method=method)
        print(
            f"{method:18} n_obs={res.n_obs} rate={res.violation_rate:.4f} "
            f"pof_p={res.pvalue_pof:.3g} ind_p={res.pvalue_ind:.3g} "
            f"cc_p={res.pvalue_cc:.3g}"
        )

    print("\n--- stress scenarios (var_series = historical rolling VaR) ---")
    scen = run_preset_scenarios(returns, var_series=rh, var_method="historical")
    for name, r in scen.items():
        print(
            f"{name:12} cum={r.cumulative_return:.4f} mdd={r.max_drawdown:.4f} "
            f"recovery_days={r.recovery_days} var_viol={r.var_violation_rate}"
        )


if __name__ == "__main__":
    main()
