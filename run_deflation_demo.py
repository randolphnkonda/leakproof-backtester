"""Compare deflated Sharpe ratios across a signal regime and a null control.

The null control uses zero-drift data in which no strategy has an edge, so the best
of N configurations should be rejected despite a positive raw Sharpe ratio. Running
both regimes shows the correction distinguishing signal from selection.
"""
from __future__ import annotations

from datetime import date

import matplotlib
import matplotlib.pyplot as plt

from backtester import BacktestConfig, analyze_sweep, build_grid, run_sweep

# Non-interactive backend for headless rendering.
matplotlib.use("Agg")


def _run_regime(regime: str):
    base = BacktestConfig(start=date(2015, 1, 1), end=date(2023, 12, 31),
                          seed=7, data_regime=regime)
    grid = build_grid(
        base,
        lookback_months=[6, 9, 12, 15, 18],
        skip_months=[0, 1, 2],
        n_long=[3, 4, 5, 6, 7],
    )
    sweep = run_sweep(grid, parallel=False)
    report = analyze_sweep(sweep)
    return sweep, report


def main() -> None:
    print("#" * 62)
    print("# Regime 1: dispersed drift (signal present)")
    print("#" * 62)
    disp_sweep, disp = _run_regime("dispersed")
    print(disp.summary())

    print()
    print("#" * 62)
    print("# Regime 2: null control (no signal)")
    print("#" * 62)
    null_sweep, null = _run_regime("null")
    print(null.summary())

    print()
    print("=" * 62)
    print("INTERPRETATION")
    print("=" * 62)
    print("Dispersed regime: the uncorrected probabilistic Sharpe is near certain.")
    print("Deflation compares against the expected best of N, so a genuine edge")
    print("survives or lands near the threshold.")
    print()
    print("Null regime: the raw Sharpe reflects selection only, and the deflated")
    print("value should fall well below 0.95 under both variance estimators.")
    print()
    print("The empirical estimator uses the observed dispersion of overlapping")
    print("configurations and is the more lenient benchmark. The analytic estimator")
    print("treats trials as independent and is stricter. The effective number of")
    print("independent trials lies below the nominal count.")

    # Plot each probability against the 0.95 threshold.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, rep, title in (
        (axes[0], disp, "Dispersed drift (real edge)"),
        (axes[1], null, "Null control (no edge)"),
    ):
        labels = ["PSR vs 0\n(naive)", "DSR\nempirical", "DSR\nanalytic"]
        vals = [rep.psr_vs_zero, rep.dsr_emp, rep.dsr_ana]
        colors = ["#1D9E75" if v >= 0.95 else "#D85A30" for v in vals]
        ax.bar(labels, vals, color=colors, edgecolor="white")
        ax.axhline(0.95, color="black", linestyle="--", linewidth=1)
        ax.text(2.4, 0.955, "0.95", fontsize=9, va="bottom", ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        for i, v in enumerate(vals):
            ax.text(i, min(v + 0.02, 1.0), f"{v:.3f}", ha="center", fontsize=9)
    axes[0].set_ylabel("P(true Sharpe clears the bar)")
    fig.suptitle("Deflated Sharpe ratio against the 95% threshold", fontsize=11)
    plt.tight_layout()
    plt.savefig("deflation_verdicts.png", dpi=110)
    print("\nSaved                      : deflation_verdicts.png")


if __name__ == "__main__":
    main()
