"""Verify the deflated Sharpe implementation.

Two checks:

    1. The closed-form expected maximum agrees with a Monte Carlo simulation of the
       maximum of N null Sharpe ratios.
    2. On a zero-drift control, where no strategy has an edge, the best of N
       configurations is correctly rejected despite a positive raw Sharpe ratio.
"""
from __future__ import annotations

from datetime import date

from backtester import (
    BacktestConfig,
    analyze_sweep,
    build_grid,
    expected_max_sharpe,
    monte_carlo_expected_max,
    run_sweep,
)


def check_formula() -> None:
    worst = 0.0
    for N in (25, 75, 200):
        a = expected_max_sharpe(1.0, N)
        m = monte_carlo_expected_max(1.0, N, n_sims=200_000, seed=1)
        worst = max(worst, abs(a - m) / m)
    print(f"expected_max vs Monte Carlo: max rel. error {worst:.1%}")
    assert worst < 0.05, "analytic expected-max formula diverges from Monte Carlo"


def check_null_control() -> None:
    base = BacktestConfig(start=date(2015, 1, 1), end=date(2023, 12, 31),
                          seed=7, data_regime="null")
    grid = build_grid(base, lookback_months=[6, 9, 12, 15, 18],
                      skip_months=[0, 1, 2], n_long=[3, 4, 5, 6, 7])
    rep = analyze_sweep(run_sweep(grid, parallel=False))
    print(f"null control: best raw Sharpe {rep.best_sharpe_ann:.2f} "
          f"over N={rep.n_trials} trials")
    print(f"  DSR_empirical = {rep.dsr_emp:.3f}, DSR_analytic = {rep.dsr_ana:.3f}")
    assert rep.dsr_emp < 0.95, "empirical DSR failed to reject a no-edge winner"
    assert rep.dsr_ana < 0.95, "analytic DSR failed to reject a no-edge winner"
    print("  both below 0.95: correctly rejected")


def main() -> None:
    check_formula()
    check_null_control()
    print("RESULT: PASS")


if __name__ == "__main__":
    main()
