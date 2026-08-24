"""Run a parameter sweep and report the distribution of results.

Demonstrates grid construction, parallel execution, reproducibility, and equivalence
between serial and parallel runs.
"""
from __future__ import annotations

from datetime import date

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from backtester import BacktestConfig, build_grid, run_sweep

# Non-interactive backend for headless rendering.
matplotlib.use("Agg")


def main() -> None:
    base = BacktestConfig(start=date(2015, 1, 1), end=date(2023, 12, 31), seed=7)

    # Strategy parameters are swept at a fixed data seed.
    grid = build_grid(
        base,
        lookback_months=[6, 9, 12, 15, 18],
        skip_months=[0, 1, 2],
        n_long=[3, 4, 5, 6, 7],
    )
    print(f"Grid size (N trials)      : {len(grid)}")

    # Serial execution.
    serial = run_sweep(grid, parallel=False)
    print(f"Serial wall time          : {serial.wall_seconds:.2f}s")

    # Parallel execution.
    parallel = run_sweep(grid, parallel=True, max_workers=2)
    print(f"Parallel wall time        : {parallel.wall_seconds:.2f}s "
          f"(workers requested: 2)")
    speedup = serial.wall_seconds / parallel.wall_seconds
    print(f"Speedup                   : {speedup:.2f}x "
          f"({'below 1 on a single core' if speedup < 1 else 'multi-core speedup'})")

    # Serial and parallel results must agree.
    same = np.allclose(serial.sharpes, parallel.sharpes, equal_nan=True)
    print(f"Serial == Parallel        : {'PASS' if same else 'FAIL'}")

    # A repeated sweep must be identical.
    again = run_sweep(grid, parallel=False)
    repro = np.allclose(serial.sharpes, again.sharpes, equal_nan=True)
    print(f"Reproducible re-run        : {'PASS' if repro else 'FAIL'}")

    # Distribution of trial Sharpe ratios.
    s = serial.sharpes
    print("-" * 60)
    print(f"Sharpe  min / median / max: "
          f"{np.nanmin(s):.2f} / {np.nanmedian(s):.2f} / {np.nanmax(s):.2f}")
    print(f"Sharpe  mean / std        : {np.nanmean(s):.2f} / {np.nanstd(s, ddof=1):.2f}")
    b = serial.best
    print(f"Best config               : lookback={b.config.lookback_months}, "
          f"skip={b.config.skip_months}, n_long={b.config.n_long}")
    print(f"Best Sharpe (pre-deflation): {b.sharpe:.2f}  "
          f"[skew={b.skew:.2f}, kurt_excess={b.kurt_excess:.2f}, T={b.n_periods}]")
    print(f"Note: synthetic data, identical across trials. A best Sharpe of "
          f"{b.sharpe:.2f}")
    print(f"      from {serial.n_trials} trials on one path reflects selection, not edge.")

    # Write outputs.
    df = serial.to_frame().sort_values("sharpe", ascending=False)
    df.to_csv("sweep_results.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    finite = s[np.isfinite(s)]
    ax.hist(finite, bins=20, edgecolor="white")
    ax.axvline(np.nanmax(s), color="crimson", linestyle="--",
               label=f"best = {np.nanmax(s):.2f}")
    ax.set_title(f"Sharpe distribution across {serial.n_trials} trials (synthetic data)")
    ax.set_xlabel("Annualised Sharpe")
    ax.set_ylabel("Number of configs")
    ax.legend()
    plt.tight_layout()
    plt.savefig("sweep_sharpe_hist.png", dpi=110)
    print("Saved                     : sweep_results.csv, sweep_sharpe_hist.png")


if __name__ == "__main__":
    main()
