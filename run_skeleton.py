"""Run a single backtest on synthetic data and write the equity curve.

Produces equity_curve.csv and equity_curve.png, and checks that repeated runs of the
same configuration are identical.
"""
from __future__ import annotations

from datetime import date

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from backtester import BacktestConfig, run_backtest


def main() -> None:
    cfg = BacktestConfig(
        start=date(2015, 1, 1),
        end=date(2023, 12, 31),
        n_long=5,
        seed=42,
    )
    result = run_backtest(cfg)

    ec = result.equity_curve
    total_return = ec.iloc[-1] / ec.iloc[0] - 1.0
    print("=" * 60)
    print("SINGLE BACKTEST")
    print("=" * 60)
    print(f"Factor            : {cfg.factor}")
    print(f"Universe          : {cfg.universe_id}")
    print(f"Period            : {cfg.start} -> {cfg.end}")
    print(f"Trading days      : {len(ec)}")
    print(f"Rebalance         : {cfg.rebalance}, long top {cfg.n_long}")
    print(f"Fills (trades)    : {result.n_trades}")
    print(f"Start equity      : {ec.iloc[0]:,.0f}")
    print(f"End equity        : {ec.iloc[-1]:,.0f}")
    print(f"Total return      : {total_return:+.1%}")
    print(f"Sharpe (raw)      : {result.sharpe:.2f}")
    print("=" * 60)
    print("Note: synthetic data. Use a store for market prices.")

    # Repeated runs of one configuration must be identical.
    again = run_backtest(cfg)
    assert again.equity_curve.equals(ec), "non-deterministic run!"
    print("Determinism check : PASS (identical curve on re-run)")

    ec.to_csv("equity_curve.csv")
    ax = ec.plot(figsize=(10, 5), title="Skeleton equity curve (synthetic data)")
    ax.set_ylabel("Equity")
    ax.set_xlabel("Date")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("equity_curve.png", dpi=110)
    print("Saved             : equity_curve.csv, equity_curve.png")


if __name__ == "__main__":
    main()
