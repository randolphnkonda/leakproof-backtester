"""Run the full factor analysis against a store and report the results.

Example:
    python3 run_real_analysis.py --store store_sp500 \
        --start 2021-01-01 --end 2024-12-31 --n-long 20

Compares factor specifications, sweeps the parameter grid, and reports the deflated
Sharpe ratio of the best configuration. The deflated result, not the raw Sharpe
ratio, determines whether the strategy is statistically distinguishable from the
outcome of the search itself.
"""
from __future__ import annotations

import argparse
from datetime import date

import numpy as np
import pandas as pd

from backtester import BacktestConfig, analyze_sweep, build_grid, run_backtest, run_sweep
from backtester.pipeline.store import open_store


def _iso(s: str) -> date:
    return date.fromisoformat(s)


def _metrics(ec: pd.Series) -> dict:
    r = ec.pct_change().dropna()
    if len(r) < 2 or r.std(ddof=1) == 0:
        return {}
    ann_ret = (1 + r.mean()) ** 252 - 1
    ann_vol = r.std(ddof=1) * np.sqrt(252)
    return {"ann_return": ann_ret, "ann_vol": ann_vol,
            "sharpe": ann_ret / ann_vol,
            "max_dd": (ec / ec.cummax() - 1).min()}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--store", default="store_sp500")
    p.add_argument("--start", type=_iso, default=_iso("2015-01-01"))
    p.add_argument("--end", type=_iso, default=_iso("2024-12-31"))
    p.add_argument("--n-long", type=int, default=10)
    p.add_argument("--allocator", default="min_variance")
    p.add_argument("--commission-bps", type=float, default=5.0)
    p.add_argument("--slippage-bps", type=float, default=5.0)
    a = p.parse_args()

    store = open_store(a.store)
    syms = store.symbols()
    m = store._fetch_membership()
    n_former = int((~m["end_date"].isna()).sum()) if len(m) else 0
    print("=" * 68, flush=True)
    print(f"STORE: {type(store).__name__} at '{a.store}'")
    print(f"  symbols: {len(syms)} | membership intervals: {len(m)} "
          f"| closed (former members): {n_former}")
    for probe in (a.start, a.end):
        print(f"  universe as of {probe}: {len(store.universe_as_of(probe))}")
    if n_former == 0:
        print("  Warning: no closed membership intervals. Every symbol is a current")
        print("           constituent, so results are survivorship biased.")
    print(f"  costs applied: {a.commission_bps} bps commission, "
          f"{a.slippage_bps} bps slippage")

    base = dict(start=a.start, end=a.end, data_source="store", store_path=a.store,
                n_long=a.n_long, allocator=a.allocator,
                commission_bps=a.commission_bps, slippage_bps=a.slippage_bps)

    # Factor comparison.
    print("\n" + "=" * 68)
    print("FACTOR COMPARISON (identical settings, real prices)")
    print("=" * 68)
    print(f"{'factors':26} {'ann_ret':>8} {'ann_vol':>8} {'sharpe':>7} {'max_dd':>8} {'fills':>6}")
    for spec in ("momentum", "lowvol", "reversal", "momentum,lowvol",
                 "momentum,lowvol,reversal"):
        res = run_backtest(BacktestConfig(**base, factors=spec))
        mt = _metrics(res.equity_curve)
        if mt:
            print(f"{spec:26} {mt['ann_return']:>7.1%} {mt['ann_vol']:>8.1%} "
                  f"{mt['sharpe']:>7.2f} {mt['max_dd']:>8.1%} {res.n_trades:>6}")

    # Parameter sweep and deflation.
    print("\n" + "=" * 68)
    print("SWEEP AND DEFLATED SHARPE")
    print("=" * 68)
    grid = build_grid(
        BacktestConfig(**base),
        factors=["momentum", "lowvol", "momentum,lowvol"],
        lookback_months=[6, 12, 18],
        skip_months=[0, 1],
        n_long=[5, 10, 15],
    )
    print(f"Running {len(grid)} configurations...")
    sweep = run_sweep(grid, parallel=True)
    rep = analyze_sweep(sweep)

    top = sweep.to_frame().sort_values("sharpe", ascending=False).head(5)
    print("\nTop 5 by raw Sharpe:")
    print(top[["factors", "lookback_months", "skip_months", "n_long",
               "sharpe", "max_drawdown"]].to_string(index=False))

    print("\n" + "-" * 68)
    print(rep.summary())
    print("-" * 68)

    verdict = ("SURVIVES the multiple-testing correction"
               if rep.dsr_emp >= 0.95 and rep.dsr_ana >= 0.95 else
               "MARGINAL: survives under one variance estimate, not the other"
               if max(rep.dsr_emp, rep.dsr_ana) >= 0.95 else
               "DOES NOT SURVIVE: not distinguishable from data mining")
    print(f"VERDICT: the best of {rep.n_trials} configurations {verdict}.")
    print("\nThe deflated result supersedes the raw Sharpe ratio: reporting the best")
    print("of many trials without correcting for the search overstates significance.")


if __name__ == "__main__":
    main()
