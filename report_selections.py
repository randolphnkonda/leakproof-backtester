"""Report the securities a strategy actually selects, with data-quality flags.

Example:
    python3 report_selections.py --store store_sp500 --factors lowvol

Runs the strategy and records holdings at every rebalance, then joins those against
per-symbol quality statistics. Unlike audit_store.py, which ranks by full-window
volatility as a proxy, this reflects warmup requirements and therefore the actual
selections. Compare runs with and without --no-quality-filter to see the filter's
effect.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import date

import pandas as pd

from backtester import BacktestConfig
from backtester.engine import _build_allocator, _cached_exclusions
from backtester.execution import SimulatedBroker
from backtester.history import RollingHistory
from backtester.pipeline.store import open_store
from backtester.portfolio import SimplePortfolio
from backtester.quality import QualityThresholds, symbol_stats
from backtester.risk import PassThroughRiskManager
from backtester.store_data import StoreDataHandler
from backtester.strategy import CompositeStrategy


def _iso(s: str) -> date:
    return date.fromisoformat(s)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--store", default="store_sp500")
    p.add_argument("--start", type=_iso, default=_iso("2016-01-01"))
    p.add_argument("--end", type=_iso, default=_iso("2024-12-31"))
    p.add_argument("--factors", default="lowvol")
    p.add_argument("--n-long", type=int, default=20)
    p.add_argument("--no-quality-filter", action="store_true")
    a = p.parse_args()

    cfg = BacktestConfig(
        start=a.start, end=a.end, factors=a.factors, n_long=a.n_long,
        data_source="store", store_path=a.store,
        quality_filter=not a.no_quality_filter,
    )
    # Emit the header before the quality scan and backtest, which take time on a
    # populated store.
    print("=" * 70)
    print(f"SELECTIONS: factors={a.factors}, n_long={a.n_long}, "
          f"quality_filter={'OFF' if a.no_quality_filter else 'ON'}")
    print("=" * 70, flush=True)

    store = open_store(a.store)
    print(f"opening store '{a.store}'...", flush=True)
    excl = {} if a.no_quality_filter else _cached_exclusions(cfg)
    data = StoreDataHandler(store, a.start, a.end, exclude=excl)
    print("loading price panel and running the backtest...", flush=True)
    print(f"universe after filtering: {len(data.symbols)} symbols "
          f"({len(excl)} excluded)")
    if excl:
        reasons = Counter(v.split(":")[0] for v in excl.values())
        for reason, n in reasons.most_common():
            print(f"  {reason}: {n}")

    strat, alloc = CompositeStrategy(cfg), _build_allocator(cfg)
    risk, broker = PassThroughRiskManager(), SimulatedBroker(cfg)
    pf = SimplePortfolio(cfg.initial_cash)
    hist = RollingHistory(maxlen=cfg.cov_lookback_days + 5)

    picks: Counter = Counter()
    rebalances = 0
    for e in data.stream():
        for f in broker.execute_pending(e):
            pf.on_fill(f)
        broker.observe(e)
        pf.mark_to_market(e)
        hist.update(e)
        sig = strat.on_market(e)
        if sig:
            rebalances += 1
            ranked = sorted(sig.scores.items(), key=lambda kv: kv[1], reverse=True)
            for sym, _ in ranked[:a.n_long]:
                picks[sym] += 1
            for o in risk.vet(alloc.target_orders(sig, pf.state(), hist), pf.state()):
                broker.queue(o)

    print(f"\nrebalances: {rebalances}, distinct names ever held: {len(picks)}")

    stats = symbol_stats(store, list(picks), a.start, a.end).set_index("symbol")
    th = QualityThresholds(min_obs=cfg.min_obs, max_abs_daily=cfg.max_abs_daily)
    rows = []
    for sym, count in picks.most_common(20):
        st = stats.loc[sym] if sym in stats.index else None
        if st is None:
            continue
        bad = (st["obs"] < th.min_obs or st["zero_share"] > th.max_zero_share
               or st["flat_run"] >= th.max_flat_run
               or st["max_abs_daily"] > th.max_abs_daily)
        rows.append({"symbol": sym, "times_held": count, "obs": int(st["obs"]),
                     "zero_share": round(float(st["zero_share"]), 3),
                     "flat_run": int(st["flat_run"]),
                     "max_abs_daily": round(float(st["max_abs_daily"]), 2),
                     "SUSPECT": "yes" if bad else ""})
    df = pd.DataFrame(rows)
    print("\nMost frequently held names:")
    print(df.to_string(index=False))
    n_bad = int((df["SUSPECT"] == "yes").sum()) if len(df) else 0
    print(f"\n{n_bad} of the top {len(df)} most-held names are data-quality suspects.")
    if n_bad and not a.no_quality_filter:
        print("Flagged symbols passing the filter indicate thresholds are too loose.")
    elif not n_bad and not a.no_quality_filter:
        print("No flagged symbols among the actual holdings.")


if __name__ == "__main__":
    main()
