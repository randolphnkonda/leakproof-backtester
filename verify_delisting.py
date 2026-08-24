"""Verify that holdings are liquidated when a security stops trading.

Builds a store in which one symbol ceases trading mid-window, runs a backtest, and
asserts the portfolio does not still hold it at the end. Without liquidation the
position would remain indefinitely at a stale mark.
"""
from __future__ import annotations

import shutil
import tempfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from backtester import BacktestConfig
from backtester.engine import _build_allocator
from backtester.execution import SimulatedBroker
from backtester.history import RollingHistory
from backtester.pipeline.clean import clean_bars
from backtester.pipeline.store import SqliteStore
from backtester.portfolio import SimplePortfolio
from backtester.risk import PassThroughRiskManager
from backtester.store_data import StoreDataHandler
from backtester.strategy import CompositeStrategy

START, END = date(2015, 1, 1), date(2019, 12, 31)
DELIST = date(2017, 6, 1)


def _bars(sym: str, start: date, end: date, seed: int, drift: float) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = [d.date() for d in pd.bdate_range(start, end)]
    close = 100 * np.exp(np.cumsum(rng.normal(drift, 0.01, len(days))))
    return pd.DataFrame({
        "Date": [d.isoformat() for d in days],
        "Open": close, "High": close * 1.01, "Low": close * 0.99,
        "Close": close, "Volume": 1e6,
    })


def build_store(root: Path) -> SqliteStore:
    store = SqliteStore(root)
    rows = []
    # This symbol trends upward so momentum selects it, then stops trading.
    for i, (sym, end, drift) in enumerate([
        ("DOOMED", DELIST, 0.004),
        ("ALPHA", END, 0.0009), ("BETA", END, 0.0007), ("GAMMA", END, 0.0005),
        ("DELTA", END, 0.0003), ("EPSIL", END, 0.0001),
    ]):
        tidy, _ = clean_bars(_bars(sym, START, end, seed=i, drift=drift), sym)
        store.write_bars(tidy)
        rows.append({"symbol": sym, "start_date": START,
                     "end_date": (DELIST if sym == "DOOMED" else None)})
    store.write_membership(pd.DataFrame(rows))
    return store


def run(store, cfg) -> dict:
    data = StoreDataHandler(store, cfg.start, cfg.end)
    strat, alloc = CompositeStrategy(cfg), _build_allocator(cfg)
    risk, broker = PassThroughRiskManager(), SimulatedBroker(cfg)
    pf = SimplePortfolio(cfg.initial_cash)
    hist = RollingHistory(maxlen=cfg.cov_lookback_days + 5)
    for e in data.stream():
        for f in broker.execute_pending(e):
            pf.on_fill(f)
        broker.observe(e)
        pf.mark_to_market(e)
        hist.update(e)
        sig = strat.on_market(e)
        if sig:
            for o in risk.vet(alloc.target_orders(sig, pf.state(), hist), pf.state()):
                broker.queue(o)
    st = pf.state()
    return {"positions": st.positions, "equity": st.equity,
            "stuck": "DOOMED" in st.positions}


def main() -> None:
    tmp = Path(tempfile.mkdtemp())
    try:
        store = build_store(tmp / "s")
        cfg = BacktestConfig(start=START, end=END, factors="momentum",
                             n_long=3, lookback_months=6, skip_months=1,
                             data_source="store", store_path=str(tmp / "s"))
        out = run(store, cfg)
        print(f"final positions : {sorted(out['positions'])}")
        print(f"DOOMED delisted {DELIST}, backtest ends {END}")
        if out["stuck"]:
            qty = out["positions"]["DOOMED"]
            print(f"FAIL: still holding {qty:.1f} shares of a symbol that stopped "
                  f"trading {(END - DELIST).days} days before the end.")
        else:
            print("Delisted holding liquidated; no residual position.")
        assert not out["stuck"], "delisted holding was not liquidated"
        print("RESULT: PASS")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
