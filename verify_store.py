"""Verify the data pipeline and storage layer.

Checks:

    1. Membership reconstruction from a change log, including securities that leave
       and later rejoin the index.
    2. Parsing of the constituents page, covering multi-row headers, footnote
       markers, and share-class notation.
    3. Membership derivation from dated snapshots.
    4. Membership gating: bars outside a membership interval are never tradeable.
    5. Point-in-time universe resolution from a built store.
    6. A backtest through the store-backed handler, with no look-ahead.
"""
from __future__ import annotations

import tempfile
from datetime import date

import numpy as np
import pandas as pd

from backtester import BacktestConfig, run_backtest
from backtester.history import RollingHistory
from backtester.pipeline.sources import FIXTURE_JOINER, FIXTURE_LEAVER, FIXTURE_SWAP_DATE
from backtester.pipeline.store import open_store
from backtester.pipeline.universe import (
    membership_from_snapshots, parse_sp500_tables, reconstruct_membership,
)
from build_store import build


def check_reconstruction() -> None:
    current = {"A", "B", "C"}
    changes = pd.DataFrame([
        {"date": date(2020, 1, 1), "added": "C", "removed": "X"},
        {"date": date(2018, 1, 1), "added": "B", "removed": "Y"},
        {"date": date(2019, 1, 1), "added": "A", "removed": ""},   # A rejoined 2019
        {"date": date(2016, 1, 1), "added": "", "removed": "A"},   # A left 2016
    ])
    m = reconstruct_membership(current, changes, floor=date(2015, 1, 1))
    spans = {(r.symbol): [] for r in m.itertuples(index=False)}
    for r in m.itertuples(index=False):
        spans[r.symbol].append((r.start_date, r.end_date))
    # A left in 2016 and rejoined in 2019, giving two disjoint intervals.
    assert spans["A"] == [
        (date(2015, 1, 1), date(2016, 1, 1)),
        (date(2019, 1, 1), None),
    ], spans["A"]
    assert spans["B"] == [(date(2018, 1, 1), None)]
    assert spans["X"] == [(date(2015, 1, 1), date(2020, 1, 1))]
    print("reconstruction: rejoin produces two intervals, drops close correctly -> OK")


def check_wikipedia_parser() -> None:
    """Parse constituents HTML with multi-row headers, footnotes, and dashes."""
    html = """
    <table><tr><th>Symbol</th><th>Security</th><th>GICS Sector</th><th>Date added</th></tr>
    <tr><td>AAPL</td><td>Apple</td><td>IT</td><td>1982-11-30</td></tr>
    <tr><td>BRK.B</td><td>Berkshire</td><td>Financials</td><td>2010-02-16</td></tr>
    <tr><td>NVDA</td><td>Nvidia</td><td>IT</td><td>2001-11-30</td></tr></table>
    <table>
    <tr><th rowspan="2">Date</th><th colspan="2">Added</th><th colspan="2">Removed</th></tr>
    <tr><th>Ticker</th><th>Security</th><th>Ticker</th><th>Security</th></tr>
    <tr><td>March 1, 2018</td><td>NVDA[1]</td><td>Nvidia</td><td>&mdash;</td><td>&mdash;</td></tr>
    <tr><td>January 5, 2016</td><td></td><td></td><td>NVDA</td><td>Nvidia</td></tr>
    </table>"""
    current, changes = parse_sp500_tables(html)
    assert "BRK-B" in current, "dotted tickers must normalise to the Stooq form"
    assert len(current) == 3, (
        "the constituents table has a 'Date added' column and must NOT be "
        "mistaken for the change-log")
    assert len(changes) == 2, f"expected 2 change rows, got {len(changes)}"
    assert changes.iloc[0]["added"] == "NVDA", "footnote markers must be stripped"
    assert changes.iloc[0]["removed"] == "", "em-dash must mean no change"
    m = reconstruct_membership(current, changes, floor=date(2015, 1, 1))
    nvda = m[m["symbol"] == "NVDA"]
    assert len(nvda) == 2, "a name that left and rejoined needs two intervals"
    print("wikipedia parser: multi-header, footnotes, dashes, dotted tickers -> OK")


def check_snapshot_membership() -> None:
    """Derive membership intervals from snapshots, including rejoin cases."""
    snaps = pd.DataFrame([
        {"date": "2015-01-02", "tickers": "AAPL,IBM,GM"},
        {"date": "2016-06-01", "tickers": "AAPL,IBM"},          # GM out
        {"date": "2018-03-01", "tickers": "AAPL,IBM,GM"},       # GM back
        {"date": "2024-12-01", "tickers": "AAPL,GM,TSLA"},      # IBM out, TSLA in
    ])
    m = membership_from_snapshots(
        snaps, floor=date(2015, 1, 1), ceiling=date(2024, 12, 31))
    gm = m[m["symbol"] == "GM"]
    assert len(gm) == 2, f"GM left and rejoined -> 2 intervals, got {len(gm)}"
    ibm = m[m["symbol"] == "IBM"].iloc[0]
    assert ibm["end_date"] == date(2024, 12, 1), "IBM must close when it drops out"
    tsla = m[m["symbol"] == "TSLA"].iloc[0]
    assert tsla["start_date"] == date(2024, 12, 1), "TSLA must not exist before joining"

    def universe(d):
        live = (m["start_date"] <= d) & (
            m["end_date"].isna() | (pd.Series([e for e in m["end_date"]]) > d))
        return set(m.loc[live, "symbol"])
    assert "GM" in universe(date(2015, 6, 1)), "GM was a member early"
    assert "GM" not in universe(date(2017, 1, 1)), "GM was out in 2017"
    assert "GM" in universe(date(2019, 1, 1)), "GM rejoined in 2018"
    assert "TSLA" not in universe(date(2019, 1, 1)), "TSLA joined much later"
    print("snapshot membership: joins, exits and rejoins all resolve correctly -> OK")


def check_membership_gates_trading() -> None:
    """Bars outside a membership interval must never reach the strategy.

    This is what makes extra price history harmless: a name removed from the index
    keeps trading in the real world (GME, FSLR, URBN all did), and the store may hold
    those bars, but the point-in-time universe must not offer them.
    """
    import tempfile
    from backtester.pipeline.store import SqliteStore
    from backtester.store_data import StoreDataHandler

    with tempfile.TemporaryDirectory() as d:
        st = SqliteStore(d)
        days = [x.date() for x in pd.bdate_range(date(2016, 1, 1), date(2020, 12, 31))]
        c = 100 * np.exp(np.cumsum(np.random.default_rng(1).normal(0, 0.01, len(days))))
        st.write_bars(pd.DataFrame({"symbol": "LEFTIDX", "date": days, "open": c,
                                    "high": c, "low": c, "close": c, "volume": 1e6}))
        exit_date = date(2018, 1, 2)
        st.write_membership(pd.DataFrame(
            [{"symbol": "LEFTIDX", "start_date": days[0], "end_date": exit_date}]))

        before = after = 0
        for e in StoreDataHandler(st, days[0], days[-1]).stream():
            if "LEFTIDX" in e.bars:
                if e.ts >= exit_date:
                    after += 1
                else:
                    before += 1
        assert before > 0, "the name should be tradeable while it is a member"
        assert after == 0, f"post-exit bars leaked into the universe on {after} days"
    print("membership gating: post-exit bars are never offered to the strategy -> OK")


def check_pit_and_backtest(store_root: str) -> None:
    start, end = date(2015, 1, 1), date(2023, 12, 31)
    build("fixture", store_root, [], start, end, backend="auto", seed=0)
    store = open_store(store_root)

    early = store.universe_as_of(date(2016, 1, 1))
    late = store.universe_as_of(date(2022, 1, 1))
    assert FIXTURE_LEAVER in early and FIXTURE_LEAVER not in late, "leaver PIT wrong"
    assert FIXTURE_JOINER in late and FIXTURE_JOINER not in early, "joiner PIT wrong"
    print(f"point-in-time universe: {FIXTURE_LEAVER} only early, {FIXTURE_JOINER} only late -> OK")

    cfg = BacktestConfig(start=start, end=end, data_source="store",
                         store_path=store_root, allocator="min_variance")
    res = run_backtest(cfg)
    assert len(res.equity_curve) > 100 and res.n_trades > 0
    print(f"store backtest: {len(res.equity_curve)} days, {res.n_trades} fills, "
          f"Sharpe {res.sharpe:.2f} -> OK")

    _check_no_lookahead_on_store(store_root, start, end)


def _check_no_lookahead_on_store(store_root, start, end) -> None:
    from backtester.engine import _build_allocator
    from backtester.execution import SimulatedBroker
    from backtester.portfolio import SimplePortfolio
    from backtester.risk import PassThroughRiskManager
    from backtester.store_data import StoreDataHandler
    from backtester.strategy import MomentumStrategy

    cfg = BacktestConfig(start=start, end=end, data_source="store",
                         store_path=store_root, allocator="min_variance")
    data = StoreDataHandler(open_store(store_root), start, end)
    strat, alloc = MomentumStrategy(cfg), _build_allocator(cfg)
    risk, broker = PassThroughRiskManager(), SimulatedBroker(cfg)
    pf = SimplePortfolio(cfg.initial_cash)
    history = RollingHistory(maxlen=cfg.cov_lookback_days + 5)

    events = list(data.stream())
    open_by_day = {e.ts: {s: b.open for s, b in e.bars.items()} for e in events}
    checked = violations = 0
    for i, event in enumerate(events):
        for fill in broker.execute_pending(event):
            prior_ts = events[i - 1].ts
            exp_open = open_by_day[event.ts].get(fill.symbol)
            if not (fill.ts > prior_ts and exp_open is not None
                    and abs(fill.fill_price - exp_open) < 1e-9):
                violations += 1
            checked += 1
            pf.on_fill(fill)
        pf.mark_to_market(event)
        history.update(event)
        sig = strat.on_market(event)
        if sig is not None:
            for o in risk.vet(alloc.target_orders(sig, pf.state(), history), pf.state()):
                broker.queue(o)
    assert violations == 0, f"{violations} look-ahead violations on store path"
    print(f"look-ahead on store path: {checked} fills checked, 0 violations -> OK")


def main() -> None:
    check_reconstruction()
    check_wikipedia_parser()
    check_snapshot_membership()
    check_membership_gates_trading()
    with tempfile.TemporaryDirectory() as d:
        check_pit_and_backtest(d)
    print("RESULT: PASS")


if __name__ == "__main__":
    main()
