"""Verify that no fill can reference information from its own decision date.

Instruments the event loop and checks two properties for every fill:

    1. The fill date is strictly later than the order date.
    2. The fill price equals the following day's opening price.

Run for each allocator, since allocation logic determines order generation.
"""
from __future__ import annotations

from datetime import date

from backtester.engine import _build_allocator
from backtester.config import BacktestConfig
from backtester.data import SyntheticDataHandler
from backtester.execution import SimulatedBroker
from backtester.history import RollingHistory
from backtester.portfolio import SimplePortfolio
from backtester.risk import PassThroughRiskManager
from backtester.strategy import MomentumStrategy


def _check(allocator: str) -> int:
    cfg = BacktestConfig(start=date(2015, 1, 1), end=date(2018, 12, 31), seed=7,
                         allocator=allocator)
    data = SyntheticDataHandler(cfg.start, cfg.end, seed=cfg.seed, regime=cfg.data_regime)
    strat = MomentumStrategy(cfg)
    alloc = _build_allocator(cfg)
    risk = PassThroughRiskManager()
    broker = SimulatedBroker(cfg)
    history = RollingHistory(maxlen=cfg.cov_lookback_days + 5)
    pf = SimplePortfolio(cfg.initial_cash)

    # Track order creation dates and subsequent opening prices to confirm fill
    # prices independently of the broker.
    order_created_on: list[date] = []          # ts of most recent orders queued
    open_by_day: dict[date, dict[str, float]] = {}
    fills_checked = 0
    violations = 0

    events = list(data.stream())
    for i, event in enumerate(events):
        open_by_day[event.ts] = {s: b.open for s, b in event.bars.items()}

        # Execute and verify fills.
        for fill in broker.execute_pending(event):
            # Orders producing this fill were queued on the previous event.
            prior_ts = events[i - 1].ts
            expected_open = open_by_day[event.ts][fill.symbol]
            later = fill.ts > prior_ts
            price_ok = abs(fill.fill_price - expected_open) < 1e-9
            if not (later and price_ok):
                violations += 1
            fills_checked += 1
            pf.on_fill(fill)

        # Mark to market.
        pf.mark_to_market(event)
        history.update(event)
        state = pf.state()

        # Orders queued here execute on the next event.
        sig = strat.on_market(event)
        if sig is not None:
            for o in risk.vet(alloc.target_orders(sig, state, history), state):
                broker.queue(o)
            order_created_on.append(event.ts)

    print(f"[{allocator:16}] signals={len(order_created_on):3d}  "
          f"fills={fills_checked:4d}  violations={violations}")
    return violations


def main() -> None:
    total = 0
    for allocator in ("equal_weight", "min_variance", "max_decorrelation"):
        total += _check(allocator)
    print("RESULT:", "PASS - no look-ahead detected in any allocator" if total == 0
          else "FAIL - look-ahead detected")
    assert total == 0


if __name__ == "__main__":
    main()
