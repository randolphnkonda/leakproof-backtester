"""Simulated broker with next-open execution.

Orders are queued on submission and executed against the following market event's
opening prices. The engine drains the queue at the start of each event, before the
strategy observes it, so a decision taken at the close of day t cannot reference the
price at which it executes.

Commission and slippage are applied in basis points per side.
"""
from __future__ import annotations

from .config import BacktestConfig
from .events import FillEvent, MarketEvent, OrderEvent, Side


class SimulatedBroker:
    def __init__(self, cfg: BacktestConfig):
        self.cfg = cfg
        self._pending: list[OrderEvent] = []
        self._last_px: dict[str, float] = {}

    def queue(self, o: OrderEvent) -> None:
        self._pending.append(o)

    def observe(self, e: MarketEvent) -> None:
        """Record the latest close per symbol for delisting liquidation."""
        for sym, bar in e.bars.items():
            self._last_px[sym] = bar.close

    def execute_pending(self, e: MarketEvent) -> list[FillEvent]:
        if not self._pending:
            return []

        fills: list[FillEvent] = []
        carried: list[OrderEvent] = []
        for o in self._pending:
            bar = e.bars.get(o.symbol)
            if bar is None:
                # No quote available: the symbol has delisted, been acquired, or left
                # the universe. Sells execute at the last observed close so that
                # positions are closed out rather than held indefinitely at a stale
                # mark. Buys are cancelled, since no execution price is observable.
                last = self._last_px.get(o.symbol)
                if o.side is Side.SELL and last is not None:
                    notional = last * o.quantity
                    fills.append(FillEvent(
                        ts=e.ts, symbol=o.symbol, side=o.side, quantity=o.quantity,
                        fill_price=last,
                        commission=notional * (self.cfg.commission_bps / 1e4),
                        slippage=0.0,
                    ))
                continue
            base = bar.open
            # Slippage is applied adversely: buys fill higher, sells fill lower.
            slip = base * (self.cfg.slippage_bps / 1e4) * o.side.value
            fill_price = base + slip
            notional = fill_price * o.quantity
            commission = notional * (self.cfg.commission_bps / 1e4)
            fills.append(
                FillEvent(
                    ts=e.ts, symbol=o.symbol, side=o.side,
                    quantity=o.quantity, fill_price=fill_price,
                    commission=commission, slippage=abs(slip) * o.quantity,
                )
            )
        self._pending = carried
        return fills
