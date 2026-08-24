"""Position and cash ledger.

Applies fills, marks positions to market on each day's closing prices, and exposes
an immutable PortfolioState snapshot. Contains no allocation or sizing logic.
"""
from __future__ import annotations

from .events import FillEvent, MarketEvent, PortfolioState, Side


class SimplePortfolio:
    def __init__(self, initial_cash: float):
        self._cash = initial_cash
        self._positions: dict[str, float] = {}
        self._last_prices: dict[str, float] = {}
        self._ts = None
        self._equity = initial_cash
        self.n_fills = 0

    def on_fill(self, f: FillEvent) -> None:
        signed_qty = f.quantity * f.side.value
        cash_flow = -f.fill_price * signed_qty
        self._cash += cash_flow - f.commission
        self._positions[f.symbol] = (
            self._positions.get(f.symbol, 0.0) + signed_qty
        )
        if abs(self._positions[f.symbol]) < 1e-9:
            self._positions.pop(f.symbol, None)
        self.n_fills += 1

    def mark_to_market(self, e: MarketEvent) -> None:
        for sym, bar in e.bars.items():
            self._last_prices[sym] = bar.close
        holdings_value = sum(
            qty * self._last_prices.get(sym, 0.0)
            for sym, qty in self._positions.items()
        )
        self._ts = e.ts
        self._equity = self._cash + holdings_value

    def state(self) -> PortfolioState:
        return PortfolioState(
            ts=self._ts,
            cash=self._cash,
            positions=dict(self._positions),
            last_prices=dict(self._last_prices),
            equity=self._equity,
        )
