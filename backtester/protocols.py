"""Structural interfaces for backtest components.

Implementations are resolved at construction time, so the event loop depends only
on these protocols and contains no factor, allocation, or data-source logic.
"""
from __future__ import annotations

from datetime import date
from typing import Iterator, Protocol, Sequence

from .events import (
    FillEvent,
    MarketEvent,
    OrderEvent,
    PortfolioState,
    SignalEvent,
)


class DataHandler(Protocol):
    def stream(self) -> Iterator[MarketEvent]:
        """Yield one MarketEvent per trading day in chronological order."""
        ...

    def get_universe(self, as_of: date) -> set[str]:
        """Return index constituents as of the given date."""
        ...


class Strategy(Protocol):
    def on_market(self, e: MarketEvent) -> SignalEvent | None:
        """Emit a cross-sectional signal, or None outside rebalance dates."""
        ...


class HistoryView(Protocol):
    def returns(
        self, symbols: Sequence[str], lookback: int
    ) -> tuple[list[str], object]:
        """Return (symbols, T x k return matrix) ending at the current close."""
        ...


class Allocator(Protocol):
    def target_orders(
        self, sig: SignalEvent, pf: PortfolioState, history: HistoryView
    ) -> list[OrderEvent]:
        """Convert factor scores into orders.

        history supplies trailing returns for risk-based allocators. Allocators that
        depend only on the signal ignore it.
        """
        ...


class RiskManager(Protocol):
    def vet(
        self, orders: list[OrderEvent], pf: PortfolioState
    ) -> list[OrderEvent]:
        """Filter or resize orders before submission."""
        ...


class ExecutionHandler(Protocol):
    def queue(self, o: OrderEvent) -> None:
        """Accept an order for execution on the next market event."""
        ...

    def execute_pending(self, e: MarketEvent) -> list[FillEvent]:
        """Execute queued orders against this event's opening prices."""
        ...


class Portfolio(Protocol):
    def on_fill(self, f: FillEvent) -> None: ...
    def mark_to_market(self, e: MarketEvent) -> None: ...
    def state(self) -> PortfolioState: ...
