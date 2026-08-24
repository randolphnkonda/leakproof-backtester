"""Event types exchanged between backtest components.

Message flow through the event loop:

    MarketEvent  -> Strategy   emits -> SignalEvent
    SignalEvent  -> Allocator  emits -> OrderEvent
    OrderEvent   -> Broker     emits -> FillEvent
    FillEvent    -> Portfolio  updates state

MarketEvent and SignalEvent are cross-sectional: each carries the full universe for
a single day. Cross-sectional strategies rank securities against one another and
cannot produce a signal until every security for that date is available.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum


class Side(Enum):
    BUY = 1
    SELL = -1


@dataclass(frozen=True)
class Bar:
    symbol: str
    ts: date
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class MarketEvent:
    """Daily snapshot of every tradeable symbol."""
    ts: date
    bars: dict[str, Bar]


@dataclass(frozen=True)
class SignalEvent:
    """Cross-section of factor scores. Higher scores rank more attractive."""
    ts: date
    scores: dict[str, float]


@dataclass(frozen=True)
class OrderEvent:
    ts: date          # decision time (close of day t)
    symbol: str
    side: Side
    quantity: float   # shares; fractional quantities are permitted


@dataclass(frozen=True)
class FillEvent:
    ts: date          # execution time (open of day t+1)
    symbol: str
    side: Side
    quantity: float
    fill_price: float
    commission: float
    slippage: float


@dataclass(frozen=True)
class PortfolioState:
    """An immutable read-model of the ledger, handed to the Allocator/RiskManager.

    Carries last_prices (close of t) so the Allocator sizes weights->shares using
    only information knowable at decision time. The gap between this close and the
    """
    ts: date
    cash: float
    positions: dict[str, float]    # symbol -> shares
    last_prices: dict[str, float]  # symbol -> close of t
    equity: float
