"""Pre-trade risk controls.

Orders pass through this stage before reaching the broker. The default
implementation applies no constraints; position limits, exposure caps, sector
neutrality, and turnover controls belong here.
"""
from __future__ import annotations

from .events import OrderEvent, PortfolioState


class PassThroughRiskManager:
    def vet(
        self, orders: list[OrderEvent], pf: PortfolioState
    ) -> list[OrderEvent]:
        return orders
