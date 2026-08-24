"""Rolling price history for covariance estimation.

The engine appends closing prices as each market event is processed and passes a
read-only view to the allocator at decision time. The returns window therefore ends
at the current close and cannot include future observations.
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Sequence

import numpy as np

from .events import MarketEvent


class RollingHistory:
    def __init__(self, maxlen: int = 400):
        self._maxlen = maxlen
        self._closes: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=maxlen)
        )

    def update(self, e: MarketEvent) -> None:
        for sym, bar in e.bars.items():
            self._closes[sym].append(bar.close)

    def returns(
        self, symbols: Sequence[str], lookback: int
    ) -> tuple[list[str], np.ndarray]:
        """Trailing simple returns for `symbols` over `lookback` periods.

        Returns (usable_symbols, R) with R shape (lookback, k). Symbols without at
        least lookback+1 observations are dropped so the matrix stays rectangular
            rectangular and fully observed without imputation.
        """
        need = lookback + 1
        usable: list[str] = []
        cols: list[np.ndarray] = []
        for s in symbols:
            hist = self._closes.get(s)
            if hist is None or len(hist) < need:
                continue
            prices = np.array(list(hist)[-need:], dtype=float)
            rets = prices[1:] / prices[:-1] - 1.0
            usable.append(s)
            cols.append(rets)
        if not usable:
            return [], np.empty((0, 0))
        return usable, np.column_stack(cols)
