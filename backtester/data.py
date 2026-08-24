"""Synthetic market data for testing and demonstration.

Generates deterministic geometric Brownian motion bars behind the DataHandler
interface, allowing the engine and research harness to be exercised without a
populated store. See store_data.StoreDataHandler for the production data source.

Panels are cached per (start, end, seed, symbols, regime) so that a parameter sweep
over a fixed data seed generates its price series once. Prices are treated as split
and dividend adjusted throughout.
"""
from __future__ import annotations

from datetime import date
from functools import lru_cache
from typing import Iterator

import numpy as np
import pandas as pd

from .events import Bar, MarketEvent

_SYNTHETIC_SYMBOLS = (
    "AAA", "BBB", "CCC", "DDD", "EEE",
    "FFF", "GGG", "HHH", "III", "JJJ",
)


class _Panel:
    """Column-oriented price panel indexed by a shared trading calendar."""
    __slots__ = ("dates", "symbols", "open", "high", "low", "close", "volume")

    def __init__(self, dates, symbols, open_, high, low, close, volume):
        self.dates = dates
        self.symbols = symbols
        self.open = open_
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume


@lru_cache(maxsize=32)
def _build_panel(start: date, end: date, seed: int,
                 symbols: tuple[str, ...], regime: str = "dispersed") -> _Panel:
    """Cached, deterministic GBM panel. Same args -> same data.

    regime="dispersed": each name gets its own drift/vol, so cross-sectional
    regime="null": all names share zero drift and equal vol, so momentum selects
      on pure noise. Any positive Sharpe in a sweep of this regime is selection

    Cached at module level so repeated trials in a sweep (which share data) pay
    the simulation cost exactly once per worker process.
    """
    dates = [d.date() for d in pd.bdate_range(start, end)]
    n = len(dates)
    open_, high, low, close, volume = ({} for _ in range(5))
    for i, sym in enumerate(symbols):
        rng = np.random.default_rng(seed * 1000 + i)
        if regime == "null":
            mu = 0.0
            vol = 0.25 / np.sqrt(252.0)
        else:
            mu = rng.uniform(0.02, 0.18) / 252.0
            vol = rng.uniform(0.15, 0.45) / np.sqrt(252.0)
        shocks = rng.normal(mu, vol, size=n)
        c = 100.0 * np.exp(np.cumsum(shocks))
        intraday = np.abs(rng.normal(0, vol, size=n)) * c
        o = np.concatenate([[c[0]], c[:-1]])  # open approximates the prior close
        close[sym] = c
        open_[sym] = o
        high[sym] = np.maximum(o, c) + intraday * 0.5
        low[sym] = np.minimum(o, c) - intraday * 0.5
        volume[sym] = rng.uniform(1e6, 5e6, size=n)
    return _Panel(dates, list(symbols), open_, high, low, close, volume)


class SyntheticDataHandler:
    """DataHandler backed by generated price series."""

    def __init__(self, start: date, end: date, seed: int = 0,
                 symbols: tuple[str, ...] | None = None,
                 regime: str = "dispersed"):
        self.start = start
        self.end = end
        self.seed = seed
        self.regime = regime
        self.symbols = tuple(symbols) if symbols else _SYNTHETIC_SYMBOLS
        self._panel = _build_panel(start, end, seed, self.symbols, regime)

    def stream(self) -> Iterator[MarketEvent]:
        p = self._panel
        for i, d in enumerate(p.dates):
            live = self.get_universe(d)
            bars = {
                sym: Bar(
                    symbol=sym, ts=d,
                    open=float(p.open[sym][i]), high=float(p.high[sym][i]),
                    low=float(p.low[sym][i]), close=float(p.close[sym][i]),
                    volume=float(p.volume[sym][i]),
                )
                for sym in p.symbols if sym in live
            }
            yield MarketEvent(ts=d, bars=bars)

    def get_universe(self, as_of: date) -> set[str]:
        # Stub: fixed membership. Real impl walks index add/drop history backwards
        # from today's constituents to reconstruct point-in-time membership.
        return set(self.symbols)


def StooqDataHandler(*args, **kwargs):  # pragma: no cover
    """Deprecated. Build a store with build_store.py and use StoreDataHandler."""
    raise NotImplementedError(
        "Build a store with build_store.py, then set data_source='store'. "
        "StoreDataHandler reads the BarStore directly."
    )
