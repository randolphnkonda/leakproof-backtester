"""DataHandler backed by a persisted bar store.

Implements the same interface as SyntheticDataHandler, so the engine, strategy,
allocator, and broker are unchanged when switching to stored market data.

A symbol contributes a bar on a given date only if it has price data for that date
and is an index constituent on that date, which handles listings and delistings
without special-case logic.
"""
from __future__ import annotations

from datetime import date
from typing import Iterator

import numpy as np

from .events import Bar, MarketEvent
from .pipeline.store import BarStore


class StoreDataHandler:
    def __init__(self, store: BarStore, start: date, end: date,
                 symbols: list[str] | None = None,
                 exclude: dict[str, str] | set[str] | None = None):
        self.store = store
        self.start = start
        self.end = end
        # Excluded symbols are removed at the data layer so that no downstream
        # component needs data-quality logic.
        self.excluded = dict(exclude) if isinstance(exclude, dict) else {
            s_: "excluded" for s_ in (exclude or set())}
        self.symbols = [s_ for s_ in (symbols or store.symbols())
                        if s_ not in self.excluded]
        if not self.symbols or not store.has_bars():
            raise RuntimeError(
                "The data store is empty (no symbols found). Build it first, e.g. "
                "`python3 build_store.py --source fixture --store <path>`, and make sure "
                "the backend matches the store format (a Parquet store needs duckdb; a "
                "sqlite store is read with sqlite). open_store() auto-detects the format."
            )
        self._dates, self._data = store.panel(self.symbols, start, end)
        if not self._dates:
            raise RuntimeError(
                f"The store has no price bars between {start} and {end}. Rebuild it for "
                f"this date range, e.g. `python3 build_store.py --source fixture "
                f"--start {start} --end {end}`.")
        # Cache membership intervals once; compute the universe per day in memory.
        self._members = store._fetch_membership()

    def get_universe(self, as_of: date) -> set[str]:
        m = self._members
        if not len(m):
            return set(self.symbols)
        live = [
            (sd is not None and sd <= as_of) and (ed is None or ed > as_of)
            for sd, ed in zip(m["start_date"], m["end_date"])
        ]
        return set(m.loc[live, "symbol"])

    def stream(self) -> Iterator[MarketEvent]:
        for i, d in enumerate(self._dates):
            members = self.get_universe(d)
            bars: dict[str, Bar] = {}
            for sym in self.symbols:
                if sym not in members:
                    continue
                cell = self._data.get(sym)
                if cell is None:
                    continue
                c = cell["close"][i]
                if not np.isfinite(c):
                    continue
                bars[sym] = Bar(
                    symbol=sym, ts=d,
                    open=float(cell["open"][i]), high=float(cell["high"][i]),
                    low=float(cell["low"][i]), close=float(c),
                    volume=float(cell["volume"][i]),
                )
            yield MarketEvent(ts=d, bars=bars)
