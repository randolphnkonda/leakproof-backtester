"""Multi-factor composite signal generation.

The strategy accumulates closing prices from the market event stream, so it observes
only data that has already been processed by the event loop.

On each rebalance date the composite score is built in three steps:

    1. Each factor scores every symbol for which it has sufficient history.
    2. Scores are standardised across the cross-section, placing factors on a common
       scale.
    3. Standardised scores are combined as a weighted average over the factors that
       are available for that symbol, with weights renormalised accordingly.

Renormalisation in step 3 handles factors with differing warmup requirements. Treating
an unavailable factor as zero would bias the composite toward the neutral score.
"""
from __future__ import annotations

from collections import defaultdict, deque
from datetime import date

import numpy as np

from .config import BacktestConfig
from .events import MarketEvent, SignalEvent
from .factors import build_factor, zscore


def _parse_factors(cfg: BacktestConfig):
    """Resolve factor instances and normalised weights from configuration.

    Raises:
        ValueError: If the weight count does not match the factor count, or if the
            weights sum to zero in absolute value.
    """
    specs = [s for s in str(cfg.factors).split(",") if s.strip()]
    if not specs:
        specs = ["momentum"]
    factors = [build_factor(s, cfg.lookback_months, cfg.skip_months) for s in specs]

    if cfg.factor_weights:
        raw = [float(x) for x in str(cfg.factor_weights).split(",") if x.strip() != ""]
        if len(raw) != len(factors):
            raise ValueError(
                f"factor_weights has {len(raw)} entries but factors has {len(factors)}")
        w = np.asarray(raw, dtype=float)
    else:
        w = np.ones(len(factors), dtype=float)
    total = np.abs(w).sum()
    if total <= 0:
        raise ValueError("factor_weights must not sum to zero in absolute value")
    return factors, w / total


class CompositeStrategy:
    def __init__(self, cfg: BacktestConfig):
        self.cfg = cfg
        self.factors, self.weights = _parse_factors(cfg)
        self.max_warmup = max(f.warmup for f in self.factors)
        self._closes: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=self.max_warmup + 1)
        )
        self._last_rebalance_month: tuple[int, int] | None = None

    @property
    def factor_names(self) -> list[str]:
        return [f.name for f in self.factors]

    def _is_rebalance_day(self, d: date) -> bool:
        """Return True on the first observed trading day of each month."""
        key = (d.year, d.month)
        if key != self._last_rebalance_month:
            self._last_rebalance_month = key
            return True
        return False

    def on_market(self, e: MarketEvent) -> SignalEvent | None:
        for sym, bar in e.bars.items():
            self._closes[sym].append(bar.close)

        if not self._is_rebalance_day(e.ts):
            return None

        # Candidates are the symbols tradeable on this date.
        symbols = [s for s in e.bars if len(self._closes[s]) >= 2]
        if len(symbols) < 2:
            return None

        # Score each factor, then standardise across the cross-section.
        z_by_factor: list[np.ndarray] = []
        for f in self.factors:
            raw = []
            for s in symbols:
                hist = self._closes[s]
                if len(hist) < f.warmup:
                    raw.append(np.nan)
                else:
                    raw.append(f.score(np.asarray(hist, dtype=float)))
            z_by_factor.append(zscore(raw))

        if not z_by_factor:
            return None
        Z = np.vstack(z_by_factor)                # (n_factors x n_symbols)
        W = self.weights.reshape(-1, 1)

        # Weighted average over the factors available for each symbol.
        present = np.isfinite(Z)
        Zf = np.where(present, Z, 0.0)
        num = (Zf * W).sum(axis=0)
        den = (present * np.abs(W)).sum(axis=0)
        composite = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)

        scores = {s: float(v) for s, v in zip(symbols, composite) if np.isfinite(v)}
        if not scores:
            return None
        return SignalEvent(ts=e.ts, scores=scores)


# Retained for backward compatibility with single-factor configurations.
MomentumStrategy = CompositeStrategy
