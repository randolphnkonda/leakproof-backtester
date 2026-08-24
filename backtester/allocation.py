"""Portfolio allocation.

Allocators convert factor scores into target weights and then into orders:

    EqualWeightAllocator    equal weights across the selected securities
    MinVarianceAllocator    convex optimisation over an estimated covariance matrix

Both size positions using day t closing prices, the latest information available at
decision time. Orders fill at the following open, so realised entry prices differ.
The conversion from weights to orders is shared, so allocators differ only in how
weights are determined.
"""
from __future__ import annotations

import numpy as np

from .config import BacktestConfig
from .events import OrderEvent, PortfolioState, SignalEvent, Side
from .optimize import solve_max_decorrelation, solve_min_variance
from .protocols import HistoryView
from .risk_model import estimate_covariance


def _select_targets(sig: SignalEvent, n_long: int) -> list[str]:
    ranked = sorted(sig.scores.items(), key=lambda kv: kv[1], reverse=True)
    return [s for s, _ in ranked[:n_long]]


def _orders_from_weights(
    weights: dict[str, float], pf: PortfolioState, ts
) -> list[OrderEvent]:
    """Convert target weights into orders against current holdings."""
    target_shares: dict[str, float] = {}
    for sym, wgt in weights.items():
        px = pf.last_prices.get(sym)
        if px is None or px <= 0 or wgt <= 0:
            continue
        target_shares[sym] = (pf.equity * wgt) / px

    # Held securities absent from the target set receive a full exit order. This
    # includes securities that have left the universe, which would otherwise remain
    # in the portfolio indefinitely.
    orders: list[OrderEvent] = []
    for sym in set(pf.positions) | set(target_shares):
        delta = target_shares.get(sym, 0.0) - pf.positions.get(sym, 0.0)
        if abs(delta) < 1e-9:
            continue
        side = Side.BUY if delta > 0 else Side.SELL
        orders.append(OrderEvent(ts=ts, symbol=sym, side=side, quantity=abs(delta)))
    return orders


class EqualWeightAllocator:
    def __init__(self, cfg: BacktestConfig):
        self.cfg = cfg

    def target_orders(
        self, sig: SignalEvent, pf: PortfolioState, history: HistoryView
    ) -> list[OrderEvent]:
        targets = _select_targets(sig, self.cfg.n_long)
        if not targets:
            return []
        w = 1.0 / len(targets)
        return _orders_from_weights({s: w for s in targets}, pf, sig.ts)


class MinVarianceAllocator:
    """Weight the selected securities by convex optimisation.

    Falls back to equal weighting when insufficient return history is available to
    estimate a covariance matrix.
    """

    def __init__(self, cfg: BacktestConfig):
        self.cfg = cfg

    def target_orders(
        self, sig: SignalEvent, pf: PortfolioState, history: HistoryView
    ) -> list[OrderEvent]:
        targets = _select_targets(sig, self.cfg.n_long)
        if not targets:
            return []

        syms, R = history.returns(targets, self.cfg.cov_lookback_days)
        if len(syms) < 2:
            w = 1.0 / len(targets)
            return _orders_from_weights({s: w for s in targets}, pf, sig.ts)

        Sigma = estimate_covariance(np.asarray(R), method=self.cfg.cov_method)
        if self.cfg.allocator == "max_decorrelation":
            w_vec = solve_max_decorrelation(Sigma, max_weight=self.cfg.max_weight)
        else:
            w_vec = solve_min_variance(Sigma, max_weight=self.cfg.max_weight)

        weights = {s: float(w) for s, w in zip(syms, w_vec)}
        return _orders_from_weights(weights, pf, sig.ts)
