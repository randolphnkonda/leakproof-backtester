"""Backtest event loop.

Each market event is processed in a fixed three-phase order:

    1. Execute orders queued on the previous event against this event's opens.
    2. Mark the portfolio to market on this event's closes and record equity.
    3. Generate new orders from this event's closes, queued for the next event.

Executing before signal generation is required for correctness: it guarantees that
orders derived from day t data cannot fill at day t prices.

run_backtest is a pure function of its configuration, making it deterministic,
picklable, and safe to distribute across worker processes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .allocation import EqualWeightAllocator, MinVarianceAllocator
from .config import BacktestConfig, BacktestResult
from .data import SyntheticDataHandler
from .execution import SimulatedBroker
from .history import RollingHistory
from .portfolio import SimplePortfolio
from .risk import PassThroughRiskManager
from .strategy import MomentumStrategy

_ANNUALISATION = 252

# Quality screening scans the full store. Results are memoised per store, window,
# and threshold set so that parameter sweeps incur the cost once per process.
_EXCL_CACHE: dict[tuple, dict] = {}


def _cached_exclusions(cfg: BacktestConfig) -> dict:
    from .pipeline.store import open_store
    from .quality import QualityThresholds, build_exclusions
    key = (cfg.store_path, cfg.start, cfg.end, cfg.min_obs, cfg.max_abs_daily)
    if key not in _EXCL_CACHE:
        store = open_store(cfg.store_path, backend=cfg.store_backend)
        _EXCL_CACHE[key] = build_exclusions(
            store, cfg.start, cfg.end,
            QualityThresholds(min_obs=cfg.min_obs, max_abs_daily=cfg.max_abs_daily),
            cache_dir=cfg.store_path,
        )
    return _EXCL_CACHE[key]


def _build_data_handler(cfg: BacktestConfig):
    """Construct the data handler selected by the configuration."""
    if cfg.data_source == "store":
        from .pipeline.store import open_store
        from .store_data import StoreDataHandler
        store = open_store(cfg.store_path, backend=cfg.store_backend)
        excl = _cached_exclusions(cfg) if cfg.quality_filter else {}
        return StoreDataHandler(store, start=cfg.start, end=cfg.end, exclude=excl)
    return SyntheticDataHandler(
        start=cfg.start, end=cfg.end, seed=cfg.seed, regime=cfg.data_regime,
    )


def _build_allocator(cfg: BacktestConfig):
    if cfg.allocator in ("min_variance", "max_decorrelation"):
        return MinVarianceAllocator(cfg)
    return EqualWeightAllocator(cfg)


def run_backtest(cfg: BacktestConfig) -> BacktestResult:
    data = _build_data_handler(cfg)
    strategy = MomentumStrategy(cfg)
    allocator = _build_allocator(cfg)
    risk = PassThroughRiskManager()
    broker = SimulatedBroker(cfg)
    portfolio = SimplePortfolio(cfg.initial_cash)
    history = RollingHistory(maxlen=cfg.cov_lookback_days + 5)

    timestamps: list = []
    equity: list[float] = []

    for event in data.stream():
        # 1. Execute orders queued on the previous event.
        for fill in broker.execute_pending(event):
            portfolio.on_fill(fill)

        # 2. Mark to market and record equity.
        broker.observe(event)
        portfolio.mark_to_market(event)
        history.update(event)
        state = portfolio.state()
        timestamps.append(event.ts)
        equity.append(state.equity)

        # 3. Generate orders for execution on the next event.
        signal = strategy.on_market(event)
        if signal is not None:
            orders = allocator.target_orders(signal, state, history)
            orders = risk.vet(orders, state)
            for order in orders:
                broker.queue(order)

    equity_curve = pd.Series(equity, index=pd.DatetimeIndex(timestamps), name="equity")
    sharpe = _sharpe(equity_curve)
    return BacktestResult(
        config=cfg,
        equity_curve=equity_curve,
        sharpe=sharpe,
        n_trades=portfolio.n_fills,
    )


def _sharpe(equity_curve: pd.Series) -> float:
    rets = equity_curve.pct_change().dropna()
    if rets.std(ddof=1) == 0 or len(rets) < 2:
        return float("nan")
    return float(
        np.sqrt(_ANNUALISATION) * rets.mean() / rets.std(ddof=1)
    )
