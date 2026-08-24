"""Backtest configuration and result types.

BacktestConfig is frozen and hashable so that a configuration doubles as a cache key
and as the unit of work distributed across sweep workers. The number of distinct
configurations evaluated is the trial count used by the deflated Sharpe ratio.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd


@dataclass(frozen=True)
class BacktestConfig:
    start: date
    end: date
    universe_id: str = "synthetic_10"
    factor: str = "momentum_12_1"
    factors: str = "momentum"        # comma-separated specs, e.g. "momentum,lowvol"
    factor_weights: str = ""         # optional comma-separated weights; blank = equal
    lookback_months: int = 12
    skip_months: int = 1
    rebalance: str = "monthly"
    n_long: int = 5
    initial_cash: float = 1_000_000.0
    commission_bps: float = 0.0
    slippage_bps: float = 0.0
    seed: int = 0                 # makes each synthetic run deterministic
    data_regime: str = "dispersed"  # "dispersed": heterogeneous drift (real edge);
    allocator: str = "equal_weight"  # "equal_weight" | "min_variance" | "max_decorrelation"
    cov_method: str = "ledoit_wolf"  # "ledoit_wolf" | "sample"
    cov_lookback_days: int = 252     # trailing window for covariance estimation
    max_weight: float = 1.0          # per-name concentration cap (1.0 = uncapped)
    data_source: str = "synthetic"   # "synthetic" | "store" (real data via BarStore)
    store_path: str = "store"        # BarStore root when data_source == "store"
    store_backend: str = "auto"      # "auto" | "duckdb" | "sqlite"
    quality_filter: bool = True      # exclude stubs, stale series, price artifacts
    min_obs: int = 252               # minimum observations for a symbol to be usable
    max_abs_daily: float = 0.60      # single-day move above this indicates bad data


@dataclass(frozen=True)
class BacktestResult:
    config: BacktestConfig
    equity_curve: pd.Series        # indexed by date
    sharpe: float
    n_trades: int
    extras: dict = field(default_factory=dict)
