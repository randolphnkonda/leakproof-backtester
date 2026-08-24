"""The research harness: fan run_backtest across a grid, collect a SweepResult.

This is where the concurrency skill goal is actually satisfied, and where the N
for the deflated Sharpe is born. Three ideas hold it together:

  1. The unit of parallelism is run_backtest(cfg) -- a pure function of a frozen,
     picklable config. That is the only reason process-pool fan-out is clean: no
     shared state, no locks, each worker fully independent.

  2. Each trial is reduced in the worker to a compact TrialRecord (moments, not
     the equity curve). This bounds the data shipped back across process
     boundaries to O(1) per trial regardless of series length -- so the sweep
     scales to thousands of configs without drowning in pickled Series.

  3. Determinism lets us throw away every equity curve during the sweep and
     cheaply reconstruct just the winner's curve afterwards by re-running its
     config. SweepResult.best_equity_curve() does exactly that.

Statistical note carried forward to the deflated-Sharpe step: the multiple-testing
N is the number of STRATEGY configurations tried on the SAME data. So build_grid
sweeps strategy params at a fixed data seed by default; the seed axis exists for
robustness checks and must not be used to inflate N naively.
"""
from __future__ import annotations

import itertools
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from time import perf_counter
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .config import BacktestConfig
from .engine import run_backtest

_ANNUALISATION = 252


@dataclass(frozen=True)
class TrialRecord:
    """Summary statistics for a single backtest, sized for inter-process transfer."""

    config: BacktestConfig
    sharpe: float
    n_periods: int        # number of return observations
    ann_return: float
    ann_vol: float
    skew: float           # Fisher skewness of periodic returns
    kurt_excess: float    # excess kurtosis, zero under normality
    max_drawdown: float
    n_trades: int


def _summarise(cfg: BacktestConfig) -> TrialRecord:
    """Run one backtest and reduce it to summary statistics."""
    res = run_backtest(cfg)
    ec = res.equity_curve
    rets = ec.pct_change().dropna()
    n = int(len(rets))
    if n >= 2 and rets.std(ddof=1) > 0:
        ann_return = float((1.0 + rets.mean()) ** _ANNUALISATION - 1.0)
        ann_vol = float(rets.std(ddof=1) * np.sqrt(_ANNUALISATION))
        skew = float(rets.skew())
        kurt_excess = float(rets.kurt())  # pandas returns excess kurtosis
    else:
        ann_return = ann_vol = skew = kurt_excess = float("nan")
    running_max = ec.cummax()
    max_dd = float((ec / running_max - 1.0).min())
    return TrialRecord(
        config=cfg,
        sharpe=float(res.sharpe),
        n_periods=n,
        ann_return=ann_return,
        ann_vol=ann_vol,
        skew=skew,
        kurt_excess=kurt_excess,
        max_drawdown=max_dd,
        n_trades=int(res.n_trades),
    )


# Module-level entry point so the callable is picklable.
def _run_one(cfg: BacktestConfig) -> TrialRecord:
    return _summarise(cfg)


def _config_key(cfg: BacktestConfig) -> tuple:
    """Ordering key giving reproducible results regardless of completion order."""
    return (cfg.factors, cfg.factor_weights, cfg.lookback_months,
            cfg.skip_months, cfg.n_long, cfg.seed)


@dataclass(frozen=True)
class SweepResult:
    """Collected results of a parameter sweep."""

    trials: tuple[TrialRecord, ...]   # canonical order
    wall_seconds: float
    n_workers: int
    parallel: bool

    @property
    def n_trials(self) -> int:
        return len(self.trials)

    @property
    def sharpes(self) -> np.ndarray:
        """Sharpe ratio of every trial, used to estimate the trial variance."""
        return np.array([t.sharpe for t in self.trials], dtype=float)

    @property
    def best(self) -> TrialRecord:
        return max(self.trials, key=lambda t: (np.isfinite(t.sharpe), t.sharpe))

    def to_frame(self) -> pd.DataFrame:
        rows = []
        for t in self.trials:
            c = t.config
            rows.append({
                "factors": c.factors,
                "weights": c.factor_weights or "equal",
                "lookback_months": c.lookback_months,
                "skip_months": c.skip_months,
                "n_long": c.n_long,
                "seed": c.seed,
                "sharpe": t.sharpe,
                "ann_return": t.ann_return,
                "ann_vol": t.ann_vol,
                "max_drawdown": t.max_drawdown,
                "skew": t.skew,
                "kurt_excess": t.kurt_excess,
                "n_periods": t.n_periods,
                "n_trades": t.n_trades,
            })
        return pd.DataFrame(rows)

    def best_equity_curve(self) -> pd.Series:
        """Recompute the best configuration's equity curve.

        Equity curves are not retained during the sweep. Runs are deterministic, so
        re-running the winning configuration reproduces its curve exactly.
        """
        return run_backtest(self.best.config).equity_curve



def build_grid(
    base: BacktestConfig,
    *,
    lookback_months: Iterable[int] | None = None,
    skip_months: Iterable[int] | None = None,
    n_long: Iterable[int] | None = None,
    seeds: Iterable[int] | None = None,
    factors: Iterable[str] | None = None,
    factor_weights: Iterable[str] | None = None,
) -> list[BacktestConfig]:
    """Build the Cartesian product of the swept dimensions over a base config.

    Combinations where the skip window is not strictly inside the lookback window,
    or where a weight specification does not match its factor specification, are
    omitted.

    Returns:
        List of configurations to evaluate.
    """
    lb = list(lookback_months) if lookback_months is not None else [base.lookback_months]
    sk = list(skip_months) if skip_months is not None else [base.skip_months]
    nl = list(n_long) if n_long is not None else [base.n_long]
    sd = list(seeds) if seeds is not None else [base.seed]
    fc = list(factors) if factors is not None else [base.factors]
    fw = list(factor_weights) if factor_weights is not None else [base.factor_weights]

    grid: list[BacktestConfig] = []
    for lookback, skip, n, seed, fspec, wspec in itertools.product(lb, sk, nl, sd, fc, fw):
        if skip >= lookback:
            continue
        if wspec and len(str(wspec).split(",")) != len(str(fspec).split(",")):
            continue
        grid.append(replace(
            base, lookback_months=lookback, skip_months=skip, n_long=n, seed=seed,
            factors=fspec, factor_weights=wspec,
        ))
    return grid



def run_sweep(
    configs: Sequence[BacktestConfig],
    *,
    max_workers: int | None = None,
    parallel: bool = True,
) -> SweepResult:
    """Evaluate every configuration, in parallel where beneficial.

    Args:
        configs: Configurations to evaluate.
        max_workers: Worker process count, or None for the system default.
        parallel: Set False to force serial execution. Process-pool overhead
            exceeds its benefit on a single core.

    Returns:
        SweepResult in canonical order. Serial and parallel execution produce
        identical results.
    """
    t0 = perf_counter()
    use_parallel = parallel and len(configs) > 1
    if use_parallel:
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            records = list(ex.map(_run_one, configs))
        workers = max_workers or (os.cpu_count() or 1)
    else:
        records = [_run_one(c) for c in configs]
        workers = 1
    wall = perf_counter() - t0

    records.sort(key=lambda r: _config_key(r.config))
    return SweepResult(
        trials=tuple(records),
        wall_seconds=wall,
        n_workers=workers,
        parallel=use_parallel,
    )
