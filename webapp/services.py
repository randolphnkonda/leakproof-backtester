"""Application logic for the web interface.

Functions here take primitives and return plain data structures, with no dependency
on Streamlit, so the interface can be tested without a browser or running server.
app.py renders the results returned by these functions.
"""
from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd

from backtester import BacktestConfig, analyze_sweep, build_grid, run_backtest, run_sweep
from backtester.data import _SYNTHETIC_SYMBOLS, SyntheticDataHandler
from backtester.factors import AVAILABLE_FACTORS
from backtester.history import RollingHistory
from backtester.optimize import active_backend, solve_min_variance
from backtester.pipeline.store import open_store
from backtester.risk_model import estimate_covariance, ledoit_wolf_cov

_ANN = math.sqrt(252)

ALLOCATORS = ("equal_weight", "min_variance", "max_decorrelation")
FACTORS = AVAILABLE_FACTORS
COV_METHODS = ("ledoit_wolf", "sample")
DATA_CHOICES = ("synthetic (real edge)", "synthetic (null control)", "store (real data)")


def optimiser_backend() -> str:
    return active_backend()


def store_version() -> str:
    from backtester.pipeline.store import STORE_VERSION
    return STORE_VERSION


def store_exists(store_path: str) -> bool:
    """Return True if the path holds a store containing price bars.

    A directory or a membership file without bars is not a usable store.
    """
    try:
        return open_store(store_path).has_bars()
    except Exception:
        return False


def build_fixture_store(store_path: str, start: date, end: date) -> str:
    """Build a fixture store and return the backend class name used."""
    from build_store import build
    build("fixture", store_path, [], start, end, backend="auto", seed=0)
    return type(open_store(store_path)).__name__


def _make_config(data_choice: str, store_path: str, start: date, end: date,
                 lookback: int, skip: int, n_long: int, allocator: str,
                 cov_method: str, cov_lookback: int, seed: int,
                 factors: str = "momentum") -> BacktestConfig:
    if data_choice.startswith("store"):
        source, regime = "store", "dispersed"
    elif "null" in data_choice:
        source, regime = "synthetic", "null"
    else:
        source, regime = "synthetic", "dispersed"
    return BacktestConfig(
        start=start, end=end, lookback_months=lookback, skip_months=skip,
        n_long=n_long, allocator=allocator, cov_method=cov_method,
        cov_lookback_days=cov_lookback, seed=seed, factors=factors,
        data_source=source, data_regime=regime, store_path=store_path,
    )


def metrics_from_curve(ec: pd.Series) -> dict:
    rets = ec.pct_change().dropna()
    if len(rets) < 2 or rets.std(ddof=1) == 0:
        return {"sharpe": float("nan"), "ann_return": float("nan"),
                "ann_vol": float("nan"), "max_drawdown": float("nan"),
                "total_return": float("nan"), "n_periods": len(rets)}
    ann_ret = (1 + rets.mean()) ** 252 - 1
    ann_vol = rets.std(ddof=1) * _ANN
    return {
        "sharpe": ann_ret / ann_vol if ann_vol else float("nan"),
        "ann_return": float(ann_ret),
        "ann_vol": float(ann_vol),
        "max_drawdown": float((ec / ec.cummax() - 1).min()),
        "total_return": float(ec.iloc[-1] / ec.iloc[0] - 1),
        "n_periods": int(len(rets)),
    }


def single_backtest(data_choice: str, store_path: str, start: date, end: date,
                    lookback: int, skip: int, n_long: int, allocator: str,
                    cov_method: str, cov_lookback: int, seed: int,
                    factors: str = "momentum") -> dict:
    cfg = _make_config(data_choice, store_path, start, end, lookback, skip, n_long,
                       allocator, cov_method, cov_lookback, seed, factors)
    res = run_backtest(cfg)
    return {"equity": res.equity_curve, "n_trades": int(res.n_trades),
            "metrics": metrics_from_curve(res.equity_curve)}


def sweep_and_deflate(data_choice: str, store_path: str, start: date, end: date,
                      lookbacks: tuple[int, ...], skips: tuple[int, ...],
                      n_longs: tuple[int, ...], allocator: str, cov_method: str,
                      cov_lookback: int, seed: int,
                      factor_specs: tuple[str, ...] = ("momentum",)) -> dict:
    base = _make_config(data_choice, store_path, start, end, 12, 1, 5, allocator,
                        cov_method, cov_lookback, seed)
    grid = build_grid(base, lookback_months=list(lookbacks),
                      skip_months=list(skips), n_long=list(n_longs),
                      factors=list(factor_specs))
    sweep = run_sweep(grid, parallel=False)
    report = analyze_sweep(sweep)
    df = sweep.to_frame().sort_values("sharpe", ascending=False).reset_index(drop=True)
    return {
        "n_trials": sweep.n_trials,
        "sharpes": sweep.sharpes,
        "table": df,
        "report": report,
        "best_equity": sweep.best_equity_curve(),
    }


def allocator_comparison(data_choice: str, store_path: str, start: date, end: date,
                         lookback: int, skip: int, n_long: int, cov_method: str,
                         cov_lookback: int, seed: int,
                         factors: str = "momentum") -> pd.DataFrame:
    rows = []
    for alloc in ALLOCATORS:
        r = single_backtest(data_choice, store_path, start, end, lookback, skip,
                            n_long, alloc, cov_method, cov_lookback, seed, factors)
        m = r["metrics"]
        rows.append({"allocator": alloc, "sharpe": m["sharpe"],
                     "ann_return": m["ann_return"], "ann_vol": m["ann_vol"],
                     "max_drawdown": m["max_drawdown"], "trades": r["n_trades"]})
    return pd.DataFrame(rows)


def _window_returns(lookback: int, seed: int) -> np.ndarray:
    data = SyntheticDataHandler(date(2015, 1, 1), date(2023, 12, 31), seed=seed)
    hist = RollingHistory(maxlen=lookback + 5)
    for e in data.stream():
        hist.update(e)
    _, R = hist.returns(list(_SYNTHETIC_SYMBOLS), lookback)
    return R


def weight_stability(window: int, B: int, seed: int) -> dict:
    R = _window_returns(window, seed)
    k = R.shape[1]
    _, shrink = ledoit_wolf_cov(R)

    def boot(method):
        rng = np.random.default_rng(1)
        T = R.shape[0]
        W = [solve_min_variance(estimate_covariance(R[rng.integers(0, T, T)], method=method))
             for _ in range(B)]
        return np.array(W)

    Ws, Wl = boot("sample"), boot("ledoit_wolf")
    inst_s = float(Ws.std(axis=0, ddof=1).mean())
    inst_l = float(Wl.std(axis=0, ddof=1).mean())
    return {
        "symbols": list(_SYNTHETIC_SYMBOLS),
        "sample_mean": Ws.mean(0), "sample_std": Ws.std(0),
        "lw_mean": Wl.mean(0), "lw_std": Wl.std(0),
        "instability_sample": inst_s, "instability_lw": inst_l,
        "reduction": 1 - inst_l / inst_s if inst_s else float("nan"),
        "shrinkage": float(shrink), "k": k, "window": window,
    }


def store_summary(store_path: str) -> dict:
    store = open_store(store_path)
    m = store._fetch_membership()
    syms = store.symbols()
    return {"backend": type(store).__name__, "n_symbols": len(syms),
            "symbols": syms, "n_intervals": len(m), "membership": m}


def store_universe(store_path: str, as_of: date) -> list[str]:
    return sorted(open_store(store_path).universe_as_of(as_of))


def store_prices(store_path: str, symbols: list[str], start: date, end: date) -> pd.DataFrame:
    store = open_store(store_path)
    dates, data = store.panel(symbols, start, end)
    if not dates:
        return pd.DataFrame()
    frame = {s: data[s]["close"] for s in symbols if s in data}
    return pd.DataFrame(frame, index=pd.DatetimeIndex(dates))


def factor_comparison(data_choice: str, store_path: str, start: date, end: date,
                      lookback: int, skip: int, n_long: int, allocator: str,
                      cov_method: str, cov_lookback: int, seed: int,
                      specs: tuple[str, ...]) -> pd.DataFrame:
    """Evaluate each factor specification under identical settings."""
    rows = []
    for spec in specs:
        r = single_backtest(data_choice, store_path, start, end, lookback, skip,
                            n_long, allocator, cov_method, cov_lookback, seed, spec)
        m = r["metrics"]
        rows.append({"factors": spec, "sharpe": m["sharpe"],
                     "ann_return": m["ann_return"], "ann_vol": m["ann_vol"],
                     "max_drawdown": m["max_drawdown"], "trades": r["n_trades"]})
    return pd.DataFrame(rows)
