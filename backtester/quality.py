"""Data-quality screening.

Three defects occur in real equity price data and produce plausible but incorrect
factor results:

    Stubs                   securities acquired or delisted near the start of the
                            window leave too few observations for any statistic.
    Stale series            repeated identical closes measure near-zero volatility
                            and rank at the top of a low-volatility screen.
    Unadjusted actions      an unadjusted split appears as an extreme single-day
                            return, distorting momentum and drawdown statistics.

Exclusions carry a reason string so that screening decisions are reportable rather
than silent.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class QualityThresholds:
    min_obs: int = 252            # minimum observations required
    max_zero_share: float = 0.20  # maximum share of returns that are exactly zero
    max_flat_run: int = 5         # maximum run of identical consecutive closes
    max_abs_daily: float = 0.60   # maximum plausible single-day absolute return


def _longest_flat_run(c: np.ndarray) -> int:
    """Return the longest run of identical consecutive prices."""
    if len(c) < 2:
        return 0
    same = c[1:] == c[:-1]
    if not same.any():
        return 0
    # Run lengths of True blocks: index of each element minus the index of the last
    # False before it.
    idx = np.flatnonzero(~same)
    if len(idx) == 0:
        return int(same.sum())
    pos = np.arange(len(same))
    last_false = np.maximum.accumulate(np.where(~same, pos, -1))
    return int((pos - last_false)[same].max())


def symbol_stats(store, symbols, start: date, end: date, panel=None) -> pd.DataFrame:
    """Compute per-symbol data-quality statistics.

    Args:
        store: Bar store to read from.
        symbols: Symbols to evaluate.
        start: Window start date.
        end: Window end date.
        panel: Optional pre-loaded (dates, data) pair, avoiding a second scan.

    Returns:
        DataFrame with one row per symbol.
    """
    dates, data = panel if panel is not None else store.panel(list(symbols), start, end)
    rows = []
    for sym in symbols:
        cell = data.get(sym)
        if cell is None:
            continue
        close = cell["close"]
        c = close[np.isfinite(close)]
        if len(c) < 2:
            rows.append({"symbol": sym, "obs": len(c), "zero_share": 1.0,
                         "flat_run": len(c), "max_abs_daily": 0.0})
            continue
        rets = c[1:] / c[:-1] - 1.0
        longest = _longest_flat_run(c)
        rows.append({
            "symbol": sym,
            "obs": int(len(c)),
            "zero_share": float(np.mean(rets == 0.0)),
            "flat_run": int(longest),
            "max_abs_daily": float(np.max(np.abs(rets))),
        })
    return pd.DataFrame(rows)


def exclusions(stats: pd.DataFrame,
               th: QualityThresholds | None = None) -> dict[str, str]:
    """Return a mapping of excluded symbol to the reason for exclusion."""
    th = th or QualityThresholds()
    out: dict[str, str] = {}
    for r in stats.itertuples(index=False):
        if r.obs < th.min_obs:
            out[r.symbol] = f"stub: only {r.obs} observations"
        elif r.zero_share > th.max_zero_share:
            out[r.symbol] = f"stale: {r.zero_share:.0%} zero returns"
        elif r.flat_run >= th.max_flat_run:
            out[r.symbol] = f"stale: {r.flat_run} identical closes in a row"
        elif r.max_abs_daily > th.max_abs_daily:
            out[r.symbol] = (f"corporate action: {r.max_abs_daily:.1f}x single-day "
                             f"move looks unadjusted")
    return out


def build_exclusions(store, start: date, end: date,
                     th: QualityThresholds | None = None,
                     cache_dir=None, verbose: bool = False) -> dict[str, str]:
    """Compute exclusions for a store, caching the result on disk.

    Parameter sweeps invoke this from every worker process, so results are cached
    under cache_dir. The cache key includes the thresholds.

    Args:
        store: Bar store to screen.
        start: Window start date.
        end: Window end date.
        th: Screening thresholds.
        cache_dir: Directory for the cache file, or None to disable caching.
        verbose: Emit progress output for the initial scan.

    Returns:
        Mapping of excluded symbol to reason.
    """
    import json
    from pathlib import Path

    th = th or QualityThresholds()
    symbols = store.symbols()
    key = f"{start}_{end}_{len(symbols)}_{th.min_obs}_{th.max_zero_share}_" \
          f"{th.max_flat_run}_{th.max_abs_daily}"
    cache_file = None
    if cache_dir is not None:
        cache_file = Path(cache_dir) / f"quality_{key}.json"
        if cache_file.exists():
            try:
                return json.loads(cache_file.read_text())
            except Exception:
                pass

    if verbose:
        print(f"  scanning {len(symbols)} symbols for data quality "
              f"(one-off, then cached)...", flush=True)
    panel = store.panel(symbols, start, end)
    out = exclusions(symbol_stats(store, symbols, start, end, panel=panel), th)

    if cache_file is not None:
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(out))
        except Exception:
            pass
    return out
