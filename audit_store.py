"""Report data-quality defects in a price store.

Example:
    python3 audit_store.py --store store_sp500 --start 2016-01-01 --end 2024-12-31
    python3 audit_store.py --store store_sp500 --coverage

Three defects distort factor results without producing errors: stale price series,
which measure near-zero volatility and rank highly in low-volatility screens;
unadjusted corporate actions, which appear as extreme single-day returns; and short
histories, which yield unstable statistics.

The --coverage report additionally shows universe size by year and where each
security's history begins, revealing provider history limits.
"""
from __future__ import annotations

import argparse
from datetime import date

import numpy as np
import pandas as pd

from backtester.pipeline.store import open_store


def _iso(s: str) -> date:
    return date.fromisoformat(s)


def audit(store_path: str, start: date, end: date, top: int = 15) -> pd.DataFrame:
    store = open_store(store_path)
    symbols = store.symbols()
    dates, data = store.panel(symbols, start, end)
    if not dates:
        raise SystemExit("store has no bars in that window")

    rows = []
    for sym in symbols:
        cell = data.get(sym)
        if cell is None:
            continue
        close = cell["close"]
        ok = np.isfinite(close)
        n = int(ok.sum())
        if n < 2:
            continue
        c = close[ok]
        rets = c[1:] / c[:-1] - 1.0

        # Longest run of an unchanged close: the signature of a stale series.
        same = np.concatenate([[False], c[1:] == c[:-1]])
        longest, cur = 0, 0
        for flag in same:
            cur = cur + 1 if flag else 0
            longest = max(longest, cur)
        zero_share = float((rets == 0).mean())

        big = float(np.max(np.abs(rets))) if len(rets) else 0.0
        split_like = int(np.sum((rets < -0.45) | (rets > 0.9)))

        rows.append({
            "symbol": sym,
            "obs": n,
            "ann_vol": float(np.std(rets, ddof=1) * np.sqrt(252)) if len(rets) > 1 else np.nan,
            "longest_flat_run": longest,
            "zero_return_share": zero_share,
            "max_abs_daily": big,
            "split_like_moves": split_like,
        })
    return pd.DataFrame(rows)


def coverage_report(store_path: str, start: date, end: date) -> None:
    """Report symbol counts by year and the first observation date per symbol.

    A provider that serves only recent history leaves early years sparsely
    populated, so cross-sectional rankings over that period are computed on an
    unrepresentative universe.
    """
    store = open_store(store_path)
    symbols = store.symbols()
    dates, data = store.panel(symbols, start, end)
    if not dates:
        print("no bars in window")
        return
    years = sorted({d.year for d in dates})

    print("\n" + "=" * 70)
    print("COVERAGE BY YEAR (symbols with at least one bar)")
    print("=" * 70)
    print(f"{'year':>6} {'symbols':>9} {'trading days':>13}")
    for y in years:
        mask = np.array([d.year == y for d in dates])
        n = 0
        for sym in symbols:
            cell = data.get(sym)
            if cell is not None and np.isfinite(cell["close"][mask]).any():
                n += 1
        print(f"{y:>6} {n:>9} {int(mask.sum()):>13}")

    # Bars extending beyond a membership interval are normal: index removal is not
    # delisting, and the engine gates trading on membership, so these bars are never
    # tradeable. Reported for context, since a security whose issuer ceased to exist
    # but which still shows quotes may indicate a spliced or reused series.
    mem = store._fetch_membership()
    if len(mem):
        lasts = {}
        for sym in symbols:
            cell = data.get(sym)
            if cell is None:
                continue
            ok = np.flatnonzero(np.isfinite(cell["close"]))
            if len(ok):
                lasts[sym] = dates[ok[-1]]
        suspects = []
        for r in mem.itertuples(index=False):
            if r.end_date is None or r.symbol not in lasts:
                continue
            last_bar = lasts[r.symbol]
            gap = (last_bar - r.end_date).days
            if gap > 200:          # still trading 200+ days after leaving the index
                suspects.append((r.symbol, r.end_date, last_bar, gap))
        print("\n" + "=" * 70)
        print("BARS BEYOND MEMBERSHIP (informational, normally harmless)")
        print("=" * 70)
        if suspects:
            suspects.sort(key=lambda x: -x[3])
            print(f"{'symbol':>8} {'left index':>12} {'last bar':>12} {'days after':>11}")
            for sym, ed, lb, gap in suspects[:10]:
                print(f"{sym:>8} {str(ed):>12} {str(lb):>12} {gap:>11}")
            print(f"\n{len(suspects)} symbols have bars after leaving the index.")
            print("This is expected: index removal is not delisting, and membership")
            print("gating means these bars are never tradeable. Investigate only where")
            print("the issuer ceased to exist, which would indicate a spliced series.")
        else:
            print("none")

    firsts = {}
    for sym in symbols:
        cell = data.get(sym)
        if cell is None:
            continue
        ok = np.flatnonzero(np.isfinite(cell["close"]))
        if len(ok):
            firsts[sym] = dates[ok[0]]
    if firsts:
        counts = pd.Series([d.year for d in firsts.values()]).value_counts().sort_index()
        print("\nFirst observation by year (a concentration indicates a history limit):")
        for y, n in counts.items():
            print(f"  {y}: {n} symbols start here")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--store", default="store_sp500")
    p.add_argument("--start", type=_iso, default=_iso("2016-01-01"))
    p.add_argument("--end", type=_iso, default=_iso("2024-12-31"))
    p.add_argument("--min-obs", type=int, default=252)
    p.add_argument("--coverage", action="store_true",
                   help="report symbol coverage by year and where each history starts")
    a = p.parse_args()

    if a.coverage:
        coverage_report(a.store, a.start, a.end)
        return

    df = audit(a.store, a.start, a.end)
    n = len(df)
    print("scanning store (this takes ~15s on a 700-symbol store)...", flush=True)
    print("=" * 70)
    print(f"DATA QUALITY AUDIT: {n} symbols, {a.start} to {a.end}")
    print("=" * 70)

    stale = df[(df["longest_flat_run"] >= 5) | (df["zero_return_share"] > 0.20)]
    short = df[df["obs"] < a.min_obs]
    splits = df[df["split_like_moves"] > 0]

    print(f"\nStale-price suspects (>=5 identical closes in a row, or >20% zero "
          f"returns): {len(stale)}")
    if len(stale):
        s = stale.sort_values("zero_return_share", ascending=False).head(12)
        print(s[["symbol", "obs", "ann_vol", "longest_flat_run",
                 "zero_return_share"]].to_string(index=False))

    print(f"\nPossible unadjusted actions (daily move < -45% or > +90%): {len(splits)}")
    if len(splits):
        print(splits.sort_values("max_abs_daily", ascending=False).head(12)[
            ["symbol", "obs", "max_abs_daily", "split_like_moves"]].to_string(index=False))

    print(f"\nShort history (< {a.min_obs} observations): {len(short)}")
    if len(short):
        print(", ".join(short.sort_values("obs")["symbol"].head(20)))

    # The decisive question: would the low-volatility factor pick the stale names?
    print("\n" + "=" * 70)
    print("LOWEST-VOLATILITY CANDIDATES")
    print("=" * 70)
    ranked = df.dropna(subset=["ann_vol"]).sort_values("ann_vol")
    bottom = ranked.head(20)
    n_stale = int(((bottom["longest_flat_run"] >= 5) |
                   (bottom["zero_return_share"] > 0.20)).sum())
    print(f"Of the 20 lowest-volatility symbols, {n_stale} show stale-price flags.")
    print("Note: this ranks by full-window volatility, a proxy. Factors score a")
    print("security only once its warmup window is met, so short-history entries may")
    print("never be selected. Use report_selections.py for actual holdings.")
    print(bottom[["symbol", "obs", "ann_vol", "longest_flat_run",
                  "zero_return_share"]].head(12).to_string(index=False))
    if n_stale >= 3:
        print("\nLow-ranked volatility here may reflect stale prices rather than low")
        print("risk. The liquidity floor in factors.py and cfg.quality_filter address")
        print("this. Confirm with report_selections.py.")
    else:
        print("\nNo stale-price concentration among the low-volatility candidates.")


if __name__ == "__main__":
    main()
