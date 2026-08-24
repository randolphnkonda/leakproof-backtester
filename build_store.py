"""Build a price and membership store from Stooq or from fixture data.

Examples:
    python3 build_store.py --source fixture --store store
    python3 build_store.py --source stooq --store store \
        --symbols AAPL MSFT AMZN --start 2015-01-01 --end 2023-12-31

Both sources share the same cleaning, storage, and membership path, so a fixture
build exercises the same code as a live one. See build_sp500.py to build the full
index universe.
"""
from __future__ import annotations

import argparse
import time
from datetime import date
from pathlib import Path

import pandas as pd

from backtester.pipeline.clean import clean_bars
from backtester.pipeline.sources import (
    FIXTURE_SWAP_DATE, fetch_stooq, generate_fixture, read_raw_csv, write_raw_csv,
)
from backtester.pipeline.store import open_store
from backtester.pipeline.universe import (
    fetch_wikipedia_sp500, fixture_membership, reconstruct_membership,
)


def _iso(s: str) -> date:
    return date.fromisoformat(s)


def build(source: str, store_root: str, symbols, start: date, end: date,
          backend: str, seed: int = 0) -> None:
    root = Path(store_root)
    raw_dir = root / "raw"
    store = open_store(root, backend=backend)
    print(f"Store backend: {type(store).__name__}  ->  {root}")

    # Ingest raw CSVs.
    if source == "fixture":
        tickers = generate_fixture(raw_dir, start, end, seed=seed)
        print(f"Fixture: wrote {len(tickers)} raw CSVs to {raw_dir}")
    else:
        tickers = []
        for tk in symbols:
            try:
                write_raw_csv(tk, fetch_stooq(tk), raw_dir)
                tickers.append(tk)
                time.sleep(0.4)
            except Exception as e:
                print(f"  skip {tk}: {e}")
        print(f"Stooq: fetched {len(tickers)}/{len(symbols)} tickers")

    # Validate and persist.
    total_in = total_out = 0
    for tk in tickers:
        tidy, rep = clean_bars(read_raw_csv(raw_dir / f"{tk.upper()}.csv"), tk)
        store.write_bars(tidy)
        total_in += rep.rows_in
        total_out += rep.rows_out
        print("  " + rep.line())
    print(f"Cleaned {total_out}/{total_in} rows into the bars store")

    # Membership intervals.
    if source == "fixture":
        members = fixture_membership(floor=start, swap=FIXTURE_SWAP_DATE)
    else:
        try:
            current, changes = fetch_wikipedia_sp500()
            members = reconstruct_membership(current, changes, floor=start)
        except Exception as e:
            print(f"  membership: Wikipedia unavailable ({e}); "
                  f"falling back to full-window membership for fetched tickers")
            members = pd.DataFrame(
                [{"symbol": tk.upper(), "start_date": start, "end_date": None}
                 for tk in tickers])
    store.write_membership(members)
    print(f"Membership: {len(members)} intervals written")

    d0, d1 = start, end
    print(f"Universe as of {d0}: {sorted(store.universe_as_of(d0))}")
    print(f"Universe as of {d1}: {sorted(store.universe_as_of(d1))}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["fixture", "stooq"], default="fixture")
    p.add_argument("--store", default="store")
    p.add_argument("--symbols", nargs="*", default=[])
    p.add_argument("--start", type=_iso, default=_iso("2015-01-01"))
    p.add_argument("--end", type=_iso, default=_iso("2023-12-31"))
    p.add_argument("--backend", choices=["auto", "duckdb", "sqlite"], default="auto")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    build(a.source, a.store, a.symbols, a.start, a.end, a.backend, a.seed)


if __name__ == "__main__":
    main()
