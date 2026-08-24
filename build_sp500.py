"""Build an S&P 500 price and membership store from live market data.

Example:
    python3 build_sp500.py --store store_sp500 --provider alpaca \
        --start 2016-01-01 --end 2024-12-31

Resolves point-in-time index membership, then fetches daily bars for every symbol
that was a constituent at any point in the window, including securities since
removed from the index. Omitting those securities would introduce survivorship bias.

Ingestion is resumable: raw CSVs are cached and skipped on subsequent runs unless
--refresh is given. Per-symbol failures are collected and reported rather than
aborting the build.
"""
from __future__ import annotations

import argparse
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from backtester.pipeline.clean import clean_bars
from backtester.pipeline.sources import (
    PROVIDERS, diagnose_stooq, fetch_many, read_raw_csv, supports_batch,
    write_raw_csv,
)
from backtester.pipeline.store import open_store
from backtester.pipeline.universe import (
    describe_tables, fetch_sp500_snapshots, fetch_wikipedia_html,
    fetch_wikipedia_sp500, membership_from_snapshots, reconstruct_membership,
)


def _iso(s: str) -> date:
    return date.fromisoformat(s)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--store", default="store_sp500")
    p.add_argument("--start", type=_iso, default=_iso("2010-01-01"))
    p.add_argument("--end", type=_iso, default=_iso("2024-12-31"))
    p.add_argument("--limit", type=int, default=0,
                   help="cap the number of symbols (0 = all). Useful for a first run.")
    p.add_argument("--backend", choices=["auto", "duckdb", "sqlite"], default="auto")
    p.add_argument("--refresh", action="store_true", help="re-download cached CSVs")
    p.add_argument("--extend-history", action="store_true",
                   help="re-fetch symbols whose cached history starts AFTER --start. "
                        "Use when backfilling from a provider with deeper history "
                        "(e.g. Alpaca's free feed only reaches ~2020).")
    p.add_argument("--history-grace-days", type=int, default=365,
                   help="slack for --extend-history: a name that listed later than "
                        "--start is not re-fetched forever.")
    p.add_argument("--sleep", type=float, default=0.35, help="pause between requests")
    p.add_argument("--max-per-hour", type=int, default=0,
                   help="OPTIONAL throttle for long unattended runs (0 = off). Off by "
                        "default: it is faster to run flat out and resume later, since "
                        "raw CSVs are cached.")
    p.add_argument("--workers", type=int, default=8,
                   help="parallel download threads (ignored for batch providers). "
                        "Ingestion is I/O bound, so threads overlap the waiting.")
    p.add_argument("--membership", choices=["snapshots", "wikipedia", "csv"],
                   default="snapshots",
                   help="snapshots (default): dated full-membership lists back to 1996 "
                        "from the fja05680/sp500 repo -- the reliable source. "
                        "wikipedia: 'selected changes' only, known to be incomplete.")
    p.add_argument("--provider", choices=list(PROVIDERS), default="stooq",
                   help="price source. stooq is keyless but often blocks automated "
                        "downloads; tiingo/alpaca need a free API key in the environment.")
    p.add_argument("--members-csv", default="",
                   help="fallback: CSV with columns symbol,start_date,end_date "
                        "(end_date blank for current members). Skips Wikipedia.")
    p.add_argument("--refresh-wiki", action="store_true",
                   help="re-download the Wikipedia page instead of using the cache")
    p.add_argument("--dump-tables", action="store_true",
                   help="diagnostic: print every table on the Wikipedia page and exit")
    p.add_argument("--probe-stooq", default="",
                   help="diagnostic: probe Stooq for this ticker and exit (e.g. AAPL)")
    p.add_argument("--allow-survivorship", action="store_true",
                   help="proceed even if no change-log was found. THIS REINTRODUCES "
                        "SURVIVORSHIP BIAS -- results will be optimistic and not "
                        "defensible. Off by default for a reason.")
    a = p.parse_args()

    root = Path(a.store)
    raw_dir = root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    if a.probe_stooq:
        from backtester.pipeline.sources import diagnose_stooq
        print(f"Probing Stooq for {a.probe_stooq}...")
        for url, result in diagnose_stooq(a.probe_stooq):
            print(f"  {result:<55} {url[:70]}")
        return

    if a.dump_tables:
        html = fetch_wikipedia_html(cache_path=raw_dir / "wikipedia_sp500.html")
        print(describe_tables(html))
        print(f"\nCached HTML: {raw_dir / 'wikipedia_sp500.html'}")
        return

    # Resolve point-in-time membership.
    if a.membership == "snapshots" and not a.members_csv:
        print("Fetching point-in-time S&P 500 membership snapshots (1996-)...")
        snaps = fetch_sp500_snapshots(cache_path=raw_dir / "sp500_snapshots.csv")
        members = membership_from_snapshots(snaps, floor=a.start, ceiling=a.end)
        current = set(members.loc[members["end_date"].isna(), "symbol"])
        changes = pd.DataFrame()
        print(f"  snapshots: {len(snaps)} dated lists")
        print(f"  membership intervals in window: {len(members)}")
        if members.empty:
            raise SystemExit("No membership intervals overlap the requested window.")
    elif a.members_csv:
        print(f"Reading membership from {a.members_csv} (skipping Wikipedia)...")
        members = pd.read_csv(a.members_csv)
        for c in ("start_date", "end_date"):
            members[c] = pd.to_datetime(members[c], errors="coerce").dt.date
        members["end_date"] = members["end_date"].where(pd.notna(members["end_date"]), None)
        current, changes = set(members.loc[members["end_date"].isna(), "symbol"]), pd.DataFrame()
    else:
        print("WARNING: Wikipedia publishes only 'selected changes', which is known to "
              "be incomplete. Prefer --membership snapshots.")
        # The page HTML is cached under the store, so re-runs do not re-download.
        cache = root / "raw" / "wikipedia_sp500.html"
        if a.refresh_wiki and cache.exists():
            cache.unlink()
        print("Fetching S&P 500 constituents and change-log from Wikipedia...")
        try:
            current, changes = fetch_wikipedia_sp500(cache_path=cache)
        except Exception as e:
            raise SystemExit(
                f"Wikipedia fetch/parse failed: {e}\n"
                f"Workarounds: retry (transient 403s happen), or supply your own "
                f"membership with --members-csv path.csv "
                f"(columns: symbol,start_date,end_date).")
        members = reconstruct_membership(current, changes, floor=a.start)
        if len(changes) == 0:
            msg = (
                "\nNO CHANGE-LOG ROWS were parsed from Wikipedia.\n"
                "Membership would then be just TODAY'S constituents held for the whole\n"
                "window -- which reintroduces SURVIVORSHIP BIAS, the exact error this\n"
                "project is built to avoid. Every backtest on such a store would be\n"
                "optimistic and indefensible.\n\n"
                "Fix one of:\n"
                "  * python3 build_sp500.py --dump-tables   (see the page's real tables)\n"
                "  * --members-csv path.csv                 (supply membership yourself)\n"
                "  * --allow-survivorship                   (ONLY for a plumbing smoke\n"
                "                                            test; results are not valid)")
            if not a.allow_survivorship:
                raise SystemExit(msg)
            print(msg)
            print("\n--allow-survivorship set: continuing with BIASED membership.\n")
    # Keep only intervals that overlap the backtest window.
    members = members[(members["start_date"] <= a.end) &
                      ((members["end_date"].isna()) | (members["end_date"] > a.start))]
    symbols = sorted(members["symbol"].unique())
    if a.membership != "snapshots":
        print(f"  current members: {len(current)}")
        print(f"  change-log rows: {len(changes)}")
    print(f"  symbols touching the window: {len(symbols)} "
          f"(includes names that later left -- that is the point)")
    if a.limit:
        print(f"  NOTE: --limit takes the first {a.limit} alphabetically, which is a "
              f"biased sample. Fine for a smoke test, not for real results.")
    if a.limit:
        symbols = symbols[:a.limit]
        members = members[members["symbol"].isin(symbols)]
        print(f"  limited to {len(symbols)} symbols")

    # Ingest, resuming from cached files where possible.
    def _cache_covers(tk: str) -> bool:
        """Return True if the cached file covers the requested start date.

        Without --extend-history any existing file counts as a hit. With it, a file
        is only reused when its first bar precedes the requested start, allowing a
        grace period for securities that listed later.
        """
        p = raw_dir / f"{tk.upper()}.csv"
        if not p.exists():
            return False
        if not a.extend_history:
            return True
        try:
            head = pd.read_csv(p, nrows=1)
            first = pd.to_datetime(head.iloc[0, 0]).date()
        except Exception:
            return False
        # Grace period accommodates securities that listed after the start date.
        return first <= a.start + timedelta(days=a.history_grace_days)

    todo = [t for t in symbols if a.refresh or not _cache_covers(t)]
    cached = [t for t in symbols if t not in set(todo)]
    fetched, failed = [], []
    state = {"n": 0, "stop": False, "consecutive_429": 0}
    t0 = time.time()

    if supports_batch(a.provider):
        print(f"  {len(todo)} to fetch via {a.provider} in batches of 100 "
              f"({len(cached)} already cached)")
    else:
        print(f"  {len(todo)} to fetch via {a.provider} with {a.workers} threads "
              f"({len(cached)} already cached)")
    if a.max_per_hour:
        print(f"  throttled to {a.max_per_hour} requests/hour")

    def _ok(tk, csv_text):
        write_raw_csv(tk, csv_text, raw_dir)
        fetched.append(tk)
        state["consecutive_429"] = 0
        state["n"] += 1
        if state["n"] % 25 == 0:
            rate = state["n"] / max(time.time() - t0, 1e-9)
            left = (len(todo) - state["n"]) / rate if rate > 0 else 0
            print(f"  ...{state['n']}/{len(todo)} fetched "
                  f"({rate*60:.0f}/min, ~{left/60:.0f} min left)")
        if a.max_per_hour:
            time.sleep(3600.0 / a.max_per_hour)

    def _err(tk, msg):
        failed.append((tk, msg))
        state["n"] += 1
        if "429" in msg or "rate limit" in msg.lower() or "quota" in msg.lower():
            state["consecutive_429"] += 1
            if state["consecutive_429"] == 3 and not state["stop"]:
                state["stop"] = True
                print(f"\n  Rate limited at {tk}. Remaining symbols will fail fast.")
                print("  Re-run later: everything already on disk is cached and skipped.")

    if todo:
        fetch_many(todo, provider=a.provider, start=a.start, end=a.end,
                   workers=(1 if a.max_per_hour else a.workers), on_result=_ok,
                   on_error=_err)
    elapsed = time.time() - t0
    print(f"  ingest took {elapsed/60:.1f} min")

    print(f"Ingest via {a.provider}: {len(fetched)} fetched, {len(cached)} cached, "
          f"{len(failed)} failed")
    # Separate genuine coverage gaps from transport errors: they call for different
    # actions. "no rows" means the feed does not carry that symbol (try another
    # provider); an HTTP error means the request failed (retry).
    no_rows = [(t, m) for t, m in failed if "no rows" in m]
    http_err = [(t, m) for t, m in failed if "no rows" not in m]
    if no_rows:
        print(f"  {len(no_rows)} not covered by this feed (e.g. "
              f"{', '.join(t for t, _ in no_rows[:8])})")
    if http_err:
        print(f"  {len(http_err)} request errors (retry these): "
              f"{', '.join(t for t, _ in http_err[:8])}")
        for tk, err in http_err[:5]:
            print(f"    {tk}: {err}")

    # A failure on a name that has LEFT the index is not a neutral gap: those are
    # exactly the names whose absence reintroduces survivorship bias. Renamed or
    # delisted tickers are the usual cause, and providers often drop the old symbol.
    if failed:
        still_current = set(members.loc[members["end_date"].isna(), "symbol"])
        lost_former = [tk for tk, _ in failed if tk not in still_current]
        if lost_former:
            print(f"\n  Survivorship warning: {len(lost_former)} of the {len(failed)} "
                  f"failures are FORMER members (no longer in the index):")
            print(f"    {', '.join(lost_former[:15])}")
            print("    Usually renamed or delisted tickers the provider no longer "
                  "serves.\n    Their absence biases results upward; see the coverage "
                  "figure below.")

    # Validate and persist bars.
    store = open_store(root, backend=a.backend)
    print(f"Store backend: {type(store).__name__}")
    rows_in = rows_out = 0
    stored = []
    for tk in symbols:
        path = raw_dir / f"{tk.upper()}.csv"
        if not path.exists():
            continue
        try:
            tidy, rep = clean_bars(read_raw_csv(path), tk)
        except Exception as e:
            print(f"  clean failed {tk}: {str(e)[:60]}")
            continue
        if rep.rows_out == 0:
            continue
        # Keep only the requested window.
        tidy = tidy[(tidy["date"] >= a.start) & (tidy["date"] <= a.end)]
        if tidy.empty:
            continue
        store.write_bars(tidy)
        stored.append(tk)
        rows_in += rep.rows_in
        rows_out += rep.rows_out
    print(f"Cleaned and stored {rows_out}/{rows_in} rows across {len(stored)} symbols")

    # Persist membership for the symbols actually stored.
    # Abort before writing membership if nothing was stored, so that a failed run
    # cannot overwrite an existing store.
    if not stored:
        raise SystemExit(
            "\nNO SYMBOLS STORED -- aborting BEFORE writing membership so an existing\n"
            "store is left intact.\n"
            f"  * HTTP 429 means the provider rate-limited you. Tiingo's free tier allows\n"
            f"    50 requests/hour and 500 unique symbols/month. Wait an hour and re-run;\n"
            f"    raw CSVs are cached under {raw_dir}, so completed symbols are not\n"
            f"    re-fetched and your quota is not spent twice.\n"
            "  * Do NOT delete the store between runs: the raw cache is what protects\n"
            "    your quota.\n"
            "  * --max-per-hour throttles requests to stay under the limit.")

    # Coverage measured before filtering, to quantify what was lost.
    wanted = set(members["symbol"])
    former_wanted = {s_ for s_ in wanted
                     if s_ not in set(members.loc[members["end_date"].isna(), "symbol"])}
    former_have = former_wanted & set(stored)

    members = members[members["symbol"].isin(stored)]
    store.write_membership(members)
    print(f"Membership: {len(members)} intervals for {members['symbol'].nunique()} symbols")
    if former_wanted:
        pct = 100.0 * len(former_have) / len(former_wanted)
        print(f"Former-member coverage: {len(former_have)}/{len(former_wanted)} "
              f"({pct:.0f}%)")

    # Report universe size at several points in the window.
    for probe in (a.start, date((a.start.year + a.end.year) // 2, 6, 30), a.end):
        try:
            n = len(store.universe_as_of(probe))
            print(f"  universe as of {probe}: {n} members")
        except Exception:
            pass
    print(f"\nDone. Run a backtest with:  data_source='store', store_path='{a.store}'")


if __name__ == "__main__":
    main()
