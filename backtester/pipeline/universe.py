"""Point-in-time index membership.

Membership is represented as (symbol, start_date, end_date) intervals, with a null
end date for current constituents. A security that leaves and later rejoins the index
produces multiple disjoint intervals.

Two reconstruction paths are supported. membership_from_snapshots derives intervals
from dated constituent lists and is preferred, since membership on any date is read
directly. reconstruct_membership replays an add and drop change log backwards from
the current constituents, which depends on the log being complete.
"""
from __future__ import annotations

from datetime import date

import pandas as pd


def reconstruct_membership(
    current: set[str], changes: pd.DataFrame, floor: date
) -> pd.DataFrame:
    """Derive membership intervals from a change log.

    Args:
        current: Symbols currently in the index.
        changes: Frame with date, added, and removed columns. Either of added or
            removed may be empty for a given row.
        floor: Lower bound for intervals extending before the recorded changes.

    Returns:
        Frame of (symbol, start_date, end_date) sorted by symbol and start date.
    """
    adds: dict[str, list[date]] = {}
    drops: dict[str, list[date]] = {}
    for row in changes.itertuples(index=False):
        a = (getattr(row, "added", "") or "").strip()
        r = (getattr(row, "removed", "") or "").strip()
        if a:
            adds.setdefault(a, []).append(row.date)
        if r:
            drops.setdefault(r, []).append(row.date)

    out: list[dict] = []
    for s in set(current) | set(adds) | set(drops):
        a = sorted(adds.get(s, []))
        r = sorted(drops.get(s, []))
        starts = list(a)
        # A constituent whose first recorded event is a removal, or which has no
        # recorded events, was a member before the log begins.
        if not a and (s in current or r):
            starts = [floor] + starts
        elif s in current and r and a and a[0] > r[0]:
            starts = [floor] + starts
        starts = sorted(set(starts))

        drops_q = list(r)
        for st in starts:
            end = next((d for d in drops_q if d > st), None)
            if end is not None:
                drops_q = [d for d in drops_q if d != end]
            out.append({"symbol": s, "start_date": st, "end_date": end})

    return pd.DataFrame(out).sort_values(["symbol", "start_date"]).reset_index(drop=True)


def fixture_membership(floor: date, swap: date) -> pd.DataFrame:
    """Return fixture membership containing a single constituent swap."""
    from .sources import FIXTURE_CORE, FIXTURE_JOINER, FIXTURE_LEAVER
    rows = [{"symbol": s, "start_date": floor, "end_date": None} for s in FIXTURE_CORE]
    rows.append({"symbol": FIXTURE_LEAVER, "start_date": floor, "end_date": swap})
    rows.append({"symbol": FIXTURE_JOINER, "start_date": swap, "end_date": None})
    return pd.DataFrame(rows)


_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
# Wikipedia returns HTTP 403 to clients that do not identify themselves.
# pandas.read_html sends no User-Agent, so the page is fetched separately.
_WIKI_UA = ("LeakProofBacktester/1.0 (educational research project; "
            "contact: local user) python-urllib")


def fetch_wikipedia_html(url: str = _WIKI_URL, timeout: int = 20,
                         cache_path=None) -> str:
    """Download the constituents page, reusing a cached copy when available."""
    import urllib.request
    from pathlib import Path

    if cache_path is not None:
        p = Path(cache_path)
        if p.exists():
            return p.read_text(encoding="utf-8")

    req = urllib.request.Request(url, headers={
        "User-Agent": _WIKI_UA,
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "en-US,en;q=0.9",
    })
    html = urllib.request.urlopen(req, timeout=timeout).read().decode("utf-8", "replace")

    if cache_path is not None:
        p = Path(cache_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(html, encoding="utf-8")
    return html


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten a MultiIndex column header into single strings."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [" ".join(str(x) for x in tup if str(x) != "nan").strip()
                      for tup in df.columns]
    else:
        df = df.copy()
        df.columns = [str(c).strip() for c in df.columns]
    return df


def _find_col(cols, *musts) -> str | None:
    """Return the first column whose name contains all given substrings."""
    for c in cols:
        lc = str(c).lower()
        if all(m in lc for m in musts):
            return c
    return None


def _clean_ticker(x) -> str:
    """Normalise a ticker cell to canonical form, removing footnote markers."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    t = str(x).strip()
    if t.lower() in ("nan", "none", "—", "-", "–", ""):
        return ""
    t = t.split("[")[0].strip()           # drop footnote markers
    return t.replace(".", "-").upper()


def describe_tables(html: str) -> str:
    """Return a summary of every table on the page, for diagnosing parse failures."""
    from io import StringIO
    lines = []
    for i, raw in enumerate(pd.read_html(StringIO(html))):
        t = _flatten_columns(raw)
        lines.append(f"[{i}] rows={len(t)} cols={list(t.columns)}")
    return "\n".join(lines)


def _fallback_changes(tables) -> pd.DataFrame:
    """Locate a change log by table structure when header matching fails.

    Selects the largest table containing a parseable date column and at least two
    ticker-like columns, treating the first as additions and the second as removals.
    """
    import re
    ticker_re = re.compile(r"^[A-Z][A-Z0-9.\-]{0,6}$")

    best = pd.DataFrame(columns=["date", "added", "removed"])
    for raw in tables:
        t = _flatten_columns(raw)
        if len(t) < 5 or t.shape[1] < 3:
            continue
        date_col = None
        for c in t.columns:
            parsed = pd.to_datetime(t[c], errors="coerce", format="mixed")
            if parsed.notna().mean() > 0.7:
                date_col = c
                break
        if date_col is None:
            continue
        tickerish = []
        for c in t.columns:
            if c == date_col:
                continue
            vals = [str(v).split("[")[0].strip() for v in t[c].fillna("")]
            hits = sum(bool(ticker_re.match(v)) for v in vals if v and v.lower() != "nan")
            if hits >= max(2, int(0.3 * len(vals))):
                tickerish.append(c)
        if len(tickerish) >= 2:
            parsed = pd.to_datetime(t[date_col], errors="coerce", format="mixed")
            cand = pd.DataFrame({
                "date": parsed.dt.date,
                "added": [_clean_ticker(v) for v in t[tickerish[0]]],
                "removed": [_clean_ticker(v) for v in t[tickerish[1]]],
            }).dropna(subset=["date"])
            if len(cand) > len(best):
                best = cand
    return best.reset_index(drop=True)


def parse_sp500_tables(html: str) -> tuple[set[str], pd.DataFrame]:
    """Extract current constituents and the change log from page HTML.

    Tables are identified by column content rather than position. The change log is
    distinguished by the presence of a removal column; the constituents table
    contains a "Date added" column and must not be matched on that alone.

    Raises:
        RuntimeError: If the constituents table cannot be located.
    """
    from io import StringIO
    tables = pd.read_html(StringIO(html))

    current: set[str] = set()
    changes = pd.DataFrame(columns=["date", "added", "removed"])

    for raw in tables:
        t = _flatten_columns(raw)
        cols = list(t.columns)

        sym_col = _find_col(cols, "symbol") or _find_col(cols, "ticker")
        name_col = _find_col(cols, "security") or _find_col(cols, "company")
        date_col = _find_col(cols, "date")
        add_col = (_find_col(cols, "added", "ticker") or _find_col(cols, "added", "symbol"))
        rem_col = (_find_col(cols, "removed", "ticker") or _find_col(cols, "removed", "symbol"))
        has_removed = rem_col is not None or _find_col(cols, "removed") is not None

        # Change-log: a date plus BOTH an added and a removed column. Prefer the
        # explicit "... Ticker" subcolumns, but fall back to any added/removed pair,
        # because the page's header wording has changed before and will again.
        if add_col is None:
            add_col = _find_col(cols, "added")
        if rem_col is None:
            rem_col = _find_col(cols, "removed")
        if changes.empty and date_col and add_col and rem_col:
            parsed = pd.to_datetime(t[date_col], errors="coerce", format="mixed")
            changes = pd.DataFrame({
                "date": parsed.dt.date,
                "added": [_clean_ticker(v) for v in t[add_col]],
                "removed": [_clean_ticker(v) for v in t[rem_col]],
            }).dropna(subset=["date"])
            continue

        # Constituents: a symbol column, a company/security column, and NO removed
        # column. "Date added" here is a listing date, not a change-log entry.
        if not current and sym_col and not has_removed and (name_col or len(t) > 50):
            current = {x for x in (_clean_ticker(v) for v in t[sym_col]) if x}
            continue

    if changes.empty:
        changes = _fallback_changes(tables)

    if not current:
        raise RuntimeError(
            "Could not find the S&P 500 constituents table on the page. Run "
            "`python3 build_sp500.py --dump-tables` to see the tables actually present, "
            "or pass --members-csv to bypass Wikipedia.")
    return current, changes.reset_index(drop=True)


def fetch_wikipedia_sp500(cache_path=None) -> tuple[set[str], pd.DataFrame]:
    """Live source: current constituents + the add/drop change-log from Wikipedia.

    Returns (current_symbols, changes_df) with columns date, added, removed --
    ready for reconstruct_membership.
    """
    return parse_sp500_tables(fetch_wikipedia_html(cache_path=cache_path))


# Snapshot-based membership. Wikipedia publishes selected changes only, which is
# documented as incomplete. The fja05680/sp500 dataset (MIT licensed) publishes dated
# snapshots of the full constituent list from 1996 onward, so membership on any date
# is read directly rather than inferred from an event sequence.
SP500_SNAPSHOT_URL = (
    "https://raw.githubusercontent.com/fja05680/sp500/master/"
    "S%26P%20500%20Historical%20Components%20%26%20Changes(08-17-2024).csv"
)
SP500_SNAPSHOT_URL_FALLBACK = (
    "https://raw.githubusercontent.com/fja05680/sp500/master/"
    "S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv"
)


def fetch_sp500_snapshots(cache_path=None, url: str | None = None,
                          timeout: int = 30) -> pd.DataFrame:
    """Download dated membership snapshots, caching to disk when a path is given.

    Returns:
        Frame with date and tickers columns.

    Raises:
        RuntimeError: If no candidate URL returns usable data.
    """
    import urllib.request
    from pathlib import Path

    if cache_path is not None:
        p = Path(cache_path)
        if p.exists():
            return pd.read_csv(p)

    urls = [url] if url else [SP500_SNAPSHOT_URL, SP500_SNAPSHOT_URL_FALLBACK]
    last = None
    for u in urls:
        try:
            req = urllib.request.Request(u, headers={"User-Agent": _WIKI_UA})
            text = urllib.request.urlopen(req, timeout=timeout).read().decode(
                "utf-8", "replace")
            from io import StringIO
            df = pd.read_csv(StringIO(text))
            if cache_path is not None:
                p = Path(cache_path)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(text, encoding="utf-8")
            return df
        except Exception as e:
            last = e
    raise RuntimeError(f"could not download S&P 500 snapshots: {last}")


def membership_from_snapshots(snapshots: pd.DataFrame, floor: date,
                              ceiling: date | None = None) -> pd.DataFrame:
    """Convert dated membership snapshots into membership intervals.

    An interval opens at the first snapshot containing a symbol and closes at the
    next snapshot from which it is absent, so securities that leave and rejoin the
    index produce multiple disjoint intervals.

    Args:
        snapshots: Frame with a date column and a comma-separated tickers column.
        floor: Lower bound applied to interval start dates.
        ceiling: Optional upper bound; intervals starting later are dropped.
    """
    date_col = next((c for c in snapshots.columns if "date" in str(c).lower()),
                    snapshots.columns[0])
    tick_col = next((c for c in snapshots.columns if "ticker" in str(c).lower()),
                    snapshots.columns[-1])

    snaps = snapshots[[date_col, tick_col]].copy()
    snaps[date_col] = pd.to_datetime(snaps[date_col], errors="coerce", format="mixed")
    snaps = snaps.dropna(subset=[date_col]).sort_values(date_col)

    open_since: dict[str, date] = {}
    rows: list[dict] = []
    prev: set[str] = set()
    last_d: date | None = None

    for r in snaps.itertuples(index=False):
        d = r[0].date()
        members = {t for t in (_clean_ticker(x) for x in str(r[1]).split(",")) if t}
        if not members:
            continue
        for sym in members - prev:
            open_since[sym] = d
        for sym in prev - members:
            start = open_since.pop(sym, None)
            if start is not None:
                rows.append({"symbol": sym, "start_date": start, "end_date": d})
        prev, last_d = members, d

    for sym, start in open_since.items():
        rows.append({"symbol": sym, "start_date": start, "end_date": None})

    m = pd.DataFrame(rows)
    if m.empty:
        return m
    # Restrict to intervals overlapping the requested window.
    m = m[(m["end_date"].isna()) | (pd.Series([e for e in m["end_date"]]) > floor)]
    if ceiling is not None:
        m = m[m["start_date"] <= ceiling]
    m["start_date"] = [max(s, floor) for s in m["start_date"]]
    return m.sort_values(["symbol", "start_date"]).reset_index(drop=True)
