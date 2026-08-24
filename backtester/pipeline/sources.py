"""Market data providers.

All providers emit the same CSV schema, so downstream cleaning and storage are
provider independent:

    Date,Open,High,Low,Close,Volume

with dates ascending in YYYY-MM-DD format.

    fetch_stooq         keyless, but frequently blocks automated access
    fetch_tiingo        requires TIINGO_API_KEY
    fetch_alpaca        requires ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY
    generate_fixture    writes schema-identical files for offline testing

fetch_many handles bulk retrieval, batching or parallelising according to provider
capability.
"""
from __future__ import annotations

import urllib.request
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Stooq requires a session cookie from the quote page, a browser User-Agent, and a
# Referer header. Without them it returns an HTML challenge page instead of CSV.
_STOOQ_QUOTE = "https://stooq.com/q/d/?s={sym}"
_STOOQ_VARIANTS = (
    "https://stooq.com/q/d/l/?s={sym}&i=d",
    "https://stooq.com/q/d/l/?s={sym}&i=d&d1={d1}&d2={d2}",
    "https://stooq.pl/q/d/l/?s={sym}&i=d",
)
_BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


def stooq_symbol(ticker: str) -> str:
    """Convert a ticker to Stooq notation, for example aapl.us."""
    t = ticker.lower().replace(".", "-")
    return t if t.endswith(".us") else f"{t}.us"


def provider_symbol(ticker: str, provider: str) -> str:
    """Convert a canonical ticker into a provider's symbol notation.

    Share classes differ by provider. The canonical form is hyphenated, BRK-B, which
    Tiingo accepts directly; Alpaca expects BRK.B and Stooq expects brk-b.us. An
    incorrect conversion returns HTTP 400 and is indistinguishable from a genuine
    coverage gap.
    """
    import re
    t = ticker.strip().upper()
    if provider == "alpaca":
        # BRK-B -> BRK.B, but leave ordinary tickers untouched.
        return re.sub(r"^([A-Z]+)-([A-Z])$", r"\1.\2", t)
    if provider == "tiingo":
        return t
    if provider == "stooq":
        return stooq_symbol(t)
    return t


def _make_opener():
    """Return a urllib opener that retains cookies across requests."""
    import http.cookiejar
    import urllib.request
    jar = http.cookiejar.CookieJar()
    return urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))


_OPENER = None


def _get(url: str, referer: str | None = None, timeout: int = 20) -> str:
    import urllib.request
    global _OPENER
    if _OPENER is None:
        _OPENER = _make_opener()
    headers = {
        "User-Agent": _BROWSER_UA,
        "Accept": "text/csv,text/plain,text/html;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    if referer:
        headers["Referer"] = referer
    req = urllib.request.Request(url, headers=headers)
    return _OPENER.open(req, timeout=timeout).read().decode("utf-8", "replace")


def _looks_like_csv(text: str) -> bool:
    return text.lstrip().lower().startswith("date,")


def _diagnose_body(text: str) -> str:
    """Classify a non-CSV response body into a diagnostic message."""
    low = text[:2000].lower()
    if "exceeded" in low or "limit" in low:
        return "daily hit limit exceeded (wait, or slow down with --sleep)"
    if "captcha" in low or "recaptcha" in low:
        return "captcha challenge -- Stooq is blocking automated access from this IP"
    if low.lstrip().startswith("<!doctype html") or "<html" in low:
        return "HTML page instead of CSV (blocked, redirected, or unknown symbol)"
    if "no data" in low:
        return "no data for this symbol"
    return f"unexpected body: {text[:60]!r}"


def fetch_stooq(ticker: str, timeout: int = 15,
                start=None, end=None, warm: bool = True) -> str:
    """Download daily history for one ticker from Stooq.

    Establishes a session on the quote page, then tries each download URL variant.

    Returns:
        Raw CSV text.

    Raises:
        RuntimeError: If no variant returns CSV, with a diagnostic message.
    """
    sym = stooq_symbol(ticker)
    if warm:
        try:
            _get(_STOOQ_QUOTE.format(sym=sym), timeout=timeout)
        except Exception:
            pass  # Session setup is best effort; the download may still succeed.

    d1 = start.strftime("%Y%m%d") if start else "19900101"
    d2 = end.strftime("%Y%m%d") if end else "20991231"
    reasons = []
    for tpl in _STOOQ_VARIANTS:
        url = tpl.format(sym=sym, d1=d1, d2=d2)
        try:
            text = _get(url, referer=_STOOQ_QUOTE.format(sym=sym), timeout=timeout)
        except Exception as e:
            reasons.append(f"{type(e).__name__}")
            continue
        if _looks_like_csv(text):
            return text
        reasons.append(_diagnose_body(text))
    raise RuntimeError(f"{ticker}: {reasons[0] if reasons else 'no response'}")


def diagnose_stooq(ticker: str = "AAPL") -> list[tuple[str, str]]:
    """Probe each Stooq endpoint and report the response classification."""
    sym = stooq_symbol(ticker)
    out = []
    try:
        body = _get(_STOOQ_QUOTE.format(sym=sym))
        out.append(("quote page (session warm-up)",
                    "OK" if body else "empty"))
    except Exception as e:
        out.append(("quote page (session warm-up)", f"{type(e).__name__}: {e}"))
    for tpl in _STOOQ_VARIANTS:
        url = tpl.format(sym=sym, d1="20150101", d2="20241231")
        try:
            text = _get(url, referer=_STOOQ_QUOTE.format(sym=sym))
            out.append((url, "CSV OK" if _looks_like_csv(text) else _diagnose_body(text)))
        except Exception as e:
            out.append((url, f"{type(e).__name__}: {e}"))
    return out


def write_raw_csv(ticker: str, csv_text: str, raw_dir: Path) -> Path:
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"{ticker.upper()}.csv"
    path.write_text(csv_text)
    return path


def read_raw_csv(path: Path) -> pd.DataFrame:
    """Read a provider CSV into an unvalidated DataFrame."""
    return pd.read_csv(path)


# --------------------------------------------------------------------------- #
# Offline fixture: schema-identical raw CSVs, so the pipeline runs with no network.
# A deliberate listing event (OLDCO leaves, NEWCO joins mid-history) exercises the
# point-in-time universe and survivorship handling on the real code path.
# --------------------------------------------------------------------------- #
FIXTURE_CORE = ("AAPL", "MSFT", "AMZN", "GOOGL", "JPM", "JNJ",
                "XOM", "PG", "KO", "DIS", "INTC", "CSCO")
FIXTURE_SWAP_DATE = date(2019, 6, 21)
FIXTURE_LEAVER = "OLDCO"     # in the index until the swap date, then delisted
FIXTURE_JOINER = "NEWCO"     # lists and joins the index on the swap date


def _gbm_raw(ticker: str, start: date, end: date, seed: int) -> pd.DataFrame:
    """Generate a synthetic OHLCV frame in the provider CSV schema."""
    dates = [d.date() for d in pd.bdate_range(start, end)]
    n = len(dates)
    rng = np.random.default_rng(seed)
    mu = rng.uniform(0.03, 0.16) / 252.0
    vol = rng.uniform(0.16, 0.40) / np.sqrt(252.0)
    close = 100.0 * np.exp(np.cumsum(rng.normal(mu, vol, n)))
    o = np.concatenate([[close[0]], close[:-1]])
    intr = np.abs(rng.normal(0, vol, n)) * close
    hi = np.maximum(o, close) + intr * 0.5
    lo = np.minimum(o, close) - intr * 0.5
    volu = rng.uniform(1e6, 5e6, n).round()
    return pd.DataFrame({
        "Date": [d.isoformat() for d in dates],
        "Open": o.round(4), "High": hi.round(4), "Low": lo.round(4),
        "Close": close.round(4), "Volume": volu.astype("int64"),
    })


def generate_fixture(raw_dir: Path, start: date, end: date, seed: int = 0) -> list[str]:
    """Write fixture CSV files and return the tickers produced."""
    tickers: list[str] = []
    for i, tk in enumerate(FIXTURE_CORE):
        write_raw_csv(tk, _gbm_raw(tk, start, end, seed + i).to_csv(index=False), raw_dir)
        tickers.append(tk)
    # The departing symbol has data up to the swap date; the joining symbol from it.
    write_raw_csv(FIXTURE_LEAVER,
                  _gbm_raw(FIXTURE_LEAVER, start, FIXTURE_SWAP_DATE, seed + 90).to_csv(index=False),
                  raw_dir)
    write_raw_csv(FIXTURE_JOINER,
                  _gbm_raw(FIXTURE_JOINER, FIXTURE_SWAP_DATE, end, seed + 91).to_csv(index=False),
                  raw_dir)
    tickers += [FIXTURE_LEAVER, FIXTURE_JOINER]
    return tickers


# Authenticated providers. Both serve delisted symbols, which survivorship-free
# backtests require. Credentials are read from the environment.
import json
import os


def _http_json(req, timeout: int, retries: int = 3):
    """Issue a JSON request, retrying on rate limits and server errors.

    Honours the Retry-After header where present, otherwise backs off exponentially.
    """
    import time as _time
    import urllib.error
    import urllib.request
    last = None
    for attempt in range(retries):
        try:
            return json.loads(urllib.request.urlopen(req, timeout=timeout).read().decode())
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (429, 500, 502, 503, 504):
                wait = e.headers.get("Retry-After") if e.headers else None
                delay = float(wait) if (wait and str(wait).isdigit()) else 2.0 * (2 ** attempt)
                if attempt < retries - 1:
                    _time.sleep(min(delay, 60.0))
                    continue
            raise
    raise last


def _to_csv_text(rows: list[dict]) -> str:
    """Serialise bar records into the common provider CSV schema."""
    out = ["Date,Open,High,Low,Close,Volume"]
    for r in rows:
        out.append(f"{r['date']},{r['open']},{r['high']},{r['low']},"
                   f"{r['close']},{r['volume']}")
    return "\n".join(out) + "\n"


def fetch_tiingo(ticker: str, start=None, end=None, timeout: int = 20) -> str:
    key = os.environ.get("TIINGO_API_KEY", "").strip()
    if not key:
        raise RuntimeError("TIINGO_API_KEY is not set in the environment")
    import urllib.request
    sd = (start.isoformat() if start else "1990-01-01")
    ed = (end.isoformat() if end else "2099-12-31")
    url = (f"https://api.tiingo.com/tiingo/daily/"
           f"{provider_symbol(ticker, 'tiingo').lower()}/prices"
           f"?startDate={sd}&endDate={ed}&format=json")
    req = urllib.request.Request(url, headers={
        "Content-Type": "application/json",
        "Authorization": f"Token {key}",
        "User-Agent": "LeakProofBacktester/1.0",
    })
    try:
        data = _http_json(req, timeout)
    except json.JSONDecodeError:
        # Tiingo reports an exhausted monthly symbol allowance with HTTP 200 and a
        # plain-text body, so a decode failure usually indicates quota exhaustion.
        raise RuntimeError(
            f"{ticker}: Tiingo returned a non-JSON body -- usually the monthly symbol "
            f"allowance (500 on the free tier) is exhausted")
    if isinstance(data, dict) and data.get("detail"):
        raise RuntimeError(f"{ticker}: Tiingo: {str(data['detail'])[:80]}")
    if not data:
        raise RuntimeError(f"{ticker}: Tiingo returned no rows")
    # Adjusted fields are split and dividend adjusted, as factor calculations expect.
    rows = [{"date": d["date"][:10], "open": d.get("adjOpen") or d["open"],
             "high": d.get("adjHigh") or d["high"], "low": d.get("adjLow") or d["low"],
             "close": d.get("adjClose") or d["close"],
             "volume": d.get("adjVolume") or d.get("volume") or 0} for d in data]
    return _to_csv_text(rows)


def fetch_alpaca(ticker: str, start=None, end=None, timeout: int = 20,
                 feed: str | None = None) -> str:
    """Fetch Alpaca daily bars for one symbol.

    The free tier provides the IEX feed. SIP requires a paid entitlement and can be
    selected with ALPACA_FEED=sip.
    """
    kid = os.environ.get("ALPACA_API_KEY_ID", "").strip()
    sec = os.environ.get("ALPACA_API_SECRET_KEY", "").strip()
    if not (kid and sec):
        raise RuntimeError("ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY are not set")
    import urllib.parse
    import urllib.request
    sd = (start.isoformat() if start else "1990-01-01")
    ed = (end.isoformat() if end else "2099-12-31")
    rows, page_token = [], None
    while True:
        psym = provider_symbol(ticker, "alpaca")
        q = {"symbols": psym, "timeframe": "1Day", "start": sd, "end": ed,
             "adjustment": "all", "limit": "10000",
             "feed": feed or os.environ.get("ALPACA_FEED", "iex")}
        if page_token:
            q["page_token"] = page_token
        url = "https://data.alpaca.markets/v2/stocks/bars?" + urllib.parse.urlencode(q)
        req = urllib.request.Request(url, headers={
            "APCA-API-KEY-ID": kid, "APCA-API-SECRET-KEY": sec,
            "User-Agent": "LeakProofBacktester/1.0",
        })
        data = _http_json(req, timeout)
        for b in (data.get("bars") or {}).get(psym, []):
            rows.append({"date": b["t"][:10], "open": b["o"], "high": b["h"],
                         "low": b["l"], "close": b["c"], "volume": b["v"]})
        page_token = data.get("next_page_token")
        if not page_token:
            break
    if not rows:
        raise RuntimeError(f"{ticker}: Alpaca returned no rows")
    return _to_csv_text(rows)


PROVIDERS = ("stooq", "tiingo", "alpaca")


def fetch_prices(ticker: str, provider: str = "stooq", start=None, end=None) -> str:
    """Fetch daily bars for one symbol from the named provider.

    Returns:
        CSV text in the common provider schema.

    Raises:
        ValueError: If the provider is not recognised.
    """
    if provider == "stooq":
        return fetch_stooq(ticker, start=start, end=end)
    if provider == "tiingo":
        return fetch_tiingo(ticker, start=start, end=end)
    if provider == "alpaca":
        return fetch_alpaca(ticker, start=start, end=end)
    raise ValueError(f"unknown provider: {provider!r}")


# Bulk ingestion. Alpaca accepts multiple symbols per request, so a large universe
# resolves in a small number of paginated calls. Providers without a bulk endpoint
# are fetched concurrently, since the work is I/O bound. Tiingo additionally limits
# free-tier clients to 50 requests per hour; interrupted runs resume from the cache.
ALPACA_BATCH_SIZE = 100


def fetch_alpaca_batch(tickers, start=None, end=None, timeout: int = 30,
                       feed: str | None = None) -> dict[str, str]:
    """Fetch daily bars for multiple symbols in a single request.

    Responses are paginated and each page may contain any subset of the requested
    symbols, so rows accumulate per symbol across pages.

    Returns:
        Mapping of canonical ticker to CSV text, omitting symbols with no data.
    """
    kid = os.environ.get("ALPACA_API_KEY_ID", "").strip()
    sec = os.environ.get("ALPACA_API_SECRET_KEY", "").strip()
    if not (kid and sec):
        raise RuntimeError("ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY are not set")
    import urllib.parse
    import urllib.request

    # Requests use provider notation; responses are mapped back to canonical tickers.
    canon = [t.strip().upper() for t in tickers]
    syms = [provider_symbol(t, "alpaca") for t in canon]
    back = dict(zip(syms, canon))
    sd = (start.isoformat() if start else "1990-01-01")
    ed = (end.isoformat() if end else "2099-12-31")
    acc: dict[str, list[dict]] = {c: [] for c in canon}
    page_token = None
    while True:
        q = {"symbols": ",".join(syms), "timeframe": "1Day", "start": sd, "end": ed,
             "adjustment": "all", "limit": "10000",
             "feed": feed or os.environ.get("ALPACA_FEED", "iex")}
        if page_token:
            q["page_token"] = page_token
        url = "https://data.alpaca.markets/v2/stocks/bars?" + urllib.parse.urlencode(q)
        req = urllib.request.Request(url, headers={
            "APCA-API-KEY-ID": kid, "APCA-API-SECRET-KEY": sec,
            "User-Agent": "LeakProofBacktester/1.0",
        })
        data = _http_json(req, timeout)
        for sym, bars in (data.get("bars") or {}).items():
            key = back.get(sym, sym)
            for b in bars:
                acc.setdefault(key, []).append(
                    {"date": b["t"][:10], "open": b["o"], "high": b["h"],
                     "low": b["l"], "close": b["c"], "volume": b["v"]})
        page_token = data.get("next_page_token")
        if not page_token:
            break
    return {s_: _to_csv_text(rows) for s_, rows in acc.items() if rows}


def supports_batch(provider: str) -> bool:
    """Return True if the provider accepts multiple symbols per request."""
    return provider == "alpaca"


def fetch_many(tickers, provider: str = "stooq", start=None, end=None,
               workers: int = 1, on_result=None, on_error=None) -> None:
    """Fetch many tickers, batching or parallelising by provider capability.

    Callbacks fire as results arrive so callers can persist incrementally, allowing
    an interrupted run to retain completed work.

    Args:
        tickers: Symbols to fetch.
        provider: Provider name.
        start: Window start date.
        end: Window end date.
        workers: Thread count for providers without a bulk endpoint.
        on_result: Called with (ticker, csv_text) per successful fetch.
        on_error: Called with (ticker, message) per failure.
    """
    tickers = list(tickers)

    if supports_batch(provider):
        def _run_chunk(chunk):
            """Fetch a chunk, bisecting on failure to isolate rejected symbols.

            A single unknown or malformed ticker causes the provider to reject the
            entire request, so splitting recovers the remaining symbols at a cost of
            logarithmically many additional requests.
            """
            try:
                got = fetch_alpaca_batch(chunk, start=start, end=end)
            except Exception as e:
                if len(chunk) == 1:
                    if on_error:
                        on_error(chunk[0], f"{str(e)[:70]} (isolated)")
                    return
                mid = len(chunk) // 2
                _run_chunk(chunk[:mid])
                _run_chunk(chunk[mid:])
                return
            for t in chunk:
                if t.upper() in got and on_result:
                    on_result(t, got[t.upper()])
                elif on_error:
                    on_error(t, "no rows returned (not covered by this feed)")

        for i in range(0, len(tickers), ALPACA_BATCH_SIZE):
            _run_chunk(tickers[i:i + ALPACA_BATCH_SIZE])
        return

    if workers > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(fetch_prices, t, provider, start, end): t for t in tickers}
            for fut in as_completed(futs):
                t = futs[fut]
                try:
                    if on_result:
                        on_result(t, fut.result())
                except Exception as e:
                    if on_error:
                        on_error(t, str(e)[:80])
        return

    for t in tickers:
        try:
            if on_result:
                on_result(t, fetch_prices(t, provider=provider, start=start, end=end))
        except Exception as e:
            if on_error:
                on_error(t, str(e)[:80])
