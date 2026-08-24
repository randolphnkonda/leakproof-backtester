"""Persistent storage for price bars and index membership.

Two interchangeable backends implement the BarStore interface:

    DuckDBParquetStore      columnar Parquet queried in-process by DuckDB
    SqliteStore             stdlib sqlite3, used where DuckDB is unavailable

Panel construction and point-in-time membership resolution are implemented once in
the base class; backends supply only reading and writing.

open_store selects the backend matching the format an existing store was written in,
falling back to the preferred backend for new stores. Selecting purely on installed
packages would open a sqlite store with the Parquet reader.
"""
from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

STORE_VERSION = "2026.08.3"

_FIELDS = ("open", "high", "low", "close", "volume")
_BAR_COLS = ["symbol", "date", "open", "high", "low", "close", "volume"]


class BarStore:
    """Base class implementing panel and membership queries over a backend."""

    # Backend interface.
    def write_bars(self, tidy: pd.DataFrame) -> None: raise NotImplementedError
    def write_membership(self, intervals: pd.DataFrame) -> None: raise NotImplementedError
    def _fetch_bars(self, symbols, start, end) -> pd.DataFrame: raise NotImplementedError
    def _fetch_membership(self) -> pd.DataFrame: raise NotImplementedError
    def _bar_symbols(self) -> list[str]: raise NotImplementedError
    def has_bars(self) -> bool:
        """Return True if the store contains any price bars."""
        return bool(self._bar_symbols())

    # Shared query implementations.
    def symbols(self) -> list[str]:
        """Return all symbols present in bars or membership records."""
        syms = set(self._bar_symbols())
        m = self._fetch_membership()
        if len(m):
            syms |= set(m["symbol"].astype(str))
        return sorted(syms)

    def universe_as_of(self, as_of: date) -> set[str]:
        """Return index constituents as of the given date."""
        m = self._fetch_membership()
        if not len(m):
            return set()
        live = [
            (sd is not None and sd <= as_of) and (ed is None or ed > as_of)
            for sd, ed in zip(m["start_date"], m["end_date"])
        ]
        return set(m.loc[live, "symbol"])

    def panel(self, symbols: Iterable[str], start: date, end: date):
        """Return aligned price arrays for the requested symbols.

        Args:
            symbols: Symbols to load.
            start: Window start date.
            end: Window end date.

        Returns:
            Tuple of (dates, data), where data[symbol][field] is an array aligned to
            dates, holding NaN where no bar exists.
        """
        symbols = [s for s in symbols]
        if not symbols:
            return [], {}
        rows = self._fetch_bars(symbols, start, end)
        if not len(rows):
            return [], {}
        rows = rows.sort_values(["date", "symbol"])
        dates = sorted(rows["date"].unique())
        idx = {d: i for i, d in enumerate(dates)}
        n = len(dates)
        data: dict[str, dict[str, np.ndarray]] = {
            s: {f: np.full(n, np.nan) for f in _FIELDS} for s in symbols
        }
        for r in rows.itertuples(index=False):
            if r.symbol not in data:
                continue
            i = idx[r.date]
            cell = data[r.symbol]
            cell["open"][i] = r.open
            cell["high"][i] = r.high
            cell["low"][i] = r.low
            cell["close"][i] = r.close
            cell["volume"][i] = r.volume
        return dates, data


def _empty_bars() -> pd.DataFrame:
    return pd.DataFrame(columns=_BAR_COLS)


def _empty_membership() -> pd.DataFrame:
    return pd.DataFrame(columns=["symbol", "start_date", "end_date"])


def _normalise_membership(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce membership dates to date objects with None for open intervals.

    pandas retains a datetime64 column when no value is null, which fails to compare
    against date objects. Object dtype makes membership comparisons uniform.
    """
    if not len(df):
        return df
    out = df.copy()
    for c in ("start_date", "end_date"):
        vals = pd.to_datetime(out[c], errors="coerce")
        out[c] = [None if pd.isna(v) else v.date() for v in vals]
    out["symbol"] = out["symbol"].astype(str)
    return out


# --------------------------------------------------------------------------- #
class DuckDBParquetStore(BarStore):
    """Bar store backed by Parquet files queried through DuckDB."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.bars_dir = self.root / "bars"
        self.membership_path = self.root / "membership.parquet"
        # The bars directory is created on first write so that opening an empty
        # store does not make it appear partially populated.
        try:
            import duckdb  # noqa: F401
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "DuckDBParquetStore needs duckdb + a Parquet engine. "
                "pip install duckdb pyarrow, or use SqliteStore / backend='sqlite'."
            ) from e

    def _con(self):
        import duckdb
        return duckdb.connect()

    def _glob(self) -> str:
        return (self.bars_dir / "*.parquet").as_posix()

    def _has_parquet(self) -> bool:
        return any(self.bars_dir.glob("*.parquet"))

    def write_bars(self, tidy: pd.DataFrame) -> None:
        df = tidy.copy()
        df["date"] = pd.to_datetime(df["date"])
        self.bars_dir.mkdir(parents=True, exist_ok=True)
        sym = df["symbol"].iloc[0]
        path = self.bars_dir / f"{sym}.parquet"
        con = self._con()
        con.register("t", df)
        con.execute(f"COPY t TO '{path.as_posix()}' (FORMAT PARQUET)")
        con.close()

    def write_membership(self, intervals: pd.DataFrame) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        df = intervals.copy()
        df["start_date"] = pd.to_datetime(df["start_date"])
        df["end_date"] = pd.to_datetime(df["end_date"])
        con = self._con()
        con.register("m", df)
        con.execute(f"COPY m TO '{self.membership_path.as_posix()}' (FORMAT PARQUET)")
        con.close()

    def _bar_symbols(self) -> list[str]:
        if not self._has_parquet():
            return []
        con = self._con()
        try:
            df = con.execute(
                f"SELECT DISTINCT symbol FROM read_parquet('{self._glob()}')").df()
        except Exception:
            return []
        finally:
            con.close()
        return sorted(df["symbol"].astype(str).tolist())

    def _fetch_bars(self, symbols, start, end) -> pd.DataFrame:
        symbols = list(symbols)
        if not symbols or not self._has_parquet():
            return _empty_bars()
        con = self._con()
        placeholders = ",".join("?" * len(symbols))
        try:
            df = con.execute(
                f"SELECT symbol, date, open, high, low, close, volume "
                f"FROM read_parquet('{self._glob()}') "
                f"WHERE symbol IN ({placeholders}) AND date BETWEEN ? AND ?",
                symbols + [pd.Timestamp(start), pd.Timestamp(end)],
            ).df()
        except Exception:
            # Files removed between the existence check and the read, or unreadable.
            # Missing data is reported by the caller rather than raised here.
            return _empty_bars()
        finally:
            con.close()
        df["date"] = pd.to_datetime(df["date"]).dt.date
        return df

    def _fetch_membership(self) -> pd.DataFrame:
        if not self.membership_path.exists():
            return _empty_membership()
        con = self._con()
        df = con.execute(
            f"SELECT * FROM read_parquet('{self.membership_path.as_posix()}')").df()
        con.close()
        return _normalise_membership(df)


# --------------------------------------------------------------------------- #
class SqliteStore(BarStore):
    """Bar store backed by sqlite3, with dates stored as ISO-8601 text."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = self.root / "store.sqlite"
        con = sqlite3.connect(self.db)
        con.execute("CREATE TABLE IF NOT EXISTS bars("
                    "symbol TEXT, date TEXT, open REAL, high REAL, low REAL, "
                    "close REAL, volume REAL, PRIMARY KEY(symbol, date))")
        con.execute("CREATE TABLE IF NOT EXISTS membership("
                    "symbol TEXT, start_date TEXT, end_date TEXT)")
        con.commit()
        con.close()

    def write_bars(self, tidy: pd.DataFrame) -> None:
        con = sqlite3.connect(self.db)
        con.executemany(
            "INSERT OR REPLACE INTO bars VALUES (?,?,?,?,?,?,?)",
            [(r.symbol, r.date.isoformat(), float(r.open), float(r.high),
              float(r.low), float(r.close), float(r.volume))
             for r in tidy.itertuples(index=False)],
        )
        con.commit()
        con.close()

    def write_membership(self, intervals: pd.DataFrame) -> None:
        con = sqlite3.connect(self.db)
        con.execute("DELETE FROM membership")
        con.executemany(
            "INSERT INTO membership VALUES (?,?,?)",
            [(r.symbol, r.start_date.isoformat(),
              None if pd.isna(r.end_date) else r.end_date.isoformat())
             for r in intervals.itertuples(index=False)],
        )
        con.commit()
        con.close()

    def _bar_symbols(self) -> list[str]:
        con = sqlite3.connect(self.db)
        rows = con.execute("SELECT DISTINCT symbol FROM bars").fetchall()
        con.close()
        return sorted(r[0] for r in rows)

    def _fetch_bars(self, symbols, start, end) -> pd.DataFrame:
        symbols = list(symbols)
        if not symbols:
            return _empty_bars()
        con = sqlite3.connect(self.db)
        ph = ",".join("?" * len(symbols))
        df = pd.read_sql_query(
            f"SELECT symbol,date,open,high,low,close,volume FROM bars "
            f"WHERE symbol IN ({ph}) AND date BETWEEN ? AND ?",
            con, params=symbols + [start.isoformat(), end.isoformat()],
        )
        con.close()
        df["date"] = pd.to_datetime(df["date"]).dt.date
        return df

    def _fetch_membership(self) -> pd.DataFrame:
        con = sqlite3.connect(self.db)
        df = pd.read_sql_query("SELECT symbol,start_date,end_date FROM membership", con)
        con.close()
        return _normalise_membership(df)


# --------------------------------------------------------------------------- #
def _duck_available() -> bool:
    """Return True if DuckDB is importable. Parquet support is built in."""
    try:
        import duckdb  # noqa: F401
        return True
    except Exception:
        return False


def _choose_backend(root: Path, have_duck: bool) -> str:
    """Match an existing store's on-disk format first; only then prefer DuckDB.

    This is the fix for opening a sqlite-format store on a machine that also has
    DuckDB installed: we must read it with the backend it was written in.
    """
    parquet_exists = (root / "bars").exists() and any((root / "bars").glob("*.parquet"))
    sqlite_exists = (root / "store.sqlite").exists()
    if parquet_exists:
        return "duckdb"
    if sqlite_exists:
        return "sqlite"
    return "duckdb" if have_duck else "sqlite"


def open_store(root: str | Path, backend: str = "auto") -> BarStore:
    """Open a bar store at the given path.

    Args:
        root: Store directory.
        backend: "auto", "duckdb", or "sqlite".
    """
    root = Path(root)
    if backend == "duckdb":
        return DuckDBParquetStore(root)
    if backend == "sqlite":
        return SqliteStore(root)
    choice = _choose_backend(root, _duck_available())
    return DuckDBParquetStore(root) if choice == "duckdb" else SqliteStore(root)
