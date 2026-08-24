"""Validation and normalisation of raw price data.

Rows failing validation are dropped rather than repaired, and no prices are
interpolated or forward filled. Each call returns a report of what was removed so
that ingestion remains observable.

Stooq daily prices are split adjusted but not dividend adjusted. Dividend adjustment
is not applied here.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class CleanReport:
    """Counts of rows removed by each validation rule."""

    ticker: str
    rows_in: int
    rows_out: int
    dropped_nulls: int
    dropped_nonpositive: int
    dropped_ohlc_violations: int
    dropped_duplicates: int

    def line(self) -> str:
        kept = f"{self.rows_out}/{self.rows_in}"
        return (f"{self.ticker:8} kept {kept:>11}  "
                f"drops: null={self.dropped_nulls} nonpos={self.dropped_nonpositive} "
                f"ohlc={self.dropped_ohlc_violations} dup={self.dropped_duplicates}")


def clean_bars(raw: pd.DataFrame, ticker: str) -> tuple[pd.DataFrame, CleanReport]:
    """Validate and normalise raw OHLCV data for one security.

    Removes rows with missing or non-positive prices, rows violating OHLC ordering,
    and duplicate dates, retaining the last observation for each date.

    Args:
        raw: Frame with Date, Open, High, Low, Close, Volume columns.
        ticker: Symbol to attach to the output.

    Returns:
        Tuple of (tidy frame sorted by date, validation report).
    """
    rows_in = len(raw)
    df = raw.rename(columns=str.lower)[["date", "open", "high", "low", "close", "volume"]].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    n0 = len(df)
    df = df.dropna(subset=["date", "open", "high", "low", "close"])
    dropped_nulls = n0 - len(df)

    n1 = len(df)
    df = df[(df[["open", "high", "low", "close"]] > 0).all(axis=1)]
    dropped_nonpositive = n1 - len(df)

    n2 = len(df)
    hi_ok = df["high"] >= df[["open", "close"]].max(axis=1) - 1e-9
    lo_ok = df["low"] <= df[["open", "close"]].min(axis=1) + 1e-9
    df = df[hi_ok & lo_ok]
    dropped_ohlc = n2 - len(df)

    n3 = len(df)
    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    dropped_dup = n3 - len(df)

    df["volume"] = df["volume"].fillna(0).clip(lower=0)
    df.insert(0, "symbol", ticker.upper())
    df = df.reset_index(drop=True)

    report = CleanReport(
        ticker=ticker.upper(), rows_in=rows_in, rows_out=len(df),
        dropped_nulls=dropped_nulls, dropped_nonpositive=dropped_nonpositive,
        dropped_ohlc_violations=dropped_ohlc, dropped_duplicates=dropped_dup,
    )
    return df, report
