"""Cross-sectional factor definitions.

A factor maps a trailing window of closing prices to a single score. The interface
constrains each implementation to observe only prices up to the current close.

Every factor follows the convention that higher scores are more attractive, so
low-volatility returns negated volatility and reversal negates the recent return.
A consistent sign convention is required for weighted blending across factors.

Each factor declares a warmup length. The composite strategy omits factors that have
insufficient history rather than scoring on a partial window.
"""
from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np

TRADING_DAYS_PER_MONTH = 21


class Factor(Protocol):
    name: str
    warmup: int
    def score(self, closes: np.ndarray) -> float:
        """One raw score from a trailing window of closes (oldest -> newest)."""
        ...


class Momentum:
    """Total return from t-lookback to t-skip.

    The most recent month is typically skipped to avoid contamination from the
    short-term reversal effect.
    """

    def __init__(self, lookback_months: int = 12, skip_months: int = 1):
        self.lookback = lookback_months * TRADING_DAYS_PER_MONTH
        self.skip = skip_months * TRADING_DAYS_PER_MONTH
        self.name = f"momentum_{lookback_months}_{skip_months}"
        self.warmup = self.lookback + 1

    def score(self, closes: np.ndarray) -> float:
        then = closes[-self.warmup]
        recent = closes[-(self.skip + 1)]
        return float(recent / then - 1.0) if then > 0 else float("nan")


class LowVolatility:
    """Negated trailing volatility of daily returns.

    A liquidity floor rejects stale price series. A series that repeats the same
    close, whether through illiquidity or a stale feed, measures near-zero volatility
    and would otherwise rank at the top of the screen despite carrying normal risk.
    Rejected securities score NaN and are dropped from the cross-section.
    """

    def __init__(self, lookback_months: int = 12, max_zero_share: float = 0.25,
                 max_flat_run: int = 5):
        self.lookback = lookback_months * TRADING_DAYS_PER_MONTH
        self.name = f"lowvol_{lookback_months}"
        self.warmup = self.lookback + 1
        self.max_zero_share = max_zero_share
        self.max_flat_run = max_flat_run

    def score(self, closes: np.ndarray) -> float:
        w = closes[-self.warmup:]
        rets = w[1:] / w[:-1] - 1.0
        if len(rets) < 2:
            return float("nan")

        zero_share = float(np.mean(rets == 0.0))
        if zero_share > self.max_zero_share:
            return float("nan")
        flat = np.concatenate([[False], w[1:] == w[:-1]])
        longest = cur = 0
        for f_ in flat:
            cur = cur + 1 if f_ else 0
            longest = max(longest, cur)
        if longest >= self.max_flat_run:
            return float("nan")

        sd = float(np.std(rets, ddof=1))
        if not np.isfinite(sd) or sd <= 0:
            return float("nan")
        return -sd


class ShortTermReversal:
    """Negated recent return, capturing short-horizon mean reversion."""

    def __init__(self, lookback_months: int = 1):
        self.lookback = lookback_months * TRADING_DAYS_PER_MONTH
        self.name = f"reversal_{lookback_months}"
        self.warmup = self.lookback + 1

    def score(self, closes: np.ndarray) -> float:
        then = closes[-self.warmup]
        return float(-(closes[-1] / then - 1.0)) if then > 0 else float("nan")



def build_factor(spec: str, lookback_months: int, skip_months: int) -> Factor:
    """Instantiate a factor from its specification name.

    Args:
        spec: Factor name, one of AVAILABLE_FACTORS.
        lookback_months: Lookback window in months.
        skip_months: Months skipped before the lookback window ends.

    Raises:
        ValueError: If the specification is not recognised.
    """
    s = spec.strip().lower()
    if s in ("momentum", "mom"):
        return Momentum(lookback_months, skip_months)
    if s in ("lowvol", "low_volatility", "lowvolatility"):
        return LowVolatility(lookback_months)
    if s in ("reversal", "streversal", "short_term_reversal"):
        return ShortTermReversal(1)
    raise ValueError(f"unknown factor: {spec!r}")


AVAILABLE_FACTORS = ("momentum", "lowvol", "reversal")



def zscore(values: Sequence[float], winsorise: float = 3.0) -> np.ndarray:
    """Standardise values across the cross-section, winsorising the tails.

    Raw factor values occupy incompatible scales and cannot be averaged directly.
    Standardising within each date expresses every factor in cross-sectional standard
    deviations, which makes a weighted blend well defined. Winsorising limits the
    influence of a single outlier, which matters most in small universes.

    Args:
        values: Raw factor values for one date.
        winsorise: Absolute z-score cap.

    Returns:
        Array of z-scores with NaN preserved for unscoreable entries.
    """
    v = np.asarray(values, dtype=float)
    finite = np.isfinite(v)
    out = np.full(v.shape, np.nan)
    if finite.sum() < 2:
        return out
    mu = v[finite].mean()
    sd = v[finite].std(ddof=1)
    if sd <= 0:
        out[finite] = 0.0
        return out
    z = (v - mu) / sd
    return np.clip(z, -winsorise, winsorise)
