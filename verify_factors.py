"""Verify the factor library and composite strategy.

Checks:

    1. Standardisation centres values, winsorises outliers, handles zero variance,
       and preserves NaN for unscoreable securities.
    2. Every factor follows the higher-is-better sign convention.
    3. The liquidity floor rejects stale series without altering the ordering of
       genuine ones.
    4. Blending applies weights: a factor combined with its own negation cancels.
    5. Composite scores renormalise over the factors that meet their warmup.
    6. Composite strategies introduce no look-ahead.
"""
from __future__ import annotations

from datetime import date

import numpy as np

from backtester import BacktestConfig, run_backtest
from backtester.factors import (
    LowVolatility, Momentum, ShortTermReversal, build_factor, zscore,
)
from backtester.strategy import CompositeStrategy


def check_zscore() -> None:
    z = zscore([1.0, 2.0, 3.0, 4.0])
    assert abs(float(np.nanmean(z))) < 1e-12, "z-score should be mean zero"
    assert z.max() <= 3.0 + 1e-9, "winsorisation failed"
    assert np.allclose(zscore([5.0, 5.0, 5.0]), 0.0), "zero variance should give zeros"
    z_nan = zscore([1.0, np.nan, 3.0])
    assert np.isnan(z_nan[1]) and np.isfinite(z_nan[0]), "NaN must be preserved"
    assert np.isnan(zscore([1.0])).all(), "a single observation cannot be standardised"
    print("z-scoring: standardises, winsorises, handles zero-variance and NaN -> OK")


def check_sign_convention() -> None:
    rng = np.random.default_rng(0)
    calm = 100 * np.cumprod(1 + rng.normal(0, 0.002, 400))
    wild = 100 * np.cumprod(1 + rng.normal(0, 0.02, 400))
    assert LowVolatility(12).score(calm) > LowVolatility(12).score(wild), \
        "low volatility must prefer the calmer series"

    up = 100 * np.cumprod(1 + np.full(400, 0.001))
    down = 100 * np.cumprod(1 + np.full(400, -0.001))
    assert Momentum(12, 1).score(up) > Momentum(12, 1).score(down), \
        "momentum must prefer the riser"
    assert ShortTermReversal(1).score(down) > ShortTermReversal(1).score(up), \
        "reversal must prefer the recent loser"
    print("sign convention: every factor is higher-is-better -> OK")


class _Opposite:
    """Factor returning the negation of another factor's score."""
    def __init__(self, base):
        self._base = base
        self.name = "anti_" + base.name
        self.warmup = base.warmup

    def score(self, closes):
        return -self._base.score(closes)


def check_lowvol_liquidity_floor() -> None:
    """Stale series must be rejected by the low-volatility screen."""
    lv = LowVolatility(12)
    n = lv.warmup
    rng = np.random.default_rng(3)
    healthy = 100 * np.cumprod(1 + rng.normal(0, 0.01, n))
    flat = np.full(n, 100.0)
    mostly_flat = np.full(n, 100.0)
    mostly_flat[-15:] = 100 + np.arange(15) * 0.5

    assert np.isfinite(lv.score(healthy)), "a healthy series must still score"
    assert np.isnan(lv.score(flat)), "a fully flat series must be disqualified"
    assert np.isnan(lv.score(mostly_flat)), "a mostly-flat series must be disqualified"

    calm = 100 * np.cumprod(1 + rng.normal(0, 0.002, n))
    wild = 100 * np.cumprod(1 + rng.normal(0, 0.03, n))
    assert lv.score(calm) > lv.score(wild), "the floor must not break the ordering"
    print("low-vol liquidity floor: stale series disqualified, ordering intact -> OK")


def check_blending_cancels() -> None:
    cfg = BacktestConfig(start=date(2015, 1, 1), end=date(2016, 6, 1),
                         factors="momentum", seed=7)
    strat = CompositeStrategy(cfg)
    mom = build_factor("momentum", cfg.lookback_months, cfg.skip_months)
    strat.factors = [mom, _Opposite(mom)]
    strat.weights = np.array([0.5, 0.5])
    strat.max_warmup = mom.warmup

    from backtester.data import SyntheticDataHandler
    data = SyntheticDataHandler(cfg.start, cfg.end, seed=cfg.seed)
    worst = 0.0
    for e in data.stream():
        sig = strat.on_market(e)
        if sig:
            worst = max(worst, max(abs(v) for v in sig.scores.values()))
    assert worst < 1e-9, f"a factor plus its opposite should cancel, got {worst}"
    print("blending: a factor paired with its own opposite cancels to zero -> OK")


def check_warmup_renormalisation() -> None:
    """Composites must score using whichever factors have met their warmup."""
    cfg = BacktestConfig(start=date(2015, 1, 1), end=date(2015, 6, 1),
                         factors="momentum,reversal", seed=7)
    strat = CompositeStrategy(cfg)
    from backtester.data import SyntheticDataHandler
    scored_early = False
    for e in SyntheticDataHandler(cfg.start, cfg.end, seed=cfg.seed).stream():
        sig = strat.on_market(e)
        if sig and len(sig.scores) > 0:
            scored_early = True
            break
    assert scored_early, "composite should score off the warm factor alone"
    print("warmup: composite renormalises over the factors actually available -> OK")


def check_no_lookahead_composite() -> None:
    from backtester.engine import _build_allocator
    from backtester.execution import SimulatedBroker
    from backtester.history import RollingHistory
    from backtester.portfolio import SimplePortfolio
    from backtester.risk import PassThroughRiskManager
    from backtester.data import SyntheticDataHandler

    cfg = BacktestConfig(start=date(2015, 1, 1), end=date(2018, 12, 31),
                         factors="momentum,lowvol", allocator="min_variance", seed=7)
    events = list(SyntheticDataHandler(cfg.start, cfg.end, seed=cfg.seed).stream())
    strat, alloc = CompositeStrategy(cfg), _build_allocator(cfg)
    risk, broker = PassThroughRiskManager(), SimulatedBroker(cfg)
    pf = SimplePortfolio(cfg.initial_cash)
    hist = RollingHistory(maxlen=cfg.cov_lookback_days + 5)
    opens = {e.ts: {s: b.open for s, b in e.bars.items()} for e in events}

    checked = violations = 0
    for i, e in enumerate(events):
        for fill in broker.execute_pending(e):
            if not (fill.ts > events[i - 1].ts
                    and abs(fill.fill_price - opens[e.ts][fill.symbol]) < 1e-9):
                violations += 1
            checked += 1
            pf.on_fill(fill)
        pf.mark_to_market(e)
        hist.update(e)
        sig = strat.on_market(e)
        if sig:
            for o in risk.vet(alloc.target_orders(sig, pf.state(), hist), pf.state()):
                broker.queue(o)
    assert violations == 0, f"{violations} look-ahead violations with a composite"
    print(f"look-ahead with composite: {checked} fills checked, 0 violations -> OK")


def check_runs() -> None:
    base = dict(start=date(2015, 1, 1), end=date(2020, 12, 31), seed=7)
    for spec in ("momentum", "lowvol", "reversal", "momentum,lowvol,reversal"):
        r = run_backtest(BacktestConfig(**base, factors=spec))
        assert r.n_trades > 0 and np.isfinite(r.sharpe), f"{spec} produced no result"
    print("all factor specs produce runnable backtests -> OK")


def main() -> None:
    check_zscore()
    check_sign_convention()
    check_lowvol_liquidity_floor()
    check_blending_cancels()
    check_warmup_renormalisation()
    check_no_lookahead_composite()
    check_runs()
    print("RESULT: PASS")


if __name__ == "__main__":
    main()
