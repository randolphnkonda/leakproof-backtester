"""Compare allocators and quantify the effect of covariance estimation noise.

Part A evaluates equal weighting, minimum variance, and maximum decorrelation on an
identical signal.

Part B bootstraps a return window, re-solves the minimum-variance weights on each
resample, and measures weight dispersion under sample and shrinkage covariance
estimators.
"""
from __future__ import annotations

from datetime import date

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from backtester import BacktestConfig, run_backtest
from backtester.data import SyntheticDataHandler, _SYNTHETIC_SYMBOLS
from backtester.history import RollingHistory
from backtester.optimize import active_backend, solve_min_variance
from backtester.risk_model import estimate_covariance, ledoit_wolf_cov

_ANN = np.sqrt(252)


def _metrics(ec):
    rets = ec.pct_change().dropna()
    ann_ret = (1 + rets.mean()) ** 252 - 1
    ann_vol = rets.std(ddof=1) * _ANN
    sharpe = ann_ret / ann_vol if ann_vol > 0 else float("nan")
    dd = (ec / ec.cummax() - 1).min()
    return ann_ret, ann_vol, sharpe, dd


def part_a() -> None:
    print("=" * 64)
    print(f"PART A: allocator comparison (optimiser backend: {active_backend()})")
    print("=" * 64)
    print(f"{'allocator':18} {'ann_ret':>8} {'ann_vol':>8} {'sharpe':>7} {'max_dd':>7}")
    for alloc in ("equal_weight", "min_variance", "max_decorrelation"):
        cfg = BacktestConfig(start=date(2015, 1, 1), end=date(2023, 12, 31),
                             seed=7, n_long=6, allocator=alloc)
        ar, av, sh, dd = _metrics(run_backtest(cfg).equity_curve)
        print(f"{alloc:18} {ar:>7.1%} {av:>8.3f} {sh:>7.2f} {dd:>7.1%}")
    print("Note: synthetic series are near-uncorrelated, so minimum variance")
    print("approximates inverse-volatility weighting here.")


def _window_returns(lookback: int) -> np.ndarray:
    """Return one trailing return window for the full synthetic universe."""
    data = SyntheticDataHandler(date(2015, 1, 1), date(2023, 12, 31), seed=7)
    hist = RollingHistory(maxlen=lookback + 5)
    for e in data.stream():
        hist.update(e)
    _, R = hist.returns(list(_SYNTHETIC_SYMBOLS), lookback)
    return R


def _bootstrap_weight_instability(R: np.ndarray, method: str, B: int, seed: int):
    """Resample returns with replacement and re-solve weights on each draw."""
    rng = np.random.default_rng(seed)
    T = R.shape[0]
    W = []
    for _ in range(B):
        idx = rng.integers(0, T, size=T)
        Sigma = estimate_covariance(R[idx], method=method)
        W.append(solve_min_variance(Sigma))
    return np.array(W)  # (B x k)


def part_b() -> None:
    print()
    print("=" * 64)
    print("PART B: covariance noise -> weight instability")
    print("=" * 64)
    short, long = 60, 252
    R_short = _window_returns(short)
    R_long = _window_returns(long)
    k = R_short.shape[1]

    _, shrink_short = ledoit_wolf_cov(R_short)
    _, shrink_long = ledoit_wolf_cov(R_long)
    print(f"Universe size k = {k}")
    print(f"Ledoit-Wolf shrinkage intensity @ T={short}: {shrink_short:.2f} "
          f"(noisy window -> shrink hard)")
    print(f"Ledoit-Wolf shrinkage intensity @ T={long}: {shrink_long:.2f} "
          f"(more data -> trust the sample more)")

    B = 400
    W_sample = _bootstrap_weight_instability(R_short, "sample", B, seed=1)
    W_lw = _bootstrap_weight_instability(R_short, "ledoit_wolf", B, seed=1)

    # Instability is the mean across securities of each weight's bootstrap std.
    inst_sample = W_sample.std(axis=0, ddof=1).mean()
    inst_lw = W_lw.std(axis=0, ddof=1).mean()
    print(f"\nBootstrap weight instability at T={short} (B={B} resamples):")
    print(f"  sample covariance : {inst_sample:.4f}  (avg per-name weight std)")
    print(f"  Ledoit-Wolf       : {inst_lw:.4f}")
    print(f"  reduction         : {(1 - inst_lw / inst_sample):.0%} less weight noise")
    print("Inverting a noisy covariance matrix amplifies estimation error into")
    print("unstable allocations. Shrinkage reduces the effect.")

    # Plot weight dispersion by estimator.
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(k)
    ax.bar(x - 0.2, W_sample.mean(0), width=0.4, yerr=W_sample.std(0),
           capsize=3, label="sample cov", color="#D85A30")
    ax.bar(x + 0.2, W_lw.mean(0), width=0.4, yerr=W_lw.std(0),
           capsize=3, label="Ledoit-Wolf", color="#1D9E75")
    ax.set_xticks(x)
    ax.set_xticklabels(list(_SYNTHETIC_SYMBOLS))
    ax.set_ylabel("min-variance weight")
    ax.set_title(f"Bootstrap weight spread at T={short} (error bars = std over {B} resamples)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("optimizer_weight_stability.png", dpi=110)
    print("\nSaved: optimizer_weight_stability.png")


def main() -> None:
    part_a()
    part_b()


if __name__ == "__main__":
    main()
