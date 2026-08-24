"""Probabilistic and deflated Sharpe ratios.

Implements the framework of Bailey and Lopez de Prado (2014).

A Sharpe ratio estimated from T observations is itself a random variable, and
skewness and excess kurtosis increase its standard error. The probabilistic Sharpe
ratio PSR(SR*) gives the probability that the true Sharpe ratio exceeds a benchmark
SR*, accounting for sample length and higher moments.

Selecting the best of N strategies inflates the winner's Sharpe ratio through
selection alone: the maximum of N noisy estimates is positive even when no strategy
has a true edge. The deflated Sharpe ratio sets the benchmark to the expected maximum
Sharpe ratio under the null hypothesis that all true Sharpe ratios are zero, given N
trials and the dispersion of trial Sharpe ratios. A value at or above 0.95 indicates
significance at the 95% level after accounting for the search.

All calculations use per-period units, since the higher-moment adjustment is defined
on per-period returns. Annualised inputs are divided by sqrt(252); benchmarks are
converted back to annualised units for reporting only.

The variance of trial Sharpe ratios is estimated two ways. The empirical estimator
uses the observed dispersion of trials, which understates the spread of a genuinely
independent search when configurations overlap. The analytic estimator uses the
sampling variance of a Sharpe estimator under the null, which treats trials as
independent and therefore overstates the effective number of trials. Both are
reported; the appropriate value lies between them.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

_N = NormalDist()
_EULER = 0.5772156649015329          # Euler-Mascheroni constant
_ANNUALISATION = 252
_SQRT_ANN = math.sqrt(_ANNUALISATION)


def probabilistic_sharpe_ratio(
    sr: float, sr_benchmark: float, T: int, skew: float, kurt_excess: float
) -> float:
    """Return the probability that the true Sharpe ratio exceeds a benchmark.

    Args:
        sr: Estimated per-period Sharpe ratio.
        sr_benchmark: Per-period benchmark to test against.
        T: Number of return observations.
        skew: Skewness of periodic returns.
        kurt_excess: Excess kurtosis of periodic returns.

    Returns:
        Probability in [0, 1].
    """
    # Expressed with non-excess kurtosis: (gamma4 - 1) / 4 = (excess + 2) / 4.
    var_term = 1.0 - skew * sr + ((kurt_excess + 2.0) / 4.0) * sr * sr
    denom = math.sqrt(max(var_term, 1e-12))
    z = (sr - sr_benchmark) * math.sqrt(max(T - 1, 1)) / denom
    return _N.cdf(z)


def expected_max_sharpe(var_sr: float, n_trials: int) -> float:
    """Return the expected maximum of N independent null Sharpe ratios.

    Args:
        var_sr: Variance of trial Sharpe ratios, per-period.
        n_trials: Number of trials.

    Returns:
        Expected maximum Sharpe ratio, per-period.
    """
    if n_trials < 2 or var_sr <= 0.0:
        return 0.0
    z1 = _N.inv_cdf(1.0 - 1.0 / n_trials)
    z2 = _N.inv_cdf(1.0 - 1.0 / (n_trials * math.e))
    return math.sqrt(var_sr) * ((1.0 - _EULER) * z1 + _EULER * z2)


def deflated_sharpe_ratio(
    sr: float, var_sr: float, n_trials: int, T: int, skew: float, kurt_excess: float
) -> tuple[float, float]:
    """Return the deflated Sharpe ratio and its benchmark, both per-period."""
    sr_star = expected_max_sharpe(var_sr, n_trials)
    dsr = probabilistic_sharpe_ratio(sr, sr_star, T, skew, kurt_excess)
    return dsr, sr_star


def monte_carlo_expected_max(
    var_sr: float, n_trials: int, n_sims: int = 50_000, seed: int = 0
) -> float:
    """Simulate the maximum of N null Sharpe ratios.

    Used to validate the closed-form expected_max_sharpe approximation.
    """
    rng = np.random.default_rng(seed)
    draws = rng.normal(0.0, math.sqrt(var_sr), size=(n_sims, n_trials))
    return float(draws.max(axis=1).mean())


@dataclass(frozen=True)
class DeflationReport:
    """Deflation results for the best trial in a sweep."""

    n_trials: int
    T: int
    best_sharpe_ann: float
    skew: float
    kurt_excess: float
    psr_vs_zero: float            # probabilistic Sharpe against a zero benchmark
    # empirical variance branch
    var_sr_emp: float             # per-period
    sr_star_emp_ann: float        # benchmark, annualised for display
    dsr_emp: float
    # analytic variance branch
    var_sr_ana: float             # per-period
    sr_star_ana_ann: float
    dsr_ana: float

    def summary(self) -> str:
        def verdict(p):
            return "significant" if p >= 0.95 else "NOT significant"
        lines = [
            f"Trials (N)                 : {self.n_trials}",
            f"Sample length (T days)     : {self.T}",
            f"Best Sharpe (annualised)   : {self.best_sharpe_ann:.2f}"
            f"  [skew={self.skew:.2f}, excess_kurt={self.kurt_excess:.2f}]",
            f"PSR vs 0 (no deflation)    : {self.psr_vs_zero:.3f}  "
            f"-> {verdict(self.psr_vs_zero)} at 95%",
            "-" * 60,
            f"Empirical Var(SR): benchmark SR* = {self.sr_star_emp_ann:.2f} ann",
            f"  Deflated Sharpe (DSR_emp): {self.dsr_emp:.3f}  "
            f"-> {verdict(self.dsr_emp)} at 95%",
            f"Analytic  Var(SR): benchmark SR* = {self.sr_star_ana_ann:.2f} ann",
            f"  Deflated Sharpe (DSR_ana): {self.dsr_ana:.3f}  "
            f"-> {verdict(self.dsr_ana)} at 95%",
        ]
        return "\n".join(lines)


def empirical_var_sr(sweep) -> float:
    """Return the per-period variance of observed trial Sharpe ratios."""
    s = sweep.sharpes / _SQRT_ANN
    s = s[np.isfinite(s)]
    return float(np.var(s, ddof=1)) if s.size >= 2 else 0.0


def analytic_var_sr(T: int, sr_per_period: float = 0.0) -> float:
    """Return the sampling variance of a Sharpe estimator under IID returns.

    Follows Lo (2002): (1 + SR^2 / 2) / (T - 1), evaluated at the null by default.
    """
    return (1.0 + 0.5 * sr_per_period ** 2) / max(T - 1, 1)


def analyze_sweep(sweep) -> DeflationReport:
    """Compute deflation statistics for the best trial in a sweep."""
    best = sweep.best
    T = best.n_periods
    sr_ann = best.sharpe
    sr_daily = sr_ann / _SQRT_ANN
    skew, kurt = best.skew, best.kurt_excess
    N = sweep.n_trials

    psr0 = probabilistic_sharpe_ratio(sr_daily, 0.0, T, skew, kurt)

    v_emp = empirical_var_sr(sweep)
    dsr_emp, star_emp = deflated_sharpe_ratio(sr_daily, v_emp, N, T, skew, kurt)

    v_ana = analytic_var_sr(T, 0.0)
    dsr_ana, star_ana = deflated_sharpe_ratio(sr_daily, v_ana, N, T, skew, kurt)

    return DeflationReport(
        n_trials=N, T=T, best_sharpe_ann=sr_ann, skew=skew, kurt_excess=kurt,
        psr_vs_zero=psr0,
        var_sr_emp=v_emp, sr_star_emp_ann=star_emp * _SQRT_ANN, dsr_emp=dsr_emp,
        var_sr_ana=v_ana, sr_star_ana_ann=star_ana * _SQRT_ANN, dsr_ana=dsr_ana,
    )
