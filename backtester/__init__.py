"""Event-driven multi-factor equity backtesting framework."""
from .config import BacktestConfig, BacktestResult
from .engine import run_backtest
from .sweep import (
    SweepResult,
    TrialRecord,
    build_grid,
    run_sweep,
)
from .deflated_sharpe import (
    DeflationReport,
    analyze_sweep,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    monte_carlo_expected_max,
    probabilistic_sharpe_ratio,
)

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "run_backtest",
    "SweepResult",
    "TrialRecord",
    "build_grid",
    "run_sweep",
    "DeflationReport",
    "analyze_sweep",
    "deflated_sharpe_ratio",
    "expected_max_sharpe",
    "monte_carlo_expected_max",
    "probabilistic_sharpe_ratio",
]
