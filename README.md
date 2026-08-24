# Leak-Proof Multi-Factor Backtester

[![CI](https://github.com/randolphnkonda/leakproof-backtester/actions/workflows/ci.yml/badge.svg)](https://github.com/randolphnkonda/leakproof-backtester/actions/workflows/ci.yml)

An event-driven equity backtesting framework built around statistical honesty:
point-in-time index membership, execution that cannot reference future prices, and
multiple-testing correction applied to every reported result.

On S&P 500 constituents from 2021 to 2024, the best of 54 tested configurations
achieved a **1.01 Sharpe ratio** and passed conventional significance testing at
**0.978**. After deflating for the number of configurations searched, it scored
**0.727** and **0.384**, below the 95% threshold under both variance estimators. The
reported conclusion is that the strategy is not distinguishable from data mining.
See [RESULTS.md](RESULTS.md).

---

## Why

Most backtests fail in one of three ways, and none of them produce an error:

| Failure | Effect | Mitigation here |
|---|---|---|
| Look-ahead bias | Acts on information unavailable at decision time | Orders queue on submission and execute at the next open; asserted over every fill |
| Survivorship bias | Tests only companies that still exist today | Point-in-time membership from dated snapshots, including removed constituents |
| Multiple testing | Reports the best of many attempts as if it were the only one | Deflated Sharpe ratio against the expected maximum under the null |

Each is enforced structurally rather than by convention, and each has a corresponding
check in the verification suite.

---

## Installation

```bash
git clone https://github.com/<user>/leakproof-backtester.git
cd leakproof-backtester
pip install -r requirements.txt
python3 check_install.py
```

Requires Python 3.10 or later. `check_install.py` reports which backends are active.

Optional dependencies enable the preferred backends and the web interface:

```bash
pip install -r requirements-optional.txt
```

Without them the framework uses the stdlib `sqlite3` storage backend and the scipy
optimiser. Both paths are covered by CI.

`lxml` is required only to parse index constituent pages from Wikipedia. The default
membership source uses dated snapshots and needs no HTML parser.

---

## Quick start

Run a backtest on generated data, with no market data required:

```bash
python3 run_skeleton.py
```

Build a store and run the full analysis:

```bash
export ALPACA_API_KEY_ID=...
export ALPACA_API_SECRET_KEY=...

python3 build_sp500.py --store store_sp500 --provider alpaca \
    --start 2021-01-01 --end 2024-12-31

python3 run_real_analysis.py --store store_sp500 \
    --start 2021-01-01 --end 2024-12-31 --n-long 20
```

Launch the web interface:

```bash
streamlit run app.py
```

---

## Usage

### Running a backtest

```python
from datetime import date
from backtester import BacktestConfig, run_backtest

cfg = BacktestConfig(
    start=date(2021, 1, 1),
    end=date(2024, 12, 31),
    factors="momentum,lowvol",
    n_long=20,
    allocator="min_variance",
    commission_bps=5.0,
    slippage_bps=5.0,
    data_source="store",
    store_path="store_sp500",
)

result = run_backtest(cfg)
print(result.sharpe, result.n_trades)
```

### Sweeping parameters and correcting for the search

```python
from backtester import analyze_sweep, build_grid, run_sweep

grid = build_grid(
    cfg,
    factors=["momentum", "lowvol", "momentum,lowvol"],
    lookback_months=[6, 12, 18],
    n_long=[5, 10, 20],
)

sweep = run_sweep(grid)              # distributed across worker processes
report = analyze_sweep(sweep)
print(report.summary())
```

`run_backtest` is a pure function of its configuration, so sweeps are reproducible
and serial and parallel execution produce identical results.

---

## Design

Three layers, connected by structural interfaces defined in
`backtester/protocols.py`.

**Data.** Providers emit a common CSV schema, so ingestion is provider independent.
Bars are validated, then written to Parquet and queried through DuckDB, with a
`sqlite3` backend for environments without it. Index membership is stored as
`(symbol, start_date, end_date)` intervals derived from dated constituent snapshots.

**Engine.** A sequential event loop processes one market event per trading day in a
fixed order: execute orders queued on the previous event, mark to market, then
generate new orders. Executing before signal generation is what prevents an order
from filling at a price its own decision observed.

**Research.** Parameter sweeps distribute independent backtests across processes.
Workers return summary statistics rather than equity curves, bounding inter-process
transfer; the winning configuration's curve is recomputed on demand.

Concurrency lives in the research layer and in data ingestion, both of which
parallelise cleanly. The engine itself stays sequential, since deterministic ordering
is what makes its results verifiable.

### Event flow

```
MarketEvent  -> Strategy   emits -> SignalEvent
SignalEvent  -> Allocator  emits -> OrderEvent
OrderEvent   -> Broker     emits -> FillEvent
FillEvent    -> Portfolio  updates state
```

`MarketEvent` and `SignalEvent` are cross-sectional, each carrying the full universe
for one date, because ranking strategies cannot act until every security for that
date is available.

---

## Components

### Factors

`momentum`, `lowvol`, and `reversal`, combined by weighted average. Each factor
scores the cross-section, scores are standardised within each date and winsorised at
three standard deviations, then blended. Standardisation places returns and
volatilities on a comparable scale; every factor follows a higher-is-better sign
convention so that blending does not cancel signal.

Factors declare a warmup length, and composites renormalise over whichever factors
have sufficient history rather than treating an unavailable factor as neutral.

### Portfolio construction

Long-only, fully invested quadratic programs:

```
minimum variance        minimise w'Sw   subject to sum(w) = 1, 0 <= w <= w_max
maximum decorrelation   minimise w'Cw   subject to sum(w) = 1, 0 <= w <= w_max
```

Covariance is estimated with Ledoit-Wolf shrinkage by default. Inverting a sample
covariance amplifies estimation error into unstable weights: in a bootstrap over a
60-day window, shrinkage reduced weight dispersion by 50%, with the shrinkage
intensity adapting from 0.68 at T=60 to 0.22 at T=252.

### Deflated Sharpe ratio

Implements Bailey and Lopez de Prado (2014). The probabilistic Sharpe ratio corrects
for sample length, skewness, and excess kurtosis. The deflated Sharpe ratio sets the
benchmark to the expected maximum Sharpe ratio under the null across N trials, since
the maximum of N noisy estimates is positive even when no strategy has an edge.

The variance of trial Sharpe ratios is estimated both empirically and analytically
and both are reported, because overlapping configurations make the effective number
of independent trials lower than the nominal count. The closed-form expected maximum
agrees with Monte Carlo simulation to within 2%.

### Data quality

Real price data contains defects that produce plausible but incorrect results:
securities with too few observations to support a statistic, stale series that
measure near-zero volatility and rank highly in low-volatility screens, and
unadjusted corporate actions appearing as extreme single-day returns. Screening runs
at the data layer and records a reason per exclusion.

`audit_store.py --coverage` additionally reports universe size by year and where each
security's history begins, which exposes provider history limits that would otherwise
leave early years populated by an unrepresentative subset of the universe.

---

## Verification

```bash
python3 verify_no_lookahead.py   # fills never precede their orders, per allocator
python3 verify_factors.py        # standardisation, sign convention, blending, warmup
python3 verify_delisting.py      # delisted holdings are liquidated, not retained
python3 verify_deflation.py      # closed form matches Monte Carlo; null is rejected
python3 verify_store.py          # membership reconstruction, gating, store round trip
python3 verify_app.py            # interface smoke test
```

All six run in CI on Python 3.10 through 3.12, against both the default backends and
the optional DuckDB and cvxpy paths. Linting uses ruff:

```bash
pip install ruff
ruff check .
```

These assert invariants rather than compare against recorded output, so they remain
valid across refactors. `verify_deflation.py` includes a null control: on zero-drift
data where no strategy has an edge, the best of N configurations must be rejected
despite a positive raw Sharpe ratio.

---

## Command reference

| Command | Purpose |
|---|---|
| `build_store.py` | Build a store from fixture or Stooq data |
| `build_sp500.py` | Build an S&P 500 store with point-in-time membership |
| `run_real_analysis.py` | Factor comparison, sweep, and deflated Sharpe report |
| `audit_store.py` | Data-quality and coverage diagnostics |
| `report_selections.py` | Securities a strategy actually holds, with quality flags |
| `run_skeleton.py` | Single backtest on generated data |
| `run_sweep_demo.py` | Parameter sweep and result distribution |
| `run_deflation_demo.py` | Deflation across signal and null regimes |
| `run_optimizer_demo.py` | Allocator comparison and covariance noise study |
| `check_install.py` | Report module versions and active backends |

---

## Data sources

| Provider | Credentials | Notes |
|---|---|---|
| Alpaca | `ALPACA_API_KEY_ID`, `ALPACA_API_SECRET_KEY` | Batches ~100 symbols per request. Free tier serves the IEX feed, which begins around 2020 |
| Tiingo | `TIINGO_API_KEY` | Deeper history and better delisted coverage. Free tier allows 50 requests per hour |
| Stooq | none | Keyless, but frequently blocks automated access |

Index membership is derived from the [fja05680/sp500](https://github.com/fja05680/sp500)
dataset, which publishes dated constituent snapshots from 1996 onward.

Providers differ in coverage, so combining them is often necessary. The build reports
**former-member coverage**, the share of past constituents successfully retrieved,
which bounds residual survivorship bias. The reference dataset reached 96%.

Ingestion is resumable: raw files are cached and skipped on subsequent runs, and a
failed run will not overwrite an existing store.

---

## Repository layout

```
backtester/
  events.py           event types
  protocols.py        component interfaces
  config.py           configuration and result types
  engine.py           event loop and run_backtest
  strategy.py         multi-factor composite signals
  factors.py          factor definitions and standardisation
  allocation.py       equal-weight and minimum-variance allocators
  optimize.py         convex optimisation (cvxpy, scipy fallback)
  risk_model.py       covariance estimation
  risk.py             pre-trade risk controls
  execution.py        simulated broker
  portfolio.py        position and cash ledger
  history.py          rolling returns for covariance
  sweep.py            parameter sweeps
  deflated_sharpe.py  probabilistic and deflated Sharpe ratios
  quality.py          data-quality screening
  data.py             synthetic data handler
  store_data.py       store-backed data handler
  pipeline/
    sources.py        market data providers
    clean.py          validation and normalisation
    store.py          Parquet/DuckDB and sqlite backends
    universe.py       point-in-time index membership
webapp/services.py    interface logic, independent of Streamlit
app.py                Streamlit interface
```

---

## Results

Full findings, methodology, and data limitations are in [RESULTS.md](RESULTS.md).

---

## Limitations

- Results cover 2021 to 2024. Earlier periods were excluded because the price feed
  did not cover enough of the universe to make cross-sectional ranking meaningful.
- T = 1004 trading days is a short sample, and deflation grows stricter as T falls.
- Prices are split and dividend adjusted. A live system requires raw prices and a
  separate adjustment table, since orders cannot execute at an adjusted price.
- Value and quality factors are not implemented; they require point-in-time
  fundamentals with correct reporting lags.

---

## License

MIT. See [LICENSE](LICENSE).
