# Results

Empirical findings from the Leak-Proof Multi-Factor Backtesting System, run on real
S&P 500 data. The headline is not a strategy. It is that a strategy which looks
significant by the usual test is indistinguishable from data mining once the search
is accounted for.

---

## 1. The finding

On S&P 500 constituents from 2021-01-01 to 2024-12-31, using point-in-time index
membership, 10 bps round-trip transaction costs, and 54 tested configurations, the
best strategy achieved an annualised Sharpe ratio of **1.01**. Conventional
significance testing accepts it. The deflated Sharpe ratio does not.

| Test | Value | Verdict at 95% |
|---|---|---|
| Probabilistic Sharpe vs zero (no deflation) | 0.978 | significant |
| Deflated Sharpe, empirical variance | 0.727 | **not significant** |
| Deflated Sharpe, analytic variance | 0.384 | **not significant** |

The reason is visible in the benchmark. Given 54 trials, the *expected maximum*
Sharpe under the null that every strategy is worthless is **0.70 (empirical) to 1.16
(analytic)** annualised. The winning configuration's 1.01 sits inside that band. It
does not beat what luck alone would produce from a search of that size.

Sample: T = 1004 trading days, best-configuration skewness 0.09, excess kurtosis 5.21.

### Factor comparison

Identical settings across all specifications, minimum-variance allocation, 20 names
held, monthly rebalancing.

| Factors | Ann. return | Ann. vol | Sharpe | Max drawdown | Fills |
|---|---|---|---|---|---|
| momentum | 6.8% | 16.3% | 0.42 | -20.8% | 584 |
| lowvol | 0.7% | 11.5% | 0.06 | -20.3% | 577 |
| reversal | 4.6% | 22.0% | 0.21 | -21.6% | 1155 |
| momentum + lowvol | 0.5% | 12.2% | 0.04 | -23.3% | 687 |
| momentum + lowvol + reversal | 0.6% | 18.0% | 0.04 | -29.9% | 1173 |

The low-volatility factor produces the lowest realised volatility of the set, which
is the basic sanity check that it is doing what it claims. Reversal trades roughly
twice as often and pays for it in costs.

### Top configurations by raw Sharpe

| Factors | Lookback | Skip | Names | Sharpe | Max DD |
|---|---|---|---|---|---|
| momentum | 18 | 0 | 10 | 1.007 | -14.1% |
| momentum | 18 | 1 | 10 | 1.004 | -13.2% |
| momentum | 18 | 1 | 15 | 0.959 | -12.8% |
| momentum | 12 | 0 | 15 | 0.923 | -15.2% |
| momentum | 18 | 0 | 5 | 0.905 | -14.1% |

---

## 2. Method

- **Universe.** S&P 500 with point-in-time membership reconstructed from dated
  constituent snapshots going back to 1996, so the backtest holds only names that
  were actually in the index on each date and includes companies later removed.
- **Execution.** Event-driven, one trading day per event. Signals are computed at the
  close of day t and orders fill at the open of day t+1, so a decision provably
  cannot see the price it executes at. Verified across every allocator with zero
  violations over thousands of fills.
- **Costs.** 5 bps commission and 5 bps slippage per side.
- **Signals.** Cross-sectional momentum, low volatility, and short-term reversal. Each
  factor scores the cross-section, scores are z-scored within each date and
  winsorised at three standard deviations, then blended by weight.
- **Allocation.** Long-only minimum-variance quadratic program with Ledoit and Wolf
  covariance shrinkage.
- **Search.** 54 configurations spanning three factor specifications, three lookbacks,
  two skips, and three portfolio sizes. That count is the N in the deflation.

---

## 3. Data coverage and limitations

Stated plainly, because they bound what the result can claim.

- **Window restricted to 2021-2024.** The price feed used for the bulk of the universe
  does not reach before 2020: 611 of 699 symbols have their first bar in that year.
  Any backtest starting in 2016 therefore ran on roughly 30 names, almost all of them
  companies later removed from the index, which is a failure-only universe rather
  than a representative one. The window was cut to the period with genuine coverage:
  642, 635, 615 and 612 symbols in 2021 through 2024.
- **Former-member coverage is 96%** (229 of 239). The missing 10 are mostly ticker
  renames after mergers, whose history exists under the successor symbol. Their
  absence biases results very slightly upward.
- **Short sample.** 1004 trading days and about 48 monthly rebalances. The deflated
  Sharpe grows stricter as the sample shortens, and excess kurtosis of 5.21 indicates
  fat tails, so the estimate carries real uncertainty.
- **The winner sits on a grid boundary.** Four of the top five configurations use an
  18-month lookback, the largest value tested. An optimum at the edge of the search
  space usually means the true optimum lies outside the grid or the result is fitting
  noise. Both arguments weigh against taking the number seriously.
- **Prices are split and dividend adjusted.** A live system needs raw prices plus a
  separate adjustment table, since one cannot transact at an adjusted price.

---

## 4. Errors found and fixed

Every one of these produced plausible-looking output. None announced itself. They are
recorded because finding them is the substance of the work, not an embarrassment.

- **Frozen positions.** The broker only filled orders for symbols with a bar that day,
  so when a holding was delisted its sell order was dropped and the position stayed in
  the portfolio permanently, marked at a stale price with its cash trapped.
  Reproduced with a test showing 748 shares held 943 days after the name stopped
  trading. The broker now liquidates at the last observed price.
- **Stale prices winning the low-volatility screen.** A series repeating the same close
  measures near-zero volatility and therefore wins a naive low-vol ranking while being
  nothing like low risk. Before the fix the low-volatility portfolio realised 18.0%
  volatility against momentum's 16.6%, which is impossible by construction. After
  adding a liquidity floor and a store-level quality filter it realises 11.5% against
  16.3%, and the factor selects the expected defensive names.
- **A hollow early universe.** See coverage above. This was the largest error and it
  invalidated every result before 2021.
- **A false-positive detector of my own making.** An early check flagged 113 symbols as
  "ticker reuse" for having bars after leaving the index. That was wrong: index
  removal is not delisting, and those companies kept trading. It is also harmless,
  because the engine gates trading on point-in-time membership. Verified by assertion
  and reframed as informational.
- **A date-blind cache.** Raw price files were reused whenever they existed, regardless
  of whether they covered the requested window, which would have silently defeated any
  backfill from a deeper-history provider.

---

## 5. Conclusion

The strategy does not survive. Reported without the multiple-testing correction, a
1.01 Sharpe over four years of real large-cap equity data with realistic costs would
look like a result worth trading. It is not. The expected best-of-54 under the null
covers it, and both variance estimates place it well below the 95% threshold.

That is the intended output of this system. The infrastructure exists to make the
honest answer reachable, and to make the dishonest one hard to produce by accident.
