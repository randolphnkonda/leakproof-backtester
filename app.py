"""Streamlit interface for the backtesting framework.

Run with:
    streamlit run app.py

Presentation only; all computation is in webapp/services.py. Tabs cover single
backtests, factor comparison, parameter sweeps with deflation, portfolio
construction, and store inspection.
"""
from __future__ import annotations

from datetime import date

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from webapp import services as S

# Non-interactive backend for headless rendering.
matplotlib.use("Agg")

st.set_page_config(page_title="Leak-Proof Backtester", layout="wide")


# Cached service wrappers.
@st.cache_data(show_spinner=False)
def c_single(*a):
    return S.single_backtest(*a)


@st.cache_data(show_spinner=False)
def c_sweep(*a):
    return S.sweep_and_deflate(*a)


@st.cache_data(show_spinner=False)
def c_alloc(*a):
    return S.allocator_comparison(*a)


@st.cache_data(show_spinner=False)
def c_weights(*a):
    return S.weight_stability(*a)


@st.cache_data(show_spinner=False)
def c_factors(*a):
    return S.factor_comparison(*a)


@st.cache_data(show_spinner=False)
def c_summary(p):
    return S.store_summary(p)


@st.cache_data(show_spinner=False)
def c_universe(p, d):
    return S.store_universe(p, d)


@st.cache_data(show_spinner=False)
def c_prices(p, syms, s, e):
    return S.store_prices(p, list(syms), s, e)


def sig_badge(p: float) -> str:
    return "✅ significant at 95%" if p >= 0.95 else "❌ not significant at 95%"


# Shared configuration.
with st.sidebar:
    st.header("Configuration")
    data_choice = st.selectbox("Data source", S.DATA_CHOICES, index=0)
    store_path = st.text_input("Store path", value="store")
    start = st.date_input("Start", value=date(2015, 1, 1))
    end = st.date_input("End", value=date(2023, 12, 31))
    st.divider()
    st.caption("Factors (blended as cross-sectional z-scores)")
    picked = st.multiselect("Factors", S.FACTORS, default=["momentum"])
    if not picked:
        picked = ["momentum"]
    factors = ",".join(picked)
    lookback = st.slider("Lookback (months)", 3, 24, 12)
    skip = st.slider("Skip (months)", 0, 3, 1)
    n_long = st.slider("Names held", 2, 10, 5)
    st.divider()
    st.caption("Portfolio construction")
    allocator = st.selectbox("Allocator", S.ALLOCATORS, index=1)
    cov_method = st.selectbox("Covariance", S.COV_METHODS, index=0)
    cov_lookback = st.slider("Covariance window (days)", 40, 504, 252, step=20)
    seed = st.number_input("Data seed", value=7, step=1)
    st.divider()
    st.caption(f"Optimiser backend: **{S.optimiser_backend()}**")
    st.caption(f"Store module: `{S.store_version()}`")

cfg = (data_choice, store_path, start, end, lookback, skip, n_long,
       allocator, cov_method, cov_lookback, int(seed), factors)

st.title("Leak-Proof Multi-Factor Backtesting System")
st.markdown(
    "Event-driven backtesting with point-in-time index membership, parallel parameter "
    "sweeps, deflated Sharpe ratios correcting for multiple testing, and portfolio "
    "construction under covariance estimation noise. Select a data source in the "
    "sidebar."
)

tab_bt, tab_factors, tab_sweep, tab_port, tab_data = st.tabs(
    ["Backtest", "Factors", "Sweep & Deflated Sharpe", "Portfolio & Covariance",
     "Data Store"])

# Offer to build a fixture store when the selected store does not yet exist.
if data_choice.startswith("store"):
    if not S.store_exists(store_path):
        st.warning(f"No data store at '{store_path}' yet.")
        st.caption("Build a fixture store, or run build_sp500.py for market data.")
        if st.button("Build fixture store", type="primary"):
            with st.spinner("Building store..."):
                backend = S.build_fixture_store(store_path, start, end)
            st.success(f"Built store using {backend}.")
            st.rerun()
        st.stop()
    try:
        _summ = c_summary(store_path)
        if _summ["n_symbols"] == 0:
            st.error(f"The store at '{store_path}' has no data. "
                     f"Rebuild it: `python3 build_store.py --source fixture --store {store_path}`.")
            st.stop()
    except Exception as e:
        st.error(f"Could not open the store at '{store_path}': {e}")
        st.stop()


# Single backtest.
with tab_bt:
    st.subheader("Single backtest")
    res = c_single(*cfg)
    m = res["metrics"]
    cols = st.columns(5)
    cols[0].metric("Sharpe", f"{m['sharpe']:.2f}")
    cols[1].metric("Ann. return", f"{m['ann_return']:.1%}")
    cols[2].metric("Ann. vol", f"{m['ann_vol']:.1%}")
    cols[3].metric("Max drawdown", f"{m['max_drawdown']:.1%}")
    cols[4].metric("Fills", f"{res['n_trades']}")
    st.line_chart(res["equity"].rename("equity"))
    if data_choice.startswith("synthetic"):
        st.caption("Synthetic data. Select a store for market prices.")


# Factor comparison.
with tab_factors:
    st.subheader("Factor comparison")
    st.markdown(
        "Each factor scores the cross-section, is standardised within each date, then "
        "blended by weight. Standardisation places returns and volatilities on a "
        "common scale. Higher scores rank more attractive for every factor."
    )
    default_specs = ["momentum", "lowvol", "reversal", "momentum,lowvol"]
    specs_cmp = st.multiselect("Specs to compare", default_specs + ["momentum,lowvol,reversal"],
                               default=default_specs, key="factor_cmp")
    if specs_cmp:
        fdf = c_factors(data_choice, store_path, start, end, lookback, skip, n_long,
                        allocator, cov_method, cov_lookback, int(seed), tuple(specs_cmp))
        st.dataframe(
            fdf.style.format({"sharpe": "{:.2f}", "ann_return": "{:.1%}",
                              "ann_vol": "{:.1%}", "max_drawdown": "{:.1%}"}),
            width="stretch")
        st.bar_chart(fdf.set_index("factors")["sharpe"])
    st.caption("Reversal approximates the negative of short-window momentum, so "
               "combining the two largely cancels the signal.")


# Parameter sweep and deflation.
with tab_sweep:
    st.subheader("Parameter sweep and the deflated Sharpe")
    st.markdown("The number of configurations evaluated is the trial count the "
                "deflated Sharpe ratio corrects for.")
    c1, c2, c3 = st.columns(3)
    lbs = c1.multiselect("Lookbacks", [6, 9, 12, 15, 18], default=[6, 12, 18])
    sks = c2.multiselect("Skips", [0, 1, 2], default=[0, 1])
    nls = c3.multiselect("Names held", [3, 4, 5, 6, 7], default=[3, 5, 7])
    specs = st.multiselect(
        "Factor combinations to search",
        ["momentum", "lowvol", "reversal", "momentum,lowvol",
         "momentum,reversal", "lowvol,reversal", "momentum,lowvol,reversal"],
        default=["momentum", "lowvol", "momentum,lowvol"])
    if not specs:
        specs = ["momentum"]

    if st.button("Run sweep", type="primary"):
        with st.spinner("Running sweep..."):
            st.session_state.sweep = c_sweep(
                data_choice, store_path, start, end,
                tuple(lbs), tuple(sks), tuple(nls),
                allocator, cov_method, cov_lookback, int(seed), tuple(specs))

    sw = st.session_state.get("sweep")
    if sw is None:
        st.info("Choose the grid and click Run sweep.")
    else:
        rep = sw["report"]
        c = st.columns(4)
        c[0].metric("Trials (N)", sw["n_trials"])
        c[1].metric("Best Sharpe", f"{rep.best_sharpe_ann:.2f}")
        c[2].metric("PSR vs 0 (naive)", f"{rep.psr_vs_zero:.3f}")
        c[3].metric("Sample length T", rep.T)
        st.markdown(f"**Deflated Sharpe (empirical variance):** {rep.dsr_emp:.3f} "
                    f"- {sig_badge(rep.dsr_emp)}")
        st.markdown(f"**Deflated Sharpe (analytic variance):** {rep.dsr_ana:.3f} "
                    f"- {sig_badge(rep.dsr_ana)}")
        st.caption("The empirical estimator uses the observed dispersion of trials; "
                   "the analytic estimator treats trials as independent. The "
                   "appropriate value lies between them.")

        fig, ax = plt.subplots(figsize=(8, 3.2))
        s = sw["sharpes"][np.isfinite(sw["sharpes"])]
        ax.hist(s, bins=18, edgecolor="white")
        ax.axvline(np.nanmax(sw["sharpes"]), color="crimson", linestyle="--",
                   label=f"best {np.nanmax(sw['sharpes']):.2f}")
        ax.set_xlabel("Annualised Sharpe")
        ax.set_ylabel("Configs")
        ax.legend()
        st.pyplot(fig)

        with st.expander("All trials"):
            st.dataframe(sw["table"], width="stretch")


# Portfolio construction.
with tab_port:
    st.subheader("Allocator comparison")
    comp = c_alloc(data_choice, store_path, start, end, lookback, skip, n_long,
                   cov_method, cov_lookback, int(seed), factors)
    st.dataframe(
        comp.style.format({"sharpe": "{:.2f}", "ann_return": "{:.1%}",
                           "ann_vol": "{:.1%}", "max_drawdown": "{:.1%}"}),
        width="stretch")

    st.divider()
    st.subheader("Covariance noise and weight stability")
    st.markdown("Inverting a noisy covariance matrix amplifies estimation error into "
                "unstable weights. Ledoit-Wolf shrinkage reduces this.")
    w1, w2 = st.columns(2)
    window = w1.slider("Covariance window (days)", 40, 252, 60, step=20)
    B = w2.slider("Bootstrap resamples", 50, 500, 200, step=50)
    if st.button("Run stability study", type="primary"):
        with st.spinner("Bootstrapping weights..."):
            st.session_state.ws = c_weights(window, int(B), int(seed))

    ws = st.session_state.get("ws")
    if ws is None:
        st.info("Click Run stability study.")
    else:
        a, b, c = st.columns(3)
        a.metric("Shrinkage intensity", f"{ws['shrinkage']:.2f}")
        b.metric("Sample weight noise", f"{ws['instability_sample']:.4f}")
        c.metric("Ledoit-Wolf weight noise", f"{ws['instability_lw']:.4f}",
                 delta=f"-{ws['reduction']:.0%}")
        fig, ax = plt.subplots(figsize=(9, 3.4))
        x = np.arange(ws["k"])
        ax.bar(x - 0.2, ws["sample_mean"], 0.4, yerr=ws["sample_std"], capsize=3,
               label="sample", color="#D85A30")
        ax.bar(x + 0.2, ws["lw_mean"], 0.4, yerr=ws["lw_std"], capsize=3,
               label="Ledoit-Wolf", color="#1D9E75")
        ax.set_xticks(x)
        ax.set_xticklabels(ws["symbols"], rotation=45, ha="right")
        ax.set_ylabel("min-variance weight")
        ax.legend()
        st.pyplot(fig)


# Store inspection.
with tab_data:
    st.subheader("Data store")
    if not S.store_exists(store_path):
        st.info(f"No store at '{store_path}' yet.")
        if st.button("Build fixture store", key="build_data_tab"):
            with st.spinner("Building store..."):
                backend = S.build_fixture_store(store_path, start, end)
            st.success(f"Built store using {backend}.")
            st.rerun()
        summ = None
    else:
        summ = c_summary(store_path)
        a, b, c = st.columns(3)
        a.metric("Backend", summ["backend"])
        b.metric("Symbols", summ["n_symbols"])
        c.metric("Membership intervals", summ["n_intervals"])

        st.markdown("**Point-in-time universe**")
        as_of = st.date_input("As of", value=date(2016, 1, 1), key="pit_date")
        uni = c_universe(store_path, as_of)
        st.write(f"{len(uni)} members on {as_of}: {', '.join(uni)}")

        st.markdown("**Prices**")
        picks = st.multiselect("Symbols", summ["symbols"],
                               default=summ["symbols"][:3])
        if picks:
            px = c_prices(store_path, tuple(picks), start, end)
            if not px.empty:
                st.line_chart(px)

        with st.expander("Membership intervals"):
            st.dataframe(summ["membership"], width="stretch")
