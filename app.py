"""
Streamlit dashboard for the stock-ranking decision-support prototype.

Run:
    python -m streamlit run app.py --server.fileWatcherType none

The dashboard reads local prototype outputs. If they do not exist, it runs the
standalone forward engine first.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE_SCRIPT = os.path.join(BASE_DIR, "forward_engine.py")
OUTPUT_DIR = os.path.join(BASE_DIR, "dashboard_outputs")
FORWARD_DIR = os.path.join(OUTPUT_DIR, "forward_paper_trading")
DEFAULT_SYSTEMS = ["ensemble_equal_rank_default", "ensemble_equal_rank_conservative"]
SPY_BENCHMARK = "SPY_benchmark"
INITIAL_CAPITAL = 100_000.0
ACTION_ORDER = {"SELL": 0, "REDUCE": 1, "BUY": 2, "ADD": 3, "HOLD": 4}
REQUIRED_OUTPUT_FILES = [
    os.path.join(OUTPUT_DIR, "portfolio_status_summary.csv"),
    os.path.join(OUTPUT_DIR, "latest_decisions.csv"),
    os.path.join(OUTPUT_DIR, "latest_target_weights.csv"),
    os.path.join(OUTPUT_DIR, "combined_forward_daily_nav.csv"),
    os.path.join(OUTPUT_DIR, "latest_market_state.csv"),
]


st.set_page_config(
    page_title="Thesis Decision Support",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(ttl=5)
def read_csv(path: str, parse_dates: tuple[str, ...] = ()):
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=list(parse_dates) if parse_dates else None)


@st.cache_data(ttl=5)
def read_json(path: str):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def portfolio_path(portfolio_name: str):
    return os.path.join(FORWARD_DIR, portfolio_name)


def outputs_ready():
    return all(os.path.exists(path) for path in REQUIRED_OUTPUT_FILES)


def fmt_pct(value):
    if pd.isna(value):
        return "-"
    return f"{float(value):.2%}"


def fmt_money(value):
    if pd.isna(value):
        return "-"
    return f"${float(value):,.0f}"


def load_status():
    return read_csv(os.path.join(OUTPUT_DIR, "portfolio_status_summary.csv"))


def load_latest_decisions():
    return read_csv(os.path.join(OUTPUT_DIR, "latest_decisions.csv"))


def load_latest_targets():
    return read_csv(os.path.join(OUTPUT_DIR, "latest_target_weights.csv"))


def load_nav():
    return read_csv(os.path.join(OUTPUT_DIR, "combined_forward_daily_nav.csv"), parse_dates=("date",))


def load_market_state():
    return read_csv(os.path.join(OUTPUT_DIR, "latest_market_state.csv"))


def load_account(portfolio_name):
    return read_json(os.path.join(portfolio_path(portfolio_name), "account_state.json"))


def load_positions(portfolio_name):
    path = os.path.join(portfolio_path(portfolio_name), "daily_positions.csv")
    positions = read_csv(path, parse_dates=("date",))
    if positions.empty:
        return positions
    latest_date = positions["date"].max()
    return positions[positions["date"].eq(latest_date)].copy()


def load_trade_ledger(portfolio_name):
    return read_csv(os.path.join(portfolio_path(portfolio_name), "trade_ledger.csv"))


def latest_for_portfolio(frame, portfolio_name):
    if frame.empty or "portfolio_name" not in frame.columns:
        return pd.DataFrame()
    return frame[frame["portfolio_name"].eq(portfolio_name)].copy()


def latest_spy_benchmark(status, nav):
    if not status.empty and "portfolio_name" in status.columns:
        rows = status[status["portfolio_name"].eq(SPY_BENCHMARK)]
        if not rows.empty:
            return rows.iloc[-1]
    if nav.empty or "portfolio_name" not in nav.columns:
        return pd.Series(dtype=object)
    rows = nav[nav["portfolio_name"].eq(SPY_BENCHMARK)].sort_values("date")
    if rows.empty:
        return pd.Series(dtype=object)
    latest = rows.iloc[-1].copy()
    latest["cumulative_return"] = latest.get("portfolio_value", np.nan) / INITIAL_CAPITAL - 1.0
    return latest


def run_update(force_download=False):
    command = [sys.executable, ENGINE_SCRIPT]
    if force_download:
        command.append("--force-download")
    return subprocess.run(
        command,
        cwd=BASE_DIR,
        text=True,
        capture_output=True,
        timeout=1800,
    )


def ensure_outputs_available():
    if outputs_ready():
        return
    if st.session_state.get("initial_generation_attempted", False):
        st.warning("Dashboard outputs are still missing. Run an update from the sidebar to try again.")
        return

    st.session_state.initial_generation_attempted = True
    with st.spinner("Generating dashboard data for the first run. This may take a few minutes."):
        result = run_update(force_download=False)
    if result.returncode != 0:
        st.error("The dashboard data could not be generated automatically.")
        st.code((result.stdout or "")[-4000:] + "\n" + (result.stderr or "")[-4000:])
        st.stop()
    st.cache_data.clear()


def new_york_run_hint():
    now_ny = datetime.now(ZoneInfo("America/New_York"))
    close_ready = now_ny.weekday() < 5 and (now_ny.hour, now_ny.minute) >= (16, 15)
    weekend = now_ny.weekday() >= 5
    label = now_ny.strftime("%Y-%m-%d %H:%M %Z")
    if weekend:
        return "info", f"New York time: {label}. Market is closed; latest completed session should be available if data provider has updated."
    if close_ready:
        return "success", f"New York time: {label}. Good time to run: US market should be closed and daily data should be complete."
    return "warning", f"New York time: {label}. Best to run after 16:15 New York time, when US daily bars are complete."


def signal_execution_panel(row, decisions):
    signal_date = row.get("signal_date", "") if row is not None and not row.empty else ""
    execution_date = row.get("execution_date", "") if row is not None and not row.empty else ""
    if decisions is not None and not decisions.empty:
        signal_date = decisions["data_as_of"].iloc[0] if "data_as_of" in decisions else signal_date
        execution_date = decisions["execution_date"].iloc[0] if "execution_date" in decisions else execution_date
    col1, col2, col3 = st.columns(3)
    col1.metric("Signal Date", signal_date or "-")
    col2.metric("Execution Date", execution_date or "-")
    col3.metric("Selected System", row.get("portfolio_name", "-") if row is not None and not row.empty else "-")


def action_banner(row, decisions):
    if row is None or row.empty:
        st.warning("No portfolio status is available yet. Run today's update first.")
        return

    status = str(row.get("execution_status", ""))
    pending = int(row.get("pending_decisions", 0) or 0)
    reason = row.get("pending_reason", "")
    actions = decisions[decisions["trade_instruction"].isin(["BUY", "ADD", "REDUCE", "SELL"])]
    decision_type = decisions["decision"].iloc[0] if not decisions.empty and "decision" in decisions else ""

    if status in {"executed", "already_processed"}:
        st.success(f"Action: executed. {len(actions)} trade rows were generated for this system.")
    elif status == "hold_pending_execution":
        st.warning(f"Action: HOLD today, but {pending} previous executable order is still pending. {reason}")
    elif status.startswith("pending"):
        st.warning(f"Action: pending execution. {reason}")
    elif status == "not_executable_hold" or decision_type == "HOLD":
        st.info("Action: HOLD. No new executable stock order today.")
    else:
        st.info(f"Action status: {status or 'unknown'}")


def metric_cards(row, market_row, spy_row):
    portfolio_value = float(row.get("portfolio_value", np.nan))
    spy_value = float(spy_row.get("portfolio_value", np.nan))
    value_vs_spy = portfolio_value - spy_value if np.isfinite(portfolio_value) and np.isfinite(spy_value) else np.nan
    excess_return = row.get("cumulative_return", np.nan) - spy_row.get("cumulative_return", np.nan)

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Portfolio Value", fmt_money(portfolio_value))
    col2.metric("SPY Benchmark Value", fmt_money(spy_value))
    col3.metric("Value vs SPY", fmt_money(value_vs_spy))
    col4.metric("Cumulative Return", fmt_pct(row.get("cumulative_return", np.nan)), delta=f"vs SPY {fmt_pct(excess_return)}")
    col5.metric("Cash Weight", fmt_pct(row.get("cash_weight", np.nan)))
    col6.metric("Holdings", int(row.get("n_holdings", 0) or 0))

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Execution Status", str(row.get("execution_status", "-")))
    col2.metric("Market State", str(market_row.get("market_state", "-")))
    col3.metric("Risk Score", str(market_row.get("risk_score", "-")))
    col4.metric("VIX", f"{float(market_row.get('vix', np.nan)):.2f}" if pd.notna(market_row.get("vix", np.nan)) else "-")
    col5.metric("SPY 21d Ret.", fmt_pct(market_row.get("spy_ret21", np.nan)))


def market_state_panel(market):
    if market.empty:
        st.info("No market state file found.")
        return
    row = market.iloc[0]
    display = pd.DataFrame(
        [
            ["Date", row.get("date", "")],
            ["Market State", row.get("market_state", "")],
            ["Risk Score", row.get("risk_score", "")],
            ["SPY Close", f"{float(row.get('spy_close', np.nan)):,.2f}"],
            ["SPY MA20", f"{float(row.get('spy_ma20', np.nan)):,.2f}"],
            ["SPY MA200", f"{float(row.get('spy_ma200', np.nan)):,.2f}"],
            ["SPY 21d Return", fmt_pct(row.get("spy_ret21", np.nan))],
            ["SPY 21d Ann. Vol", fmt_pct(row.get("spy_vol21_ann", np.nan))],
            ["VIX", f"{float(row.get('vix', np.nan)):.2f}"],
        ],
        columns=["Metric", "Value"],
    )
    display["Value"] = display["Value"].astype(str)
    st.dataframe(display, hide_index=True, use_container_width=True)


def actions_table(decisions):
    if decisions.empty:
        st.info("No latest decision rows for this system.")
        return
    actions = decisions[decisions["trade_instruction"].isin(["BUY", "ADD", "REDUCE", "SELL"])].copy()
    if actions.empty:
        st.info("No BUY / ADD / REDUCE / SELL rows today.")
        with st.expander("Show HOLD rows"):
            hold_cols = ["ticker", "sector", "previous_weight", "target_weight", "trade_instruction", "trigger_reason"]
            hold_display = decisions[[c for c in hold_cols if c in decisions.columns]].copy()
            for c in ["previous_weight", "target_weight"]:
                if c in hold_display:
                    hold_display[c] = hold_display[c].map(fmt_pct)
            st.dataframe(hold_display, hide_index=True, use_container_width=True)
        return
    actions["_action_order"] = actions["trade_instruction"].map(ACTION_ORDER).fillna(99)
    actions.sort_values(["_action_order", "ticker"], inplace=True)
    cols = [
        "ticker",
        "sector",
        "trade_instruction",
        "previous_weight",
        "target_weight",
        "score",
        "trigger_reason",
        "partial_update_note",
    ]
    display = actions[[c for c in cols if c in actions.columns]].copy()
    for c in ["previous_weight", "target_weight"]:
        if c in display:
            display[c] = display[c].map(fmt_pct)
    st.dataframe(display, hide_index=True, use_container_width=True)


def target_table(targets):
    if targets.empty:
        st.info("No target weights for this system.")
        return
    cols = ["ticker", "sector", "target_weight", "base_stock_weight", "score", "trigger_reason"]
    display = targets[[c for c in cols if c in targets.columns]].copy()
    for c in ["target_weight", "base_stock_weight"]:
        if c in display:
            display[c] = display[c].map(fmt_pct)
    st.dataframe(display.sort_values("target_weight", ascending=False), hide_index=True, use_container_width=True)


def sector_exposure_table(targets):
    if targets.empty or "sector" not in targets.columns or "target_weight" not in targets.columns:
        st.info("No sector exposure available.")
        return
    exposure = (
        targets.groupby("sector", as_index=False)
        .agg(target_weight=("target_weight", "sum"), n_stocks=("ticker", "nunique"))
        .sort_values("target_weight", ascending=False)
    )
    exposure["sector_cap_reference"] = "30%"
    display = exposure.copy()
    display["target_weight"] = display["target_weight"].map(fmt_pct)
    st.dataframe(display, hide_index=True, use_container_width=True)


def holdings_table(positions):
    if positions.empty:
        st.info("No current holdings yet. This usually means the first executable rebalance is still pending.")
        return
    display = positions.copy()
    for col in ["actual_weight"]:
        if col in display:
            display[col] = display[col].map(fmt_pct)
    for col in ["market_value"]:
        if col in display:
            display[col] = display[col].map(fmt_money)
    keep = ["date", "ticker", "shares", "close_price", "market_value", "actual_weight"]
    st.dataframe(display[[c for c in keep if c in display.columns]], hide_index=True, use_container_width=True)


def nav_chart(nav, selected):
    if nav.empty:
        st.info("No NAV history yet.")
        return
    subset = nav[nav["portfolio_name"].isin([selected, SPY_BENCHMARK])].copy()
    if subset.empty:
        st.info("No NAV history for selected system.")
        return
    pivot = subset.pivot_table(index="date", columns="portfolio_name", values="portfolio_value", aggfunc="last").sort_index()
    pivot["cash_reference"] = INITIAL_CAPITAL
    st.line_chart(pivot, use_container_width=True)


def advanced_section(status):
    with st.expander("Advanced: individual signal portfolios", expanded=False):
        if status.empty:
            st.info("No portfolio status available.")
            return
        display = status.copy()
        for c in ["portfolio_value"]:
            if c in display:
                display[c] = display[c].map(fmt_money)
        for c in ["cumulative_return", "cash_weight"]:
            if c in display:
                display[c] = display[c].map(fmt_pct)
        keep = [
            "portfolio_name",
            "signal_name",
            "system_mode",
            "execution_status",
            "portfolio_value",
            "cumulative_return",
            "cash_weight",
            "n_holdings",
            "pending_decisions",
            "pending_reason",
        ]
        st.dataframe(display[[c for c in keep if c in display.columns]], hide_index=True, use_container_width=True)


def main():
    st.title("Decision-Support Prototype")
    st.caption("Forward paper-trading dashboard")

    ensure_outputs_available()

    status = load_status()
    decisions = load_latest_decisions()
    targets = load_latest_targets()
    nav = load_nav()
    market = load_market_state()

    available = status["portfolio_name"].tolist() if not status.empty else DEFAULT_SYSTEMS
    main_options = [x for x in DEFAULT_SYSTEMS if x in available] or available

    with st.sidebar:
        st.header("Controls")
        hint_level, hint_text = new_york_run_hint()
        if hint_level == "success":
            st.success(hint_text)
        elif hint_level == "warning":
            st.warning(hint_text)
        else:
            st.info(hint_text)

        selected = None
        if main_options:
            selected = st.selectbox("Select system", main_options, index=0)
        else:
            st.warning("No portfolio outputs found yet.")
        show_advanced_selector = st.checkbox("Show individual signal selector", value=False)
        if show_advanced_selector and available:
            selected = st.selectbox("Advanced portfolio", available, index=available.index(selected) if selected in available else 0)

        force_download = st.checkbox("Force data refresh (!! Reset portfolio value to $100,000. All records will be removed !!)", value=False)
        if "update_running" not in st.session_state:
            st.session_state.update_running = False
        run_clicked = st.button(
            "Run today's update",
            type="primary",
            disabled=st.session_state.update_running,
        )
        if run_clicked and not st.session_state.update_running:
            st.session_state.update_running = True
            try:
                with st.spinner("Running the forward prototype update. This may take a few minutes."):
                    result = run_update(force_download=force_download)
            except Exception as exc:
                st.session_state.update_running = False
                st.error("Update failed before completion.")
                st.exception(exc)
                result = None
            finally:
                st.session_state.update_running = False
            if result is None:
                st.stop()
            if result.returncode == 0:
                st.success("Update completed.")
                st.cache_data.clear()
                st.rerun()
            else:
                st.error("Update failed.")
                st.code(result.stdout[-4000:] + "\n" + result.stderr[-4000:])

        if st.button("Refresh dashboard"):
            st.cache_data.clear()
            st.rerun()

        st.divider()
        st.caption(f"Last dashboard refresh: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        st.caption(f"Output folder: `{OUTPUT_DIR}`")

    if not selected:
        st.warning("No system is available yet. Run today's update first.")
        return

    selected_status = status[status["portfolio_name"].eq(selected)].iloc[0] if not status.empty and selected in status["portfolio_name"].values else pd.Series(dtype=object)
    selected_decisions = latest_for_portfolio(decisions, selected)
    selected_targets = latest_for_portfolio(targets, selected)
    selected_positions = load_positions(selected)
    market_row = market.iloc[0] if not market.empty else pd.Series(dtype=object)
    spy_status = latest_spy_benchmark(status, nav)

    action_banner(selected_status, selected_decisions)
    signal_execution_panel(selected_status, selected_decisions)
    st.divider()

    st.subheader("1. System Summary")
    metric_cards(selected_status, market_row, spy_status)

    st.subheader("2. Market State")
    market_state_panel(market)

    left, right = st.columns([1, 1])
    with left:
        st.subheader("3. Current Holdings")
        holdings_table(selected_positions)
    with right:
        st.subheader("4. Today's Actions")
        actions_table(selected_decisions)

    st.subheader("5. Target Portfolio")
    target_table(selected_targets)

    st.subheader("6. Sector Exposure")
    sector_exposure_table(selected_targets)

    st.subheader("7. NAV Curve")
    nav_chart(nav, selected)

    st.subheader("8. Advanced")
    advanced_section(status)

    with st.expander("Raw account state", expanded=False):
        st.json(load_account(selected))


if __name__ == "__main__":
    main()
