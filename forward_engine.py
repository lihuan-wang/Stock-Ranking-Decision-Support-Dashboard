"""
Forward decision-support engine and paper-trading update.

This script is intentionally standalone. It does not import the old live
framework and does not read previous live logs. It implements the final thesis
prototype directly:

1. Train six candidate ranking signals using 2016-now data.
2. Build an equal-rank ensemble signal.
3. Convert each signal into top20pct sector-capped buffered target weights.
4. Produce default and conservative system recommendations.
5. Maintain independent $100,000 paper-trading accounts for each signal/system.

The script is execution-neutral: it never submits broker orders.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# =============================================================================
# Configuration
# =============================================================================

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = APP_ROOT
EXPERIMENT_NAME = "decision_support_prototype"
OUT_DIR = os.path.join(APP_ROOT, "dashboard_outputs")
FORWARD_DIR = os.path.join(OUT_DIR, "forward_paper_trading")
CACHE_DIR = os.path.join(OUT_DIR, "data_cache")

START_DATE = "2016-01-01"
INITIAL_CAPITAL = 100_000.0
PREDICTION_HORIZON = 5
TARGET_TYPE = "cross_sectional_rank"
SELECTION_ENTRY_PCT = 0.20
SELECTION_EXIT_PCT = 0.30
SECTOR_CAP = 0.30
EXTREME_OPPORTUNITY_PCT = 0.10
EXTREME_DETERIORATION_PCT = 0.50
MAX_EVENT_REPLACEMENTS = 2
EVENT_TURNOVER_CAP = 0.20
TRANSACTION_COST_BPS = 10.0
TRANSACTION_COST_RATE = TRANSACTION_COST_BPS / 10_000.0
MIN_COVERAGE_RATIO = 0.80
RANDOM_SEED = 42
EPS = 1e-9

SECTORS = {
    "InfoTech": ["AAPL", "MSFT", "NVDA", "AVGO", "AMD", "ORCL", "QCOM", "TXN", "AMAT", "MU"],
    "CommServices": ["GOOGL", "META", "NFLX", "TMUS", "DIS", "VZ", "T", "CHTR", "EA", "TTWO"],
    "ConsDiscretary": ["AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "TGT", "SBUX", "BKNG", "RCL"],
    "ConsStaples": ["WMT", "PG", "KO", "PEP", "COST", "CL", "MDLZ", "GIS", "KHC", "HSY"],
    "HealthCare": ["JNJ", "UNH", "LLY", "ABBV", "MRK", "TMO", "ABT", "BMY", "AMGN", "GILD"],
    "Financials": ["JPM", "BAC", "GS", "MS", "V", "MA", "WFC", "C", "AXP", "BLK"],
    "Industrials": ["HON", "GE", "CAT", "UPS", "LMT", "RTX", "DE", "MMM", "FDX", "NOC"],
    "Energy": ["XOM", "CVX", "COP", "SLB", "EOG", "OXY", "PSX", "VLO", "HAL", "MPC"],
    "Materials": ["LIN", "APD", "SHW", "FCX", "NEM", "ECL", "DD", "PPG", "NUE", "CF"],
    "RealEstate": ["AMT", "PLD", "EQIX", "SPG", "PSA", "CCI", "DLR", "O", "VICI", "WY"],
    "Utilities": ["NEE", "DUK", "SO", "D", "EXC", "AEP", "SRE", "PCG", "XEL", "ED"],
}
ALL_STOCKS = [ticker for stocks in SECTORS.values() for ticker in stocks]
TICKER_TO_SECTOR = {ticker: sector for sector, tickers in SECTORS.items() for ticker in tickers}

CANDIDATE_SIGNALS = [
    {"signal_name": "Ridge_baseline", "model": "Ridge", "feature_variant": "baseline"},
    {"signal_name": "Ridge_sector_relative", "model": "Ridge", "feature_variant": "sector_relative"},
    {"signal_name": "RandomForest_baseline", "model": "RandomForest", "feature_variant": "baseline"},
    {"signal_name": "RandomForest_technical_v2", "model": "RandomForest", "feature_variant": "technical_v2"},
    {"signal_name": "XGBoost_baseline", "model": "XGBoost", "feature_variant": "baseline"},
    {"signal_name": "XGBoost_liquidity_volume", "model": "XGBoost", "feature_variant": "liquidity_volume"},
]

FEATURE_BLOCKS = {
    "baseline": ["baseline"],
    "sector_relative": ["baseline", "sector_relative"],
    "technical_v2": ["baseline", "technical_v2"],
    "liquidity_volume": ["baseline", "liquidity_volume"],
}

SYSTEM_MODES = {
    "default": "no_risk_control",
    "conservative": "cash_de_risking",
}


@dataclass
class AccountState:
    portfolio_name: str
    signal_name: str
    system_mode: str
    initial_capital: float = INITIAL_CAPITAL
    cash: float = INITIAL_CAPITAL
    holdings: dict | None = None
    target_weights: dict | None = None
    processed_decisions: list | None = None
    pending_decisions: list | None = None
    last_nav_date: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


# =============================================================================
# IO Helpers
# =============================================================================


def ensure_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FORWARD_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def read_json(path, default=None):
    if not os.path.exists(path):
        return {} if default is None else default
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def append_csv(path, rows):
    frame = pd.DataFrame(rows)
    if frame.empty:
        return
    frame.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def append_unique_csv(path, frame, key_column):
    if frame.empty:
        return
    if key_column not in frame.columns:
        append_csv(path, frame.to_dict("records"))
        return
    if os.path.exists(path):
        existing = pd.read_csv(path)
        if key_column not in existing.columns:
            if {"portfolio_name", "data_as_of", "execution_date"}.issubset(existing.columns):
                existing[key_column] = (
                    existing["portfolio_name"].astype(str)
                    + "|asof="
                    + existing["data_as_of"].astype(str)
                    + "|execution="
                    + existing["execution_date"].astype(str)
                )
            else:
                existing[key_column] = ""
            existing.to_csv(path, index=False)
        existing_keys = set(existing[key_column].dropna().astype(str))
        frame = frame[~frame[key_column].astype(str).isin(existing_keys)].copy()
    if not frame.empty:
        append_csv(path, frame.to_dict("records"))


def upsert_csv(path, frame, key_columns):
    if frame.empty:
        return
    key_columns = [key_columns] if isinstance(key_columns, str) else list(key_columns)
    if os.path.exists(path):
        existing = pd.read_csv(path)
        if set(key_columns).issubset(existing.columns):
            new_keys = set(map(tuple, frame[key_columns].astype(str).to_numpy()))
            old_keys = list(map(tuple, existing[key_columns].astype(str).to_numpy()))
            existing = existing[[key not in new_keys for key in old_keys]].copy()
            frame = pd.concat([existing, frame], ignore_index=True)
    frame.to_csv(path, index=False)


def read_csv(path, parse_dates=None):
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=parse_dates)


def safe_name(name):
    return str(name).replace("/", "_").replace(" ", "_")


def latest_completed_us_session_date():
    now_et = datetime.now(ZoneInfo("America/New_York"))
    today = pd.Timestamp(now_et.date())
    if now_et.weekday() >= 5:
        return (today - pd.offsets.BDay(1)).date()
    if (now_et.hour, now_et.minute) < (16, 15):
        return (today - pd.offsets.BDay(1)).date()
    return today.date()


def next_business_day(date):
    return (pd.Timestamp(date) + pd.offsets.BDay(1)).normalize()


# =============================================================================
# Data Loading
# =============================================================================


def download_ohlcv():
    import yfinance as yf

    tickers = ALL_STOCKS + ["SPY"]
    print(f"Downloading {len(tickers)} equity tickers from {START_DATE} ...", flush=True)
    raw = yf.download(tickers, start=START_DATE, auto_adjust=True, progress=False, threads=True)
    if raw.empty:
        raise RuntimeError("Yahoo Finance returned no equity data.")

    frames = []
    for ticker in tickers:
        try:
            frame = raw.xs(ticker, axis=1, level=1) if isinstance(raw.columns, pd.MultiIndex) else raw.copy()
            keep = frame[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Open", "Close"]).copy()
            keep["ticker"] = ticker
            keep["date"] = pd.to_datetime(keep.index)
            frames.append(keep.reset_index(drop=True))
        except Exception:
            print(f"  Warning: unavailable ticker {ticker}", flush=True)
    out = pd.concat(frames, ignore_index=True)
    out.sort_values(["ticker", "date"], inplace=True)
    return out[["date", "ticker", "Open", "High", "Low", "Close", "Volume"]]


def download_macro():
    import yfinance as yf

    frames = []
    for ticker, column in [("^VIX", "vix"), ("^TNX", "tnx")]:
        print(f"Downloading macro series {ticker} ...", flush=True)
        raw = yf.download(ticker, start=START_DATE, auto_adjust=False, progress=False)
        if raw.empty:
            continue
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        series = raw[["Close"]].rename(columns={"Close": column}).copy()
        series["date"] = pd.to_datetime(series.index)
        frames.append(series.reset_index(drop=True))
    if not frames:
        return pd.DataFrame(columns=["date", "vix", "tnx"])
    macro = frames[0]
    for frame in frames[1:]:
        macro = macro.merge(frame, on="date", how="outer")
    return macro.sort_values("date")


def load_market_data(force_download=False, as_of=None):
    ensure_dirs()
    price_cache = os.path.join(CACHE_DIR, "ohlcv_prices.csv")
    macro_cache = os.path.join(CACHE_DIR, "macro.csv")
    as_of_ts = pd.Timestamp(as_of) if as_of is not None else None

    if os.path.exists(price_cache) and not force_download:
        prices = pd.read_csv(price_cache, parse_dates=["date"])
        if as_of_ts is not None and prices["date"].max() < as_of_ts:
            print(
                f"Price cache ends at {prices['date'].max().date()}, before requested as-of {as_of_ts.date()}; refreshing cache ...",
                flush=True,
            )
            try:
                prices = download_ohlcv()
                prices.to_csv(price_cache, index=False)
            except Exception as exc:
                print(f"  Warning: cache refresh failed; using existing cache. Reason: {exc}", flush=True)
    else:
        prices = download_ohlcv()
        prices.to_csv(price_cache, index=False)

    if os.path.exists(macro_cache) and not force_download:
        macro = pd.read_csv(macro_cache, parse_dates=["date"])
        if as_of_ts is not None and not macro.empty and macro["date"].max() < as_of_ts:
            print(
                f"Macro cache ends at {macro['date'].max().date()}, before requested as-of {as_of_ts.date()}; refreshing cache ...",
                flush=True,
            )
            try:
                macro = download_macro()
                macro.to_csv(macro_cache, index=False)
            except Exception as exc:
                print(f"  Warning: macro refresh failed; using existing cache. Reason: {exc}", flush=True)
    else:
        macro = download_macro()
        macro.to_csv(macro_cache, index=False)

    return prices, macro


def apply_as_of(prices, macro, as_of):
    cutoff = pd.Timestamp(as_of)
    return (
        prices[prices["date"].le(cutoff)].copy(),
        macro[macro["date"].le(cutoff)].copy(),
    )


# =============================================================================
# Feature Engineering
# =============================================================================


def rsi(close, period):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    return 100.0 - 100.0 / (1.0 + gain / (loss + EPS))


def build_macro_features(spy, macro):
    out = pd.DataFrame({"date": spy["date"].sort_values().unique()})
    out = out.merge(macro, on="date", how="left").sort_values("date")
    out[["vix", "tnx"]] = out[["vix", "tnx"]].ffill()
    spy_close = spy.set_index("date")["Close"].sort_index()
    out = out.set_index("date")
    out["vix_chg_5d"] = out["vix"].diff(5)
    out["vix_ma_ratio"] = out["vix"] / (out["vix"].rolling(21).mean() + EPS) - 1.0
    out["tnx_chg_5d"] = out["tnx"].diff(5)
    out["spy_ret_5d"] = spy_close.pct_change(5)
    out["spy_ret_21d"] = spy_close.pct_change(21)
    out["spy_ma_ratio"] = spy_close / (spy_close.rolling(21).mean() + EPS) - 1.0
    return out.shift(1).reset_index()


def stock_feature_frame(df, spy):
    df = df.sort_values("date").copy()
    close, open_, high, low, volume = df["Close"], df["Open"], df["High"], df["Low"], df["Volume"]
    spy_close = spy.set_index("date")["Close"].reindex(df["date"]).ffill().to_numpy()
    out = pd.DataFrame({"date": df["date"].values})

    for n in [1, 5, 10, 21, 63]:
        out[f"ret_{n}d"] = close.pct_change(n).values
    log_ret = np.log(close / close.shift(1))
    for n in [5, 21, 63]:
        out[f"vol_{n}d"] = log_ret.rolling(n).std().values
    for n in [21, 63, 252]:
        out[f"ma_ratio_{n}"] = (close / (close.rolling(n).mean() + EPS) - 1.0).values

    out["rsi_14"] = rsi(close, 14).values
    out["rsi_28"] = rsi(close, 28).values
    out["vol_ratio_21"] = (volume / (volume.rolling(21).mean() + EPS)).values
    out["hl_range_5d"] = ((high.rolling(5).max() - low.rolling(5).min()) / (close + EPS)).values
    out["rs_spy_5d"] = close.pct_change(5).values - pd.Series(spy_close).pct_change(5).values
    out["rs_spy_21d"] = close.pct_change(21).values - pd.Series(spy_close).pct_change(21).values

    stock_ret = close.pct_change()
    spy_ret = pd.Series(spy_close, index=df.index).pct_change()
    out["tech_reversal_1d"] = (-close.pct_change(1)).values
    out["tech_reversal_5d"] = (-close.pct_change(5)).values
    out["tech_momentum_63d"] = close.pct_change(63).values
    out["tech_momentum_126d"] = close.pct_change(126).values
    out["tech_momentum_252d"] = close.pct_change(252).values
    out["tech_momentum_12_1"] = (close.shift(21) / (close.shift(252) + EPS) - 1.0).values
    for n in [21, 63]:
        out[f"tech_beta_spy_{n}d"] = (stock_ret.rolling(n).cov(spy_ret) / (spy_ret.rolling(n).var() + EPS)).values
        out[f"tech_corr_spy_{n}d"] = stock_ret.rolling(n).corr(spy_ret).values
        out[f"tech_downside_vol_{n}d"] = stock_ret.where(stock_ret < 0.0, 0.0).rolling(n).std().values
    for n in [21, 63, 252]:
        out[f"tech_drawdown_{n}d"] = (close / (close.rolling(n).max() + EPS) - 1.0).values
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    out["tech_ema_trend"] = (ema_12 / (ema_26 + EPS) - 1.0).values
    out["tech_ma21_vs_ma63"] = (close.rolling(21).mean() / (close.rolling(63).mean() + EPS) - 1.0).values
    out["tech_gap_return"] = (open_ / (close.shift(1) + EPS) - 1.0).values
    out["tech_intraday_return"] = (close / (open_ + EPS) - 1.0).values
    out["tech_range_21d"] = ((high.rolling(21).max() - low.rolling(21).min()) / (close + EPS)).values

    dollar_volume = close * volume
    out["liq_dollar_volume_log"] = np.log1p(dollar_volume).values
    out["liq_volume_z_21d"] = ((volume - volume.rolling(21).mean()) / (volume.rolling(21).std() + EPS)).values
    out["liq_dollar_volume_z_21d"] = (
        (dollar_volume - dollar_volume.rolling(21).mean()) / (dollar_volume.rolling(21).std() + EPS)
    ).values
    out["liq_volume_ratio_21d"] = (volume / (volume.rolling(21).mean() + EPS)).values
    out["liq_volume_ratio_63d"] = (volume / (volume.rolling(63).mean() + EPS)).values
    out["liq_amihud_21d"] = (stock_ret.abs() / (dollar_volume + EPS)).rolling(21).mean().values
    out["liq_range_volume_shock"] = (
        ((high - low) / (close + EPS)).to_numpy() * out["liq_volume_ratio_21d"].to_numpy()
    )

    shifted = out.set_index("date").shift(1).reset_index()
    shifted["Open"] = open_.values
    shifted["Close"] = close.values
    shifted["ticker"] = df["ticker"].iloc[0]
    shifted["sector"] = TICKER_TO_SECTOR.get(df["ticker"].iloc[0], "")
    return shifted


def build_panel(prices, macro):
    spy = prices[prices["ticker"].eq("SPY")].sort_values("date").copy()
    macro_features = build_macro_features(spy, macro)
    spy_open = spy.set_index("date")["Open"].sort_index()
    pieces = []
    for ticker in ALL_STOCKS:
        df = prices[prices["ticker"].eq(ticker)].sort_values("date")
        if len(df) < 500:
            continue
        feat = stock_feature_frame(df, spy)
        open_series = df.set_index("date")["Open"].sort_index()
        stock_fwd = open_series.shift(-(PREDICTION_HORIZON + 1)) / open_series.shift(-1) - 1.0
        spy_fwd = spy_open.shift(-(PREDICTION_HORIZON + 1)) / spy_open.shift(-1) - 1.0
        feat = feat.merge(macro_features, on="date", how="left")
        feat["raw_excess"] = feat["date"].map((stock_fwd - spy_fwd).to_dict())
        pieces.append(feat)
    panel = pd.concat(pieces, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"])
    panel.sort_values(["date", "ticker"], inplace=True)
    panel["target"] = panel.groupby("date")["raw_excess"].rank(pct=True) - 0.5

    for col in ["ret_5d", "ret_21d", "vol_21d", "rs_spy_5d", "rs_spy_21d"]:
        panel[f"{col}_rank"] = panel.groupby("date")[col].rank(pct=True)
    panel["breadth_5d"] = panel.groupby("date")["ret_5d"].transform(lambda x: (x > 0).mean())

    group = panel.groupby(["date", "sector"])
    for col in [
        "ret_5d",
        "ret_21d",
        "vol_21d",
        "rs_spy_5d",
        "rs_spy_21d",
        "ma_ratio_21",
        "rsi_14",
        "vol_ratio_21",
    ]:
        panel[f"sec_rel_{col}"] = panel[col] - group[col].transform("mean")
        panel[f"sec_z_{col}"] = (panel[col] - group[col].transform("mean")) / (group[col].transform("std") + EPS)
        panel[f"sec_rank_{col}"] = group[col].rank(pct=True)
    panel["sec_breadth_5d"] = group["ret_5d"].transform(lambda x: (x > 0).mean())

    panel.replace([np.inf, -np.inf], np.nan, inplace=True)
    return panel


def feature_columns_for(panel, variant):
    baseline = [
        "ret_1d",
        "ret_5d",
        "ret_10d",
        "ret_21d",
        "ret_63d",
        "vol_5d",
        "vol_21d",
        "vol_63d",
        "ma_ratio_21",
        "ma_ratio_63",
        "ma_ratio_252",
        "rsi_14",
        "rsi_28",
        "vol_ratio_21",
        "hl_range_5d",
        "rs_spy_5d",
        "rs_spy_21d",
        "vix",
        "vix_chg_5d",
        "vix_ma_ratio",
        "tnx",
        "tnx_chg_5d",
        "spy_ret_5d",
        "spy_ret_21d",
        "spy_ma_ratio",
        "ret_5d_rank",
        "ret_21d_rank",
        "vol_21d_rank",
        "rs_spy_5d_rank",
        "rs_spy_21d_rank",
        "breadth_5d",
    ]
    blocks = {
        "baseline": baseline,
        "technical_v2": [c for c in panel.columns if c.startswith("tech_")],
        "liquidity_volume": [c for c in panel.columns if c.startswith("liq_")],
        "sector_relative": [
            c
            for c in panel.columns
            if c.startswith("sec_rel_") or c.startswith("sec_z_") or c.startswith("sec_rank_") or c == "sec_breadth_5d"
        ],
    }
    cols = []
    for block in FEATURE_BLOCKS[variant]:
        cols.extend(blocks[block])
    return [c for c in dict.fromkeys(cols) if c in panel.columns]


# =============================================================================
# Model and Signal Layer
# =============================================================================


def fit_predict_signal(panel, signal, signal_date):
    cols = feature_columns_for(panel, signal["feature_variant"])
    train = panel[
        panel["date"].ge(pd.Timestamp(START_DATE))
        & panel["date"].lt(pd.Timestamp(signal_date))
        & panel["target"].notna()
    ].dropna(subset=cols + ["target"]).copy()
    predict = panel[panel["date"].eq(signal_date)].dropna(subset=cols).copy()
    if train.empty or predict.empty:
        raise RuntimeError(f"Insufficient data for {signal['signal_name']}.")

    X_train = train[cols].to_numpy()
    y_train = train["target"].to_numpy()
    X_pred = predict[cols].to_numpy()
    model_name = signal["model"]

    if model_name == "Ridge":
        scaler = StandardScaler().fit(X_train)
        model = Ridge(alpha=10.0)
        model.fit(scaler.transform(X_train), y_train)
        score = model.predict(scaler.transform(X_pred))
    elif model_name == "RandomForest":
        model = RandomForestRegressor(
            n_estimators=220,
            max_depth=5,
            min_samples_leaf=30,
            max_features=0.5,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        score = model.predict(X_pred)
    elif model_name == "XGBoost":
        from xgboost import XGBRegressor

        model = XGBRegressor(
            n_estimators=320,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.7,
            colsample_bytree=0.7,
            min_child_weight=30,
            objective="reg:squarederror",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        score = model.predict(X_pred)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    out = predict[["date", "ticker", "sector", "Close"]].copy()
    out["signal_name"] = signal["signal_name"]
    out["model"] = model_name
    out["feature_variant"] = signal["feature_variant"]
    out["score"] = score
    out["n_train_rows"] = len(train)
    out["trained_through"] = str(train["date"].max().date())
    out["n_features"] = len(cols)
    return out


def build_latest_signals(panel, signal_date):
    pieces = []
    for signal in CANDIDATE_SIGNALS:
        print(f"Training/scoring {signal['signal_name']} ...", flush=True)
        pieces.append(fit_predict_signal(panel, signal, signal_date))
    scored = pd.concat(pieces, ignore_index=True)
    scored["rank_score"] = scored.groupby("signal_name")["score"].rank(pct=True)
    ensemble = (
        scored.groupby(["date", "ticker", "sector"], as_index=False)
        .agg(score=("rank_score", "mean"), Close=("Close", "last"))
    )
    ensemble["signal_name"] = "ensemble_equal_rank"
    ensemble["model"] = "Ensemble"
    ensemble["feature_variant"] = "six_signal_equal_rank"
    ensemble["rank_score"] = ensemble.groupby("date")["score"].rank(pct=True)
    ensemble["n_train_rows"] = np.nan
    ensemble["trained_through"] = ""
    ensemble["n_features"] = np.nan
    return pd.concat([scored, ensemble[scored.columns]], ignore_index=True)


# =============================================================================
# Portfolio, Risk, and Trade Decision Layer
# =============================================================================


def top_pct_count(n_available, pct):
    return max(1, min(int(np.ceil(n_available * pct)), n_available))


def sector_capped_equal_weights(selected, sector_cap=SECTOR_CAP):
    if selected.empty:
        return {}
    counts = selected.groupby("sector")["ticker"].count()
    raw_sector = counts / counts.sum()
    sector_weights = raw_sector.clip(upper=sector_cap)
    remaining = 1.0 - sector_weights.sum()
    uncapped = sector_weights[raw_sector < sector_cap].index.tolist()
    while remaining > EPS and uncapped:
        base = raw_sector.loc[uncapped]
        add = remaining * base / base.sum()
        changed = False
        for sector, extra in add.items():
            new_weight = sector_weights.loc[sector] + extra
            if new_weight > sector_cap:
                remaining -= sector_cap - sector_weights.loc[sector]
                sector_weights.loc[sector] = sector_cap
                uncapped.remove(sector)
                changed = True
            else:
                sector_weights.loc[sector] = new_weight
        if not changed:
            remaining = 1.0 - sector_weights.sum()
            if abs(remaining) < 1e-6:
                break
            break
    if sector_weights.sum() <= EPS:
        return {ticker: 1.0 / len(selected) for ticker in selected["ticker"]}
    sector_weights = sector_weights / sector_weights.sum()
    weights = {}
    for sector, group in selected.groupby("sector"):
        weights.update({ticker: float(sector_weights.loc[sector] / len(group)) for ticker in group["ticker"]})
    return weights


def should_reconstitute(signal_date, account):
    current = account.get("base_stock_weights", {}) or account.get("target_weights", {}) or {}
    if not current:
        return True
    last = account.get("last_reconstitution_signal_date")
    if not last:
        return True
    prev = pd.Timestamp(last).isocalendar()
    curr = pd.Timestamp(signal_date).isocalendar()
    return (prev.year, prev.week) != (curr.year, curr.week)


def portfolio_turnover(new_weights, old_weights):
    tickers = set(new_weights) | set(old_weights)
    return float(sum(abs(float(new_weights.get(t, 0.0)) - float(old_weights.get(t, 0.0))) for t in tickers))


def partial_event_update(ranked, current_base_weights):
    current_base_weights = {ticker: float(weight) for ticker, weight in (current_base_weights or {}).items() if weight > EPS}
    if not current_base_weights:
        return list(ranked.head(top_pct_count(len(ranked), SELECTION_ENTRY_PCT))["ticker"]), {
            "partial_update_triggered": False,
            "partial_replacements": 0,
            "partial_update_turnover": 0.0,
            "partial_update_note": "no_existing_base_basket",
        }

    top_opportunity = ranked.head(top_pct_count(len(ranked), EXTREME_OPPORTUNITY_PCT))["ticker"].tolist()
    top_keep = set(ranked.head(top_pct_count(len(ranked), EXTREME_DETERIORATION_PCT))["ticker"])
    current_tickers = list(current_base_weights)
    deteriorated = [ticker for ticker in current_tickers if ticker not in top_keep]
    opportunities = [ticker for ticker in top_opportunity if ticker not in current_base_weights]
    if not deteriorated or not opportunities:
        return current_tickers, {
            "partial_update_triggered": False,
            "partial_replacements": 0,
            "partial_update_turnover": 0.0,
            "partial_update_note": "no_extreme_pair",
        }

    rank_pos = {ticker: idx for idx, ticker in enumerate(ranked["ticker"].tolist())}
    deteriorated = sorted(deteriorated, key=lambda ticker: rank_pos.get(ticker, -1), reverse=True)
    opportunities = sorted(opportunities, key=lambda ticker: rank_pos.get(ticker, 10**9))
    replacements = min(MAX_EVENT_REPLACEMENTS, len(deteriorated), len(opportunities))
    candidate_tickers = [ticker for ticker in current_tickers if ticker not in set(deteriorated[:replacements])]
    candidate_tickers.extend(opportunities[:replacements])
    candidate_tickers = list(dict.fromkeys(candidate_tickers))
    candidate = ranked[ranked["ticker"].isin(candidate_tickers)].copy()
    candidate_weights = sector_capped_equal_weights(candidate, SECTOR_CAP)
    turnover = portfolio_turnover(candidate_weights, current_base_weights)
    if turnover > EVENT_TURNOVER_CAP:
        return current_tickers, {
            "partial_update_triggered": False,
            "partial_replacements": 0,
            "partial_update_turnover": turnover,
            "partial_update_note": "blocked_by_turnover_cap",
        }
    return candidate_tickers, {
        "partial_update_triggered": True,
        "partial_replacements": replacements,
        "partial_update_turnover": turnover,
        "partial_update_note": (
            "replace "
            + ",".join(deteriorated[:replacements])
            + " with "
            + ",".join(opportunities[:replacements])
        ),
    }


def base_stock_weights(daily_scores, account, signal_date):
    ranked = daily_scores.sort_values("score", ascending=False).reset_index(drop=True)
    is_weekly = should_reconstitute(signal_date, account)
    if is_weekly:
        entry = ranked.head(top_pct_count(len(ranked), SELECTION_ENTRY_PCT))["ticker"].tolist()
        exit_set = set(ranked.head(top_pct_count(len(ranked), SELECTION_EXIT_PCT))["ticker"])
        current = account.get("base_stock_weights", {}) or account.get("target_weights", {}) or {}
        kept = [ticker for ticker, weight in current.items() if float(weight) > EPS and ticker in exit_set]
        selected_tickers = list(dict.fromkeys(kept + entry))
        event_info = {
            "partial_update_triggered": False,
            "partial_replacements": 0,
            "partial_update_turnover": 0.0,
            "partial_update_note": "weekly_reconstitution",
        }
    else:
        selected_tickers, event_info = partial_event_update(ranked, account.get("base_stock_weights", {}) or {})
    selected = ranked[ranked["ticker"].isin(selected_tickers)].copy()
    return sector_capped_equal_weights(selected, SECTOR_CAP), is_weekly, event_info


def build_market_indicators(prices, macro, signal_date):
    spy = prices[prices["ticker"].eq("SPY")].sort_values("date").set_index("date")
    close = spy["Close"]
    vix = macro.set_index("date")["vix"].sort_index().ffill() if "vix" in macro.columns else pd.Series(dtype=float)
    date = pd.Timestamp(signal_date)
    row = {
        "date": date,
        "spy_close": float(close.loc[date]),
        "spy_ma20": float(close.rolling(20).mean().loc[date]),
        "spy_ma200": float(close.rolling(200).mean().loc[date]),
        "spy_ret21": float(close.pct_change(21).loc[date]),
        "spy_vol21_ann": float(close.pct_change().rolling(21).std().loc[date] * np.sqrt(252)),
        "vix": float(vix.reindex([date]).ffill().iloc[0]) if not vix.empty else np.nan,
    }
    score = 0
    score += int(pd.notna(row["spy_ret21"]) and row["spy_ret21"] < 0.0)
    score += int(pd.notna(row["spy_ma200"]) and row["spy_close"] < row["spy_ma200"])
    score += int(pd.notna(row["vix"]) and row["vix"] > 25.0)
    score += int(pd.notna(row["spy_vol21_ann"]) and row["spy_vol21_ann"] > 0.20)
    row["risk_score"] = score
    row["market_state"] = "normal" if score == 0 else ("caution" if score == 1 else "stress")
    return row


def exposure_for_mode(system_mode, market_state):
    if system_mode == "default":
        return 1.0
    if market_state == "normal":
        return 1.0
    if market_state == "caution":
        return 0.75
    return 0.50


def build_target_weights(base_weights, stock_exposure):
    return {ticker: float(weight * stock_exposure) for ticker, weight in base_weights.items() if weight * stock_exposure > EPS}


def trade_instruction(old, new, tol=0.0005):
    if old <= tol and new > tol:
        return "BUY"
    if old > tol and new <= tol:
        return "SELL"
    if new > old + tol:
        return "ADD"
    if new < old - tol:
        return "REDUCE"
    return "HOLD"


def build_decision_rows(scored, target_weights, previous_weights, meta):
    tickers = sorted(set(previous_weights) | set(target_weights))
    rows = []
    score_map = scored.set_index("ticker")["score"].to_dict()
    sector_map = scored.set_index("ticker")["sector"].to_dict()
    price_map = scored.set_index("ticker")["Close"].to_dict()
    for ticker in tickers:
        old = float(previous_weights.get(ticker, 0.0))
        new = float(target_weights.get(ticker, 0.0))
        rows.append(
            {
                **meta,
                "ticker": ticker,
                "sector": sector_map.get(ticker, TICKER_TO_SECTOR.get(ticker, "")),
                "score": score_map.get(ticker, np.nan),
                "reference_close": price_map.get(ticker, np.nan),
                "previous_weight": old,
                "target_weight": new,
                "cash_weight": max(0.0, 1.0 - sum(target_weights.values())),
                "trade_instruction": trade_instruction(old, new),
            }
        )
    return pd.DataFrame(rows)


# =============================================================================
# Paper Trading Layer
# =============================================================================


def portfolio_dir(portfolio_name):
    return os.path.join(FORWARD_DIR, safe_name(portfolio_name))


def init_account(portfolio_name, signal_name, system_mode):
    now = datetime.now().isoformat(timespec="seconds")
    return asdict(
        AccountState(
            portfolio_name=portfolio_name,
            signal_name=signal_name,
            system_mode=system_mode,
            holdings={},
            target_weights={},
            processed_decisions=[],
            pending_decisions=[],
            created_at=now,
            updated_at=now,
        )
    )


def load_account(path, portfolio_name, signal_name, system_mode):
    os.makedirs(path, exist_ok=True)
    os.makedirs(os.path.join(path, "signals"), exist_ok=True)
    state_path = os.path.join(path, "account_state.json")
    if not os.path.exists(state_path):
        account = init_account(portfolio_name, signal_name, system_mode)
        write_json(state_path, account)
        return account
    return read_json(state_path)


def price_lookup(prices):
    return prices.set_index(["date", "ticker"]).sort_index()


def price_on(price_map, date, ticker, field):
    key = (pd.Timestamp(date), ticker)
    if key not in price_map.index:
        return np.nan
    return float(price_map.loc[key, field])


def portfolio_value_at_open(account, price_map, date):
    value = float(account.get("cash", INITIAL_CAPITAL))
    for ticker, holding in (account.get("holdings", {}) or {}).items():
        px = price_on(price_map, date, ticker, "Open")
        if np.isfinite(px):
            value += float(holding.get("shares", 0.0)) * px
    return value


def missing_execution_open_prices(account, decision, price_map, execution_date):
    current_holdings = set((account.get("holdings", {}) or {}).keys())
    target_weights = decision.set_index("ticker")["target_weight"].to_dict()
    target_tickers = {ticker for ticker, weight in target_weights.items() if float(weight) > EPS}
    tickers = sorted(current_holdings | target_tickers)
    return [ticker for ticker in tickers if not np.isfinite(price_on(price_map, execution_date, ticker, "Open"))]


def execute_decision(account, decision, price_map, execution_date):
    missing = missing_execution_open_prices(account, decision, price_map, execution_date)
    if missing:
        raise RuntimeError(
            f"Missing execution Open prices on {pd.Timestamp(execution_date).date()}: {', '.join(missing[:20])}"
        )
    current_value = portfolio_value_at_open(account, price_map, execution_date)
    current_holdings = account.get("holdings", {}) or {}
    target_weights = decision.set_index("ticker")["target_weight"].to_dict()
    tickers = sorted(set(current_holdings) | set(target_weights))
    trades = []
    desired = {}
    for ticker in tickers:
        px = price_on(price_map, execution_date, ticker, "Open")
        if not np.isfinite(px):
            continue
        target_dollars = current_value * float(target_weights.get(ticker, 0.0))
        desired[ticker] = int(np.floor(target_dollars / max(px, EPS)))

    cash = float(account.get("cash", INITIAL_CAPITAL))
    new_holdings = dict(current_holdings)
    for ticker in tickers:
        px = price_on(price_map, execution_date, ticker, "Open")
        if not np.isfinite(px):
            continue
        old_shares = int(float(current_holdings.get(ticker, {}).get("shares", 0)))
        new_shares = int(desired.get(ticker, 0))
        delta = new_shares - old_shares
        if delta == 0:
            continue
        gross = abs(delta) * px
        cost = gross * TRANSACTION_COST_RATE
        if delta > 0:
            cash -= gross + cost
            action = "BUY" if old_shares == 0 else "ADD"
        else:
            cash += gross - cost
            action = "SELL" if new_shares == 0 else "REDUCE"
        if new_shares > 0:
            new_holdings[ticker] = {"shares": new_shares, "last_price": px}
        else:
            new_holdings.pop(ticker, None)
        trades.append(
            {
                "date": str(pd.Timestamp(execution_date).date()),
                "ticker": ticker,
                "action": action,
                "shares_traded": delta,
                "fill_price": px,
                "gross_notional": gross,
                "transaction_cost": cost,
            }
        )
    account["cash"] = cash
    account["holdings"] = new_holdings
    account["target_weights"] = {k: float(v) for k, v in target_weights.items() if float(v) > EPS}
    return trades


def process_pending_decisions(account, path, prices):
    pending = account.get("pending_decisions", []) or []
    if not pending:
        return account
    latest_price_date = pd.Timestamp(prices["date"].max())
    price_map = price_lookup(prices)
    remaining = []
    processed = set(account.get("processed_decisions", []) or [])
    all_trades = []
    for decision_file in pending:
        if decision_file in processed or not os.path.exists(decision_file):
            continue
        decision = pd.read_csv(decision_file)
        if decision.empty or "execution_date" not in decision.columns:
            remaining.append(decision_file)
            continue
        execution_date = pd.Timestamp(decision["execution_date"].iloc[0])
        if execution_date <= latest_price_date:
            missing = missing_execution_open_prices(account, decision, price_map, execution_date)
            if missing:
                remaining.append(decision_file)
                account["last_pending_reason"] = (
                    f"missing execution Open prices on {execution_date.date()}: {', '.join(missing[:20])}"
                )
                continue
            trades = execute_decision(account, decision, price_map, execution_date)
            all_trades.extend(trades)
            processed.add(decision_file)
            account["last_execution_date"] = str(execution_date.date())
            account["last_pending_reason"] = ""
        else:
            remaining.append(decision_file)
            account["last_pending_reason"] = f"waiting for execution date {execution_date.date()} prices"
    if all_trades:
        append_csv(os.path.join(path, "trade_ledger.csv"), all_trades)
    account["processed_decisions"] = sorted(processed)
    account["pending_decisions"] = remaining
    account["updated_at"] = datetime.now().isoformat(timespec="seconds")
    write_json(os.path.join(path, "account_state.json"), account)
    return account


def mark_to_market(account, price_map, date):
    date = pd.Timestamp(date)
    cash = float(account.get("cash", INITIAL_CAPITAL))
    total = cash
    positions = []
    for ticker, holding in (account.get("holdings", {}) or {}).items():
        px = price_on(price_map, date, ticker, "Close")
        if not np.isfinite(px):
            continue
        shares = float(holding.get("shares", 0.0))
        value = shares * px
        total += value
        positions.append(
            {
                "date": str(date.date()),
                "ticker": ticker,
                "shares": shares,
                "close_price": px,
                "market_value": value,
            }
        )
    for row in positions:
        row["actual_weight"] = row["market_value"] / max(total, EPS)
    nav_row = {
        "date": str(date.date()),
        "portfolio_value": total,
        "cash": cash,
        "cash_weight": cash / max(total, EPS),
        "n_holdings": len(positions),
    }
    return nav_row, positions


def update_paper_account(portfolio_name, signal_name, system_mode, decision, prices, signal_date, execution_date, reset=False):
    path = portfolio_dir(portfolio_name)
    if reset and os.path.exists(path):
        shutil.rmtree(path)
    account = load_account(path, portfolio_name, signal_name, system_mode)
    price_map = price_lookup(prices)
    latest_price_date = prices["date"].max()
    os.makedirs(os.path.join(path, "status"), exist_ok=True)
    executable_decision = str(decision["decision"].iloc[0]).upper() == "REBALANCE" if not decision.empty else False
    file_folder = "signals" if executable_decision else "status"
    file_prefix = "decision" if executable_decision else "hold_status"
    decision_file = os.path.join(
        path,
        file_folder,
        f"{file_prefix}_{safe_name(portfolio_name)}_{pd.Timestamp(execution_date).date()}_asof_{pd.Timestamp(signal_date).date()}.csv",
    )
    decision_id = (
        f"{safe_name(portfolio_name)}|asof={pd.Timestamp(signal_date).date()}|"
        f"execution={pd.Timestamp(execution_date).date()}"
    )
    decision = decision.copy()
    decision["decision_id"] = decision_id

    execution_status = "pending"
    trades = []
    processed = set(account.get("processed_decisions", []) or [])
    pending = set(account.get("pending_decisions", []) or [])
    already_queued = decision_file in pending
    already_processed = decision_file in processed
    if not already_queued and not already_processed:
        decision.to_csv(decision_file, index=False)
        append_unique_csv(os.path.join(path, "daily_decisions.csv"), decision, "decision_id")
        append_unique_csv(os.path.join(path, "target_weights.csv"), decision[decision["target_weight"].gt(EPS)].copy(), "decision_id")

    if not executable_decision:
        execution_status = "hold_pending_execution" if pending else "not_executable_hold"
        if not pending:
            account["last_pending_reason"] = ""
    elif pd.Timestamp(execution_date) <= pd.Timestamp(latest_price_date) and not already_processed:
        execution_decision = pd.read_csv(decision_file) if os.path.exists(decision_file) else decision
        missing = missing_execution_open_prices(account, execution_decision, price_map, execution_date)
        if missing:
            pending.add(decision_file)
            execution_status = "pending_no_execution_price"
            account["last_pending_reason"] = (
                f"missing execution Open prices on {pd.Timestamp(execution_date).date()}: {', '.join(missing[:20])}"
            )
        else:
            trades = execute_decision(account, execution_decision, price_map, execution_date)
            append_csv(os.path.join(path, "trade_ledger.csv"), trades)
            processed.add(decision_file)
            execution_status = "executed"
            pending.discard(decision_file)
            account["last_pending_reason"] = ""
    elif already_processed:
        execution_status = "already_processed"
    else:
        pending.add(decision_file)
        execution_status = "pending_existing" if already_queued else "pending"
        account["last_pending_reason"] = f"waiting for execution date {pd.Timestamp(execution_date).date()} prices"
    account["pending_decisions"] = sorted(pending)

    nav_row, position_rows = mark_to_market(account, price_map, min(pd.Timestamp(signal_date), pd.Timestamp(latest_price_date)))
    nav_path = os.path.join(path, "daily_nav.csv")
    old_nav = read_csv(nav_path)
    old_nav_before_today = old_nav[old_nav["date"].astype(str).ne(nav_row["date"])] if not old_nav.empty else old_nav
    previous_value = old_nav_before_today["portfolio_value"].iloc[-1] if not old_nav_before_today.empty else INITIAL_CAPITAL
    nav_row["daily_return"] = nav_row["portfolio_value"] / max(float(previous_value), EPS) - 1.0
    nav_row["cumulative_return"] = nav_row["portfolio_value"] / INITIAL_CAPITAL - 1.0
    upsert_csv(nav_path, pd.DataFrame([nav_row]), "date")
    upsert_csv(os.path.join(path, "daily_positions.csv"), pd.DataFrame(position_rows), ["date", "ticker"])

    account["processed_decisions"] = sorted(processed)
    account["last_signal_date"] = str(pd.Timestamp(signal_date).date())
    account["last_execution_date"] = str(pd.Timestamp(execution_date).date()) if execution_status == "executed" else account.get("last_execution_date")
    account["last_nav_date"] = nav_row["date"]
    account["last_decision_file"] = decision_file
    account["last_execution_status"] = execution_status
    if "market_state" in decision.columns and not decision.empty:
        account["last_market_state"] = str(decision["market_state"].iloc[0])
    account["last_reconstitution_signal_date"] = str(pd.Timestamp(signal_date).date())
    account["base_stock_weights"] = {
        row["ticker"]: float(row["base_stock_weight"])
        for _, row in decision.drop_duplicates("ticker").iterrows()
        if float(row.get("base_stock_weight", 0.0)) > EPS
    }
    account["updated_at"] = datetime.now().isoformat(timespec="seconds")
    write_json(os.path.join(path, "account_state.json"), account)
    return {
        "portfolio_name": portfolio_name,
        "signal_name": signal_name,
        "system_mode": system_mode,
        "signal_date": str(pd.Timestamp(signal_date).date()),
        "execution_date": str(pd.Timestamp(execution_date).date()),
        "execution_status": execution_status,
        "portfolio_value": nav_row["portfolio_value"],
        "cumulative_return": nav_row["cumulative_return"],
        "cash_weight": nav_row["cash_weight"],
        "n_holdings": nav_row["n_holdings"],
        "n_trades": len(trades),
        "pending_decisions": len(account.get("pending_decisions", []) or []),
        "pending_reason": account.get("last_pending_reason", ""),
        "decision_file": decision_file,
    }


def build_spy_benchmark_account(prices, signal_date, execution_date):
    portfolio_name = "SPY_benchmark"
    signal_name = "SPY_benchmark"
    system_mode = "benchmark"
    path = portfolio_dir(portfolio_name)
    account = load_account(path, portfolio_name, signal_name, system_mode)
    account = process_pending_decisions(account, path, prices)
    previous_weights = account.get("target_weights", {}) or {}
    spy_row = prices[prices["ticker"].eq("SPY") & prices["date"].eq(pd.Timestamp(signal_date))]
    reference_close = float(spy_row["Close"].iloc[0]) if not spy_row.empty else np.nan
    old = float(previous_weights.get("SPY", 0.0))
    decision = pd.DataFrame(
        [
            {
                "portfolio_name": portfolio_name,
                "signal_name": signal_name,
                "system_mode": system_mode,
                "risk_overlay": "benchmark",
                "data_as_of": str(pd.Timestamp(signal_date).date()),
                "execution_date": str(pd.Timestamp(execution_date).date()),
                "decision": "REBALANCE" if old <= EPS else "HOLD",
                "trigger_reason": "benchmark_initial_buy" if old <= EPS else "benchmark_hold",
                "market_state": "benchmark",
                "risk_score": np.nan,
                "stock_exposure": 1.0,
                "basket_reconstituted": old <= EPS,
                "risk_exposure_update": False,
                "risk_state_changed": False,
                "partial_update_triggered": False,
                "partial_replacements": 0,
                "partial_update_turnover": 0.0,
                "partial_update_note": "benchmark",
                "event_turnover_cap": EVENT_TURNOVER_CAP,
                "max_event_replacements": MAX_EVENT_REPLACEMENTS,
                "ticker": "SPY",
                "sector": "Benchmark",
                "score": np.nan,
                "reference_close": reference_close,
                "previous_weight": old,
                "target_weight": 1.0,
                "cash_weight": 0.0,
                "trade_instruction": trade_instruction(old, 1.0),
                "base_stock_weight": 1.0,
            }
        ]
    )
    status = update_paper_account(
        portfolio_name,
        signal_name,
        system_mode,
        decision,
        prices,
        signal_date,
        execution_date,
        reset=False,
    )
    return status, decision


# =============================================================================
# Reports
# =============================================================================


def plot_nav_curve(status):
    frames = []
    for portfolio_name in status["portfolio_name"]:
        path = os.path.join(portfolio_dir(portfolio_name), "daily_nav.csv")
        nav = read_csv(path, parse_dates=["date"])
        if nav.empty:
            continue
        nav["portfolio_name"] = portfolio_name
        frames.append(nav)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(os.path.join(OUT_DIR, "combined_forward_daily_nav.csv"), index=False)
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, group in combined.groupby("portfolio_name"):
        alpha = 0.9 if "ensemble" in name else 0.35
        lw = 2.4 if "ensemble" in name else 1.1
        ax.plot(group["date"], group["portfolio_value"], label=name, alpha=alpha, lw=lw)
    ax.axhline(INITIAL_CAPITAL, color="#111827", ls=":", lw=1.0, label="cash reference")
    ax.set_title("Forward Paper-Trading NAV")
    ax.set_ylabel("Portfolio Value")
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "forward_paper_trading_nav_curve.png"), dpi=160)
    plt.close(fig)
    return combined


def save_readme():
    text = f"""# Generated Dashboard Outputs

These files are generated by the forward decision-support engine used by the
Streamlit dashboard.

## Fixed Research Settings

- Data period: `{START_DATE}` to latest available completed market date
- Target: Label C, cross-sectional rank of 5-day future excess return
- Candidate signals: six model/feature signals from the thesis research line
- Ensemble: equal-weight average of cross-sectional ranks from the six signals
- Portfolio rule: top20pct, sector cap 30%, equal weight within the selected basket
- Trading rule: weekly basket reconstitution with 20% entry / 30% exit buffer
- Daily event layer: immediate conservative exposure updates and capped partial stock updates
- Transaction cost: {TRANSACTION_COST_BPS:.0f} bps
- Systems: default no-risk-control and conservative cash de-risking
- Benchmark account: SPY buy-and-hold
- Initial paper capital: ${INITIAL_CAPITAL:,.0f}

## Main Outputs

- `latest_signal_scores.csv`
- `latest_ensemble_scores.csv`
- `latest_target_weights.csv`
- `latest_decisions.csv`
- `portfolio_status_summary.csv`
- `combined_forward_daily_nav.csv`
- `forward_paper_trading_nav_curve.png`
- `prototype_system_config.json`
- `forward_paper_trading/`

Each portfolio under `forward_paper_trading/` has independent decisions,
positions, trades, NAV, and account state.

## Daily Command

```bash
python forward_engine.py
```

Use `--force-download` if the local Yahoo Finance cache is stale, and use
`--reset-paper` only when intentionally restarting the forward paper-trading
accounts.
"""
    with open(os.path.join(OUT_DIR, "README.md"), "w", encoding="utf-8") as handle:
        handle.write(text)


def save_config(signal_date, execution_date, market_row):
    clean_market_row = {
        key: (str(value.date()) if isinstance(value, pd.Timestamp) else value)
        for key, value in market_row.items()
    }
    payload = {
        "experiment_name": EXPERIMENT_NAME,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "data_period_start": START_DATE,
        "signal_date": str(pd.Timestamp(signal_date).date()),
        "execution_date": str(pd.Timestamp(execution_date).date()),
        "initial_capital": INITIAL_CAPITAL,
        "target_type": TARGET_TYPE,
        "prediction_horizon": PREDICTION_HORIZON,
        "candidate_signals": CANDIDATE_SIGNALS,
        "ensemble_method": "average of per-signal cross-sectional rank scores",
        "portfolio_rule": {
            "selection": "top20pct",
            "sector_cap": SECTOR_CAP,
            "allocation": "equal_weight",
        },
        "trading_rule": {
            "rebalance_frequency": "weekly",
            "entry_pct": SELECTION_ENTRY_PCT,
            "exit_pct": SELECTION_EXIT_PCT,
            "transaction_cost_bps": TRANSACTION_COST_BPS,
        },
        "event_trigger_layer": {
            "daily_signal_refresh": True,
            "market_state_exposure_update": "conservative mode adjusts stock exposure when the target exposure changes",
            "extreme_opportunity_pct": EXTREME_OPPORTUNITY_PCT,
            "extreme_deterioration_pct": EXTREME_DETERIORATION_PCT,
            "max_event_replacements": MAX_EVENT_REPLACEMENTS,
            "event_turnover_cap": EVENT_TURNOVER_CAP,
            "logging": "trigger_reason is written to decision outputs",
        },
        "system_modes": SYSTEM_MODES,
        "benchmark": {
            "portfolio_name": "SPY_benchmark",
            "rule": "buy-and-hold SPY paper account",
            "initial_capital": INITIAL_CAPITAL,
        },
        "market_state_today": clean_market_row,
    }
    write_json(os.path.join(OUT_DIR, "prototype_system_config.json"), payload)


def save_excel(tables):
    try:
        import openpyxl  # noqa: F401
    except Exception:
        return
    with pd.ExcelWriter(os.path.join(OUT_DIR, "decision_support_prototype_summary.xlsx"), engine="openpyxl") as writer:
        for name, table in tables.items():
            table.to_excel(writer, sheet_name=name[:31], index=False)


# =============================================================================
# Main
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", default=None, help="Completed market date to use as signal date cutoff.")
    parser.add_argument("--force-download", action="store_true", help="Refresh Yahoo Finance data cache.")
    parser.add_argument("--reset-paper", action="store_true", help="Reset forward paper-trading accounts.")
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_dirs()
    if args.reset_paper and os.path.exists(FORWARD_DIR):
        shutil.rmtree(FORWARD_DIR)
        os.makedirs(FORWARD_DIR, exist_ok=True)

    print("=" * 78)
    print("Forward Decision-Support Prototype and Paper Trading")
    print(f"Output: {OUT_DIR}")
    print("=" * 78)

    as_of = args.as_of or str(latest_completed_us_session_date())
    prices, macro = load_market_data(force_download=args.force_download, as_of=as_of)
    prices, macro = apply_as_of(prices, macro, as_of)

    coverage = prices[prices["ticker"].isin(ALL_STOCKS)].groupby("date")["ticker"].nunique()
    min_coverage = int(np.ceil(MIN_COVERAGE_RATIO * len(ALL_STOCKS)))
    valid_dates = coverage[coverage.ge(min_coverage)].index
    if len(valid_dates) == 0:
        raise RuntimeError("No date has sufficient stock universe coverage.")
    signal_date = pd.Timestamp(valid_dates.max())
    execution_date = next_business_day(signal_date)
    print(f"Signal date: {signal_date.date()} | Execution date: {execution_date.date()}")

    print("Building standalone feature panel ...", flush=True)
    panel = build_panel(prices, macro)
    latest_scores = build_latest_signals(panel, signal_date)
    market_row = build_market_indicators(prices, macro, signal_date)

    latest_scores.to_csv(os.path.join(OUT_DIR, "latest_signal_scores.csv"), index=False)
    latest_scores[latest_scores["signal_name"].eq("ensemble_equal_rank")].to_csv(
        os.path.join(OUT_DIR, "latest_ensemble_scores.csv"), index=False
    )
    pd.DataFrame([market_row]).to_csv(os.path.join(OUT_DIR, "latest_market_state.csv"), index=False)

    status_rows = []
    decision_frames = []
    target_frames = []
    all_signal_names = [x["signal_name"] for x in CANDIDATE_SIGNALS] + ["ensemble_equal_rank"]

    for signal_name in all_signal_names:
        signal_scores = latest_scores[latest_scores["signal_name"].eq(signal_name)].copy()
        for system_mode in SYSTEM_MODES:
            portfolio_name = f"{signal_name}_{system_mode}"
            path = portfolio_dir(portfolio_name)
            account = load_account(path, portfolio_name, signal_name, system_mode)
            account = process_pending_decisions(account, path, prices)
            base_weights, basket_reconstituted, event_info = base_stock_weights(signal_scores, account, signal_date)
            exposure = exposure_for_mode(system_mode, market_row["market_state"])
            target_weights = build_target_weights(base_weights, exposure)
            previous = account.get("target_weights", {}) or {}
            previous_exposure = sum(float(weight) for weight in previous.values())
            exposure_update = bool(
                previous
                and system_mode == "conservative"
                and abs(exposure - previous_exposure) > 0.005
            )
            risk_state_changed = bool(account.get("last_market_state") and account.get("last_market_state") != market_row["market_state"])
            trigger_reasons = []
            if basket_reconstituted:
                trigger_reasons.append("scheduled_weekly_reconstitution")
            if exposure_update:
                trigger_reasons.append("risk_exposure_update")
            if event_info["partial_update_triggered"]:
                trigger_reasons.append("partial_stock_update")
            if risk_state_changed:
                trigger_reasons.append("market_state_changed")
            if not trigger_reasons:
                trigger_reasons.append("hold_no_trigger")
            decision_type = (
                "REBALANCE"
                if basket_reconstituted or exposure_update or event_info["partial_update_triggered"]
                else "HOLD"
            )
            meta = {
                "portfolio_name": portfolio_name,
                "signal_name": signal_name,
                "system_mode": system_mode,
                "risk_overlay": SYSTEM_MODES[system_mode],
                "data_as_of": str(signal_date.date()),
                "execution_date": str(execution_date.date()),
                "decision": decision_type,
                "trigger_reason": ";".join(trigger_reasons),
                "market_state": market_row["market_state"],
                "risk_score": market_row["risk_score"],
                "stock_exposure": exposure,
                "basket_reconstituted": basket_reconstituted,
                "risk_exposure_update": exposure_update,
                "risk_state_changed": risk_state_changed,
                "partial_update_triggered": event_info["partial_update_triggered"],
                "partial_replacements": event_info["partial_replacements"],
                "partial_update_turnover": event_info["partial_update_turnover"],
                "partial_update_note": event_info["partial_update_note"],
                "event_turnover_cap": EVENT_TURNOVER_CAP,
                "max_event_replacements": MAX_EVENT_REPLACEMENTS,
            }
            decision = build_decision_rows(signal_scores, target_weights, previous, meta)
            decision["base_stock_weight"] = decision["ticker"].map(base_weights).fillna(0.0)
            status = update_paper_account(
                portfolio_name,
                signal_name,
                system_mode,
                decision,
                prices,
                signal_date,
                execution_date,
                reset=False,
            )
            status_rows.append(status)
            decision_frames.append(decision)
            target_frames.append(decision[decision["target_weight"].gt(EPS)].copy())

    spy_status, spy_decision = build_spy_benchmark_account(prices, signal_date, execution_date)
    status_rows.append(spy_status)
    decision_frames.append(spy_decision)
    target_frames.append(spy_decision[spy_decision["target_weight"].gt(EPS)].copy())

    status = pd.DataFrame(status_rows)
    decisions = pd.concat(decision_frames, ignore_index=True)
    targets = pd.concat(target_frames, ignore_index=True)
    status.to_csv(os.path.join(OUT_DIR, "portfolio_status_summary.csv"), index=False)
    decisions.to_csv(os.path.join(OUT_DIR, "latest_decisions.csv"), index=False)
    targets.to_csv(os.path.join(OUT_DIR, "latest_target_weights.csv"), index=False)
    pd.DataFrame(CANDIDATE_SIGNALS).to_csv(os.path.join(OUT_DIR, "candidate_signals_used.csv"), index=False)
    combined_nav = plot_nav_curve(status)
    save_config(signal_date, execution_date, market_row)
    save_readme()
    save_excel(
        {
            "Status": status,
            "Latest Decisions": decisions,
            "Latest Targets": targets,
            "Market State": pd.DataFrame([market_row]),
            "Signal Scores": latest_scores,
            "Combined NAV": combined_nav,
        }
    )

    print("\n=== Portfolio status summary ===")
    print(
        status[
            [
                "portfolio_name",
                "execution_status",
                "portfolio_value",
                "cumulative_return",
                "cash_weight",
                "n_holdings",
                "n_trades",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
