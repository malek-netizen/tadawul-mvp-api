from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import requests
import pandas as pd
import numpy as np
import time
import joblib
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

app = FastAPI(title="Tadawul MVP API", version="2.3.0-rules-v1.3-enhanced")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "model.joblib"
TICKERS_PATH = "tickers_sa.txt"

TP_PCT = 0.05
SL_PCT = 0.02

RSI_MIN = 40
RSI_MAX = 65
RSI_OVERBOUGHT = 70

MAX_ABOVE_EMA20 = 0.06
MAX_ABOVE_EMA50 = 0.08

SPIKE_RET1 = 0.08
CONSOL_DAYS = 4
CONSOL_RANGE_MAX = 0.06

LOOKBACK_LOW_DAYS = 10
MIN_TREND_SCORE = 2
ML_THRESHOLD = 0.60
TOP10_WORKERS = 6

model = None

# ===============================
# FETCH DATA
# ===============================

def fetch_yahoo_prices(ticker: str, range_: str = "1y", interval: str = "1d"):
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        r = requests.get(url, params={"range": range_, "interval": interval}, timeout=20)
        if r.status_code != 200:
            return None
        js = r.json()
        result = js["chart"]["result"][0]
        quote = result["indicators"]["quote"][0]

        df = pd.DataFrame({
            "open": quote["open"],
            "high": quote["high"],
            "low": quote["low"],
            "close": quote["close"],
            "volume": quote["volume"],
        })

        df = df.dropna(subset=["close"]).reset_index(drop=True)
        return df
    except:
        return None


# ===============================
# INDICATORS
# ===============================

def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def rsi(close, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))

def build_features(df):
    d = df.copy()
    close = d["close"]

    d["ema20"] = ema(close, 20)
    d["ema50"] = ema(close, 50)
    d["rsi14"] = rsi(close, 14)

    d["ret1"] = close.pct_change(1)
    d["ret3"] = close.pct_change(3)
    d["ret5"] = close.pct_change(5)

    d = d.dropna().reset_index(drop=True)
    return d


# ===============================
# === NEW LOGIC ADDED ===
# ===============================

def ema_slope_up(feat_df: pd.DataFrame, col: str, lookback: int = 5) -> bool:
    if len(feat_df) < lookback + 1:
        return False
    x = feat_df[col].tail(lookback).values
    return float(x[-1]) > float(x[0])

def breakout_entry(df_ohlc: pd.DataFrame, days: int = CONSOL_DAYS):
    if len(df_ohlc) < days + 5:
        return None
    base = df_ohlc.tail(days)
    return float(base["high"].max()) * 1.002


def is_consolidating(feat_df: pd.DataFrame, days: int = CONSOL_DAYS) -> bool:
    if len(feat_df) < days + 5:
        return False
    last = feat_df.tail(days)
    hi = float(last["high"].max())
    lo = float(last["low"].min())
    mid = float(last["close"].mean())
    if mid <= 0:
        return False
    rng = (hi - lo) / mid
    return rng <= CONSOL_RANGE_MAX


# ===============================
# RULES (ENHANCED)
# ===============================

def passes_rules(df_ohlc, feat_df):
    reasons = []
    last = feat_df.iloc[-1]

    close = float(last["close"])
    ema20 = float(last["ema20"])
    ema50 = float(last["ema50"])
    rsi14 = float(last["rsi14"])

    # === Regime filter (NEW) ===
    if close < ema50:
        reasons.append("Rejected: below EMA50")

    if not ema_slope_up(feat_df, "ema20", 5):
        reasons.append("Rejected: EMA20 slope down")

    if not ema_slope_up(feat_df, "ema50", 8):
        reasons.append("Rejected: EMA50 slope down")

    # === Momentum confirmation (NEW) ===
    if float(last["ret3"]) < 0:
        reasons.append("Rejected: negative 3-day momentum")

    if float(last["ret1"]) < -0.015:
        reasons.append("Rejected: strong daily drop")

    # RSI band
    if not (RSI_MIN <= rsi14 <= RSI_MAX):
        reasons.append("Rejected: RSI not in optimal band")

    return (len(reasons) == 0), reasons


# ===============================
# ANALYZE
# ===============================

def analyze_one(ticker: str):

    t = ticker.strip().upper()
    if not t.endswith(".SR"):
        t += ".SR"

    df = fetch_yahoo_prices(t)
    if df is None:
        return {"ticker": t, "recommendation": "NO_TRADE"}

    feat_df = build_features(df)
    if len(feat_df) < 60:
        return {"ticker": t, "recommendation": "NO_TRADE"}

    last_close = float(feat_df["close"].iloc[-1])

    ok, reasons = passes_rules(df, feat_df)

    # === Smart Entry (NEW) ===
    entry = last_close
    if is_consolidating(feat_df):
        be = breakout_entry(df)
        if be:
            entry = be
    else:
        entry = float(feat_df["ema20"].iloc[-1])

    take_profit = entry * (1 + TP_PCT)
    stop_loss = entry * (1 - SL_PCT)

    recommendation = "BUY" if ok else "NO_TRADE"

    return {
        "ticker": t,
        "recommendation": recommendation,
        "entry": round(entry, 4),
        "take_profit": round(take_profit, 4),
        "stop_loss": round(stop_loss, 4),
        "reason": " | ".join(reasons) if reasons else "Approved",
        "last_close": round(last_close, 4),
    }


@app.get("/predict")
def predict(ticker: str):
    return analyze_one(ticker)
