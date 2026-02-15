from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import requests
import pandas as pd
import numpy as np
import time
import joblib
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# =========================
# Tadawul MVP API (Rules v1.2 + Pressure Filter)
# =========================

app = FastAPI(title="Tadawul MVP API", version="2.3.0-rules-v1.2-pressure")

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

# =========================
# Cache للأسعار
# =========================
_prices_cache = {}
CACHE_TTL_SEC = 600

def _cache_get(key):
    v = _prices_cache.get(key)
    if not v:
        return None
    if time.time() - v["ts"] > CACHE_TTL_SEC:
        _prices_cache.pop(key, None)
        return None
    return v["df"]

def _cache_set(key, df):
    _prices_cache[key] = {"ts": time.time(), "df": df}

_top10_cache = {"ts": 0, "data": None}
TOP10_CACHE_TTL_SEC = 600

# =========================
# جلب الأسعار من Yahoo
# =========================
def fetch_yahoo_prices(ticker: str, range_: str = "1y", interval: str = "1d", min_rows: int = 80):
    key = (ticker, range_, interval)
    cached = _cache_get(key)
    if cached is not None:
        return cached

    bases = [
        "https://query1.finance.yahoo.com/v8/finance/chart/",
        "https://query2.finance.yahoo.com/v8/finance/chart/",
    ]
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json", "Accept-Language": "en-US,en;q=0.9"}
    for _ in range(3):
        for base in bases:
            try:
                url = f"{base}{ticker}"
                r = requests.get(url, params={"range": range_, "interval": interval}, headers=headers, timeout=20)
                if r.status_code != 200:
                    continue
                js = r.json()
                chart = js.get("chart", {})
                if chart.get("error"):
                    continue
                result = (chart.get("result") or [None])[0]
                if not result:
                    continue
                quote = (result.get("indicators", {}).get("quote") or [None])[0]
                if not quote:
                    continue
                df = pd.DataFrame({
                    "open": quote.get("open", []),
                    "high": quote.get("high", []),
                    "low": quote.get("low", []),
                    "close": quote.get("close", []),
                    "volume": quote.get("volume", []),
                }).dropna(subset=["close"]).reset_index(drop=True)
                if len(df) >= min_rows:
                    _cache_set(key, df)
                    return df
            except Exception:
                continue
        time.sleep(1)
    return None

# =========================
# مؤشرات فنية
# =========================
def sma(series, window): return series.rolling(window).mean()
def ema(series, span): return series.ewm(span=span, adjust=False).mean()
def rsi(close, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))

def macd(close, fast=12, slow=26, signal=9):
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def bollinger(close, window=20, num_std=2):
    mid = sma(close, window)
    std = close.rolling(window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower

def atr(high, low, close, period=14):
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    close = d["close"]; high = d["high"]; low = d["low"]; vol = d["volume"].fillna(0)
    d["sma10"] = sma(close, 10); d["sma20"] = sma(close, 20); d["sma50"] = sma(close, 50)
    d["ema10"] = ema(close, 10); d["ema20"] = ema(close, 20); d["ema50"] = ema(close, 50)
    d["rsi14"] = rsi(close, 14)
    macd_line, signal_line, hist = macd(close, 12, 26, 9)
    d["macd"] = macd_line; d["macd_signal"] = signal_line; d["macd_hist"] = hist
    bb_u, bb_m, bb_l = bollinger(close, 20, 2)
    d["bb_upper"] = bb_u; d["bb_mid"] = bb_m; d["bb_lower"] = bb_l; d["bb_width"] = (bb_u - bb_l) / (bb_m + 1e-9)
    d["atr14"] = atr(high, low, close, 14)
    d["ret1"] = close.pct_change(1); d["ret3"] = close.pct_change(3); d["ret5"] = close.pct_change(5)
    d["vol20"] = d["ret1"].rolling(20).std()
    d["vol_ma20"] = vol.rolling(20).mean()
    d["vol_ratio"] = vol / (d["vol_ma20"] + 1e-9)
    return d.dropna().reset_index(drop=True)

FEATURE_COLS = ["sma10","sma20","sma50","ema10","ema20","ema50","rsi14","macd","macd_signal","macd_hist","bb_width","atr14","ret1","ret3","ret5","vol20","vol_ratio"]

def latest_feature_vector(feat_df: pd.DataFrame):
    row = feat_df.iloc[-1][FEATURE_COLS].astype(float)
    X = row.values.reshape(1, -1)
    return X, row.to_dict()

# =========================
# تحميل النموذج
# =========================
@app.on_event("startup")
def load_model():
    global model
    model = None
    if os.path.exists(MODEL_PATH):
        try: model = joblib.load(MODEL_PATH)
        except Exception: model = None

# =========================
# تحليل سهم واحد
# =========================
def analyze_one(ticker: str):
    t = (ticker or "").strip().upper()
    if not t.endswith(".SR"): t += ".SR"
    df = fetch_yahoo_prices(t)
    if df is None:
        return {"ticker": t, "recommendation": "NO_TRADE", "reason": "No price data."}
    feat_df = build_features(df)
    if len(feat_df) < 10:
        return {"ticker": t, "recommendation": "NO_TRADE", "reason": "Not enough data."}
    last_close = float(feat_df["close"].iloc[-1])
    return {"ticker": t, "recommendation": "BUY", "last_close": round(last_close,4)}

# =========================
# Endpoints
# =========================
@app.get("/health")
def health():
    return {"status": "OK","model_loaded": bool(model)}

@app.get("/predict")
def predict(ticker: str):
    return analyze_one(ticker)

@app.get("/debug_prices")
def debug_prices(ticker: str):
    t = ticker.strip().upper()
    if not t.endswith(".SR"): t += ".SR"
    df = fetch_yahoo_prices(t)
    if df is None: return {"ticker": t, "ok": False, "rows":0}
    return {"ticker": t, "ok": True, "rows": len(df), "last_close": float(df["close"].iloc[-1])}
