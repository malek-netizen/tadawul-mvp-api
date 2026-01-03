from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import requests
import pandas as pd
import time
import joblib
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

app = FastAPI(title="Tadawul MVP API", version="1.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "model.joblib"
TICKERS_PATH = "tickers_sa.txt"

model = None           # will hold pipeline
feature_cols = None
meta = {"threshold": 0.60}

# ----------------------------
# Yahoo Fetch
# ----------------------------
def fetch_yahoo_prices(ticker: str, range_: str = "1y", interval: str = "1d", min_rows: int = 60):
    bases = [
        "https://query1.finance.yahoo.com/v8/finance/chart/",
        "https://query2.finance.yahoo.com/v8/finance/chart/",
    ]
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
        "Accept-Language": "en-US,en;q=0.9",
    }

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
                    return df
            except Exception:
                continue

        time.sleep(1)

    return None

# ----------------------------
# Indicators
# ----------------------------
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
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1
    ).max(axis=1)
    return tr.rolling(period).mean()

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    close = d["close"]
    high = d["high"]
    low = d["low"]
    vol = d["volume"].fillna(0)

    d["sma10"] = sma(close, 10)
    d["sma20"] = sma(close, 20)
    d["sma50"] = sma(close, 50)

    d["ema10"] = ema(close, 10)
    d["ema20"] = ema(close, 20)
    d["ema50"] = ema(close, 50)

    d["rsi14"] = rsi(close, 14)

    macd_line, signal_line, hist = macd(close, 12, 26, 9)
    d["macd"] = macd_line
    d["macd_signal"] = signal_line
    d["macd_hist"] = hist

    bb_u, bb_m, bb_l = bollinger(close, 20, 2)
    d["bb_upper"] = bb_u
    d["bb_mid"] = bb_m
    d["bb_lower"] = bb_l
    d["bb_width"] = (bb_u - bb_l) / (bb_m + 1e-9)

    d["atr14"] = atr(high, low, close, 14)

    d["ret1"] = close.pct_change(1)
    d["ret5"] = close.pct_change(5)
    d["vol20"] = d["ret1"].rolling(20).std()

    d["vol_ma20"] = vol.rolling(20).mean()
    d["vol_ratio"] = vol / (d["vol_ma20"] + 1e-9)

    return d.dropna().reset_index(drop=True)

DEFAULT_FEATURE_COLS = [
    "sma10","sma20","sma50",
    "ema10","ema20","ema50",
    "rsi14",
    "macd","macd_signal","macd_hist",
    "bb_width",
    "atr14",
    "ret1","ret5","vol20",
    "vol_ratio",
]

def latest_feature_vector(feat_df: pd.DataFrame):
    cols = feature_cols or DEFAULT_FEATURE_COLS
    row = feat_df.iloc[-1][cols].astype(float)
    X = row.values.reshape(1, -1)
    return X

# ----------------------------
# Load model at startup
# ----------------------------
@app.on_event("startup")
def load_model():
    global model, feature_cols, meta
    model = None
    feature_cols = None
    meta = {"threshold": 0.60}

    if not os.path.exists(MODEL_PATH):
        return

    try:
        payload = joblib.load(MODEL_PATH)
        # payload can be either pipeline directly OR dict with keys
        if isinstance(payload, dict) and "pipeline" in payload:
            model = payload["pipeline"]
            feature_cols = payload.get("feature_cols")
            meta = payload.get("meta", meta)
        else:
            model = payload  # backward compat
    except Exception:
        model = None

# ----------------------------
# Helpers
# ----------------------------
def normalize_ticker(t: str):
    t = (t or "").strip().upper()
    if not t.endswith(".SR"):
        t += ".SR"
    return t

def rule_fallback(feat_df: pd.DataFrame):
    entry = float(feat_df["close"].iloc[-1])
    r = float(feat_df["rsi14"].iloc[-1])
    mh = float(feat_df["macd_hist"].iloc[-1])
    ema20_v = float(feat_df["ema20"].iloc[-1])
    buy = (r < 70) and (mh > 0) and (entry > ema20_v)
    prob = 0.55 if buy else 0.45
    return buy, prob

def analyze_one(ticker: str):
    t = normalize_ticker(ticker)

    df = fetch_yahoo_prices(t, range_="1y", interval="1d", min_rows=80)
    if df is None:
        return {"ticker": t, "recommendation": "NO_TRADE", "confidence": 0.0, "reason": "No price data returned from provider."}

    feat_df = build_features(df)
    if len(feat_df) < 30:
        return {"ticker": t, "recommendation": "NO_TRADE", "confidence": 0.0, "reason": "Not enough data after feature engineering."}

    last_close = float(feat_df["close"].iloc[-1])
    entry = last_close
    take_profit = entry * 1.05
    stop_loss = entry * 0.98

    threshold = float(meta.get("threshold", 0.60))

    if model is None:
        buy, prob = rule_fallback(feat_df)
        return {
            "ticker": t,
            "recommendation": "BUY" if buy else "NO_TRADE",
            "confidence": round(float(prob), 4),
            "entry": round(entry, 4),
            "take_profit": round(take_profit, 4),
            "stop_loss": round(stop_loss, 4),
            "reason": "Fallback rule-based (model not loaded).",
            "last_close": round(last_close, 4),
        }

    X = latest_feature_vector(feat_df)
    prob = float(model.predict_proba(X)[0, 1])
    rec = "BUY" if prob >= threshold else "NO_TRADE"
    reason = "" if rec == "BUY" else f"Probability below threshold ({threshold})"

    return {
        "ticker": t,
        "recommendation": rec,
        "confidence": round(prob, 4),
        "entry": round(entry, 4),
        "take_profit": round(take_profit, 4),
        "stop_loss": round(stop_loss, 4),
        "reason": reason,
        "last_close": round(last_close, 4),
    }

def load_tickers_from_file(path: str):
    if not os.path.exists(path):
        return []
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip().upper()
            if not t:
                continue
            if not t.endswith(".SR"):
                t += ".SR"
            items.append(t)
    seen=set()
    out=[]
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

# ----------------------------
# Endpoints
# ----------------------------
@app.get("/health")
def health():
    return {"status": "OK", "model_loaded": bool(model), "version": "v1-ai", "threshold": meta.get("threshold", 0.60)}

@app.get("/model_info")
def model_info():
    return {"model_loaded": bool(model), "feature_cols": feature_cols or DEFAULT_FEATURE_COLS, "meta": meta}

@app.get("/tickers_status")
def tickers_status():
    exists = os.path.exists(TICKERS_PATH)
    count = 0
    sample = []
    if exists:
        tickers = load_tickers_from_file(TICKERS_PATH)
        count = len(tickers)
        sample = tickers[:10]
    return {"exists": exists, "path": os.path.abspath(TICKERS_PATH), "count": count, "sample": sample}

@app.get("/predict")
def predict(ticker: str):
    return analyze_one(ticker)

@app.get("/top10")
def top10(max_workers: int = 10):
    tickers = load_tickers_from_file(TICKERS_PATH)
    if not tickers:
        return {"items": [], "error": f"Tickers file not found or empty: {TICKERS_PATH}"}

    results = []
    errors = 0

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(analyze_one, t): t for t in tickers}
        for fut in as_completed(futures):
            try:
                results.append(fut.result())
            except Exception:
                errors += 1

    # BUY first then confidence desc
    def score(item):
        rec = item.get("recommendation", "NO_TRADE")
        conf = float(item.get("confidence", 0.0) or 0.0)
        return (1 if rec == "BUY" else 0, conf)

    results_sorted = sorted(results, key=score, reverse=True)
    return {
        "items": results_sorted[:10],
        "total": len(results),
        "errors": errors,
        "source": "tickers_sa.txt",
        "note": "Sorted by (BUY first) then confidence desc"
    }
