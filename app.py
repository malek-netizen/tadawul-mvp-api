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
# 1) إعداد FastAPI + CORS
# =========================
app = FastAPI(title="Tadawul MVP API", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "model.joblib"
TICKERS_PATH = "tickers_sa.txt"

model = None

# =========================
# 2) جلب الأسعار من Yahoo Chart (بديل yfinance)
# =========================
def fetch_yahoo_prices(ticker: str, range_: str = "1y", interval: str = "1d", min_rows: int = 60):
    """
    Fetch OHLCV data from Yahoo Finance chart endpoint with retries and fallback.
    Returns: DataFrame columns = open, high, low, close, volume
    """
    bases = [
        "https://query1.finance.yahoo.com/v8/finance/chart/",
        "https://query2.finance.yahoo.com/v8/finance/chart/",
    ]

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
        "Accept-Language": "en-US,en;q=0.9",
    }

    for attempt in range(3):
        for base in bases:
            try:
                url = f"{base}{ticker}"
                r = requests.get(
                    url,
                    params={"range": range_, "interval": interval},
                    headers=headers,
                    timeout=20,
                )
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

                df = pd.DataFrame(
                    {
                        "open": quote.get("open", []),
                        "high": quote.get("high", []),
                        "low": quote.get("low", []),
                        "close": quote.get("close", []),
                        "volume": quote.get("volume", []),
                    }
                )

                df = df.dropna(subset=["close"]).reset_index(drop=True)

                # ✅ كان 120 — صار 60 لتفادي فشل أسهم بياناتها أقل
                if len(df) >= min_rows:
                    return df

            except Exception:
                continue

        time.sleep(1)

    return None

# =========================
# 3) حساب المؤشرات الفنية (بدون مكتبات إضافية)
# =========================
def sma(series, window):
    return series.rolling(window).mean()

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
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
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

    d = d.dropna().reset_index(drop=True)
    return d

# =========================
# 4) تحويل آخر صف Features إلى Vector للنموذج
# =========================
FEATURE_COLS = [
    "sma10", "sma20", "sma50",
    "ema10", "ema20", "ema50",
    "rsi14",
    "macd", "macd_signal", "macd_hist",
    "bb_width",
    "atr14",
    "ret1", "ret5", "vol20",
    "vol_ratio",
]

def latest_feature_vector(feat_df: pd.DataFrame):
    row = feat_df.iloc[-1][FEATURE_COLS].astype(float)
    X = row.values.reshape(1, -1)
    return X, row.to_dict()

# =========================
# 5) تحميل النموذج عند بدء التشغيل
# =========================
@app.on_event("startup")
def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
        except Exception:
            model = None
    else:
        model = None

# =========================
# 6) Endpoints مساعدة
# =========================
@app.get("/health")
def health():
    return {"status": "OK", "model_loaded": bool(model), "version": "v1-full"}

@app.get("/debug_prices")
def debug_prices(ticker: str):
    t = ticker.strip().upper()
    if not t.endswith(".SR"):
        t += ".SR"
    df = fetch_yahoo_prices(t, range_="1y", interval="1d")
    if df is None:
        return {"ticker": t, "ok": False, "rows": 0}
    return {"ticker": t, "ok": True, "rows": int(len(df)), "last_close": float(df["close"].iloc[-1])}

# =========================
# 7) منطق تحليل سهم واحد (نفس /predict) — مفصول لإعادة الاستخدام
# =========================
def analyze_one(ticker: str):
    t = ticker.strip().upper()
    if not t.endswith(".SR"):
        t += ".SR"

    df = fetch_yahoo_prices(t, range_="1y", interval="1d")
    if df is None:
        return {
            "ticker": t,
            "recommendation": "NO_TRADE",
            "confidence": 0.0,
            "reason": "No price data returned from provider."
        }

    feat_df = build_features(df)
    if len(feat_df) < 5:
        return {
            "ticker": t,
            "recommendation": "NO_TRADE",
            "confidence": 0.0,
            "reason": "Not enough data after feature engineering."
        }

    last_close = float(feat_df["close"].iloc[-1])
    entry = last_close
    take_profit = entry * 1.05
    stop_loss = entry * 0.98

    # Fallback إذا النموذج غير موجود
    if model is None:
        r = float(feat_df["rsi14"].iloc[-1])
        mh = float(feat_df["macd_hist"].iloc[-1])
        ema20_v = float(feat_df["ema20"].iloc[-1])
        buy = (r < 70) and (mh > 0) and (entry > ema20_v)

        return {
            "ticker": t,
            "recommendation": "BUY" if buy else "NO_TRADE",
            "confidence": 0.55 if buy else 0.45,
            "entry": round(entry, 4),
            "take_profit": round(take_profit, 4),
            "stop_loss": round(stop_loss, 4),
            "reason": "Fallback rule-based (model.joblib not found).",
            "last_close": round(last_close, 4),
        }

    # نموذج ML
    X, _ = latest_feature_vector(feat_df)
    prob = float(model.predict_proba(X)[0, 1])  # احتمال BUY
    recommendation = "BUY" if prob >= 0.60 else "NO_TRADE"
    reason = "" if recommendation == "BUY" else "Probability below threshold"

    return {
        "ticker": t,
        "recommendation": recommendation,
        "confidence": round(prob, 4),
        "entry": round(entry, 4),
        "take_profit": round(take_profit, 4),
        "stop_loss": round(stop_loss, 4),
        "reason": reason,
        "last_close": round(last_close, 4),
    }

# =========================
# 8) Endpoint الرئيسي للتطبيق
# =========================
@app.get("/predict")
def predict(ticker: str):
    return analyze_one(ticker)

# =========================
# 9) Endpoint: Top 10
# =========================
def load_tickers_from_file(path: str):
    if not os.path.exists(path):
        return []
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip().upper()
            if not t:
                continue
            # نتأكد من .SR
            if not t.endswith(".SR"):
                t += ".SR"
            items.append(t)
    # إزالة التكرار مع الحفاظ على الترتيب
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

@app.get("/top10")
def top10(universe: str = "all", max_workers: int = 8):
    """
    يقرأ tickers_sa.txt ويحلل كل الأسهم (بشكل متوازي محدود)
    ويرجع أفضل 10 حسب confidence.
    """
    tickers = load_tickers_from_file(TICKERS_PATH)
    if not tickers:
        return {"items": [], "error": f"Tickers file not found or empty: {TICKERS_PATH}"}

    results = []
    errors = 0

    # تنفيذ متوازي محدود
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(analyze_one, t): t for t in tickers}
        for fut in as_completed(futures):
            try:
                r = fut.result()
                results.append(r)
            except Exception:
                errors += 1

    # نرتب: أولاً BUY ثم confidence الأعلى
    def score(item):
        rec = item.get("recommendation", "NO_TRADE")
        conf = float(item.get("confidence", 0.0) or 0.0)
        return (1 if rec == "BUY" else 0, conf)

    results_sorted = sorted(results, key=score, reverse=True)
    top_items = results_sorted[:10]

    return {
        "items": top_items,
        "total": len(results),
        "errors": errors,
        "source": "tickers_sa.txt",
        "note": "Sorted by (BUY first) then confidence desc"
    }
