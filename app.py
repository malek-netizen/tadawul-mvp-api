from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import requests
import pandas as pd
import time
import joblib
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import threading

# =========================
# Tadawul MVP API (Rules v1.0) - Direct Top10
# هدف: +5% | وقف: -2% | تاسي فقط
# =========================

app = FastAPI(title="Tadawul MVP API", version="2.0.1-top10-direct")

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
# إعدادات النظام
# =========================
TP_PCT = 0.05
SL_PCT = 0.02

# فلترة RSI
RSI_MIN = 40
RSI_MAX = 65
RSI_OVERBOUGHT = 70

# قرب السعر من المتوسطات
MAX_ABOVE_EMA20 = 0.06
MAX_ABOVE_EMA50 = 0.08

# فلترة الاندفاع
SPIKE_RET1 = 0.08
CONSOL_DAYS = 4
CONSOL_RANGE_MAX = 0.06

# فلترة الهبوط/كسر قاع
LOOKBACK_LOW_DAYS = 10
MIN_TREND_SCORE = 2

# =========================
# Cache (لتقليل الضغط والتعليق)
# =========================
CACHE_LOCK = threading.Lock()
CACHE = {
    "top10": None,
    "expires_at": None,
}

CACHE_TTL_MIN = 15  # 15 دقيقة

def cache_get():
    with CACHE_LOCK:
        if CACHE["top10"] is None or CACHE["expires_at"] is None:
            return None
        if datetime.utcnow() >= CACHE["expires_at"]:
            return None
        return CACHE["top10"]

def cache_set(value):
    with CACHE_LOCK:
        CACHE["top10"] = value
        CACHE["expires_at"] = datetime.utcnow() + timedelta(minutes=CACHE_TTL_MIN)

# =========================
# 1) جلب الأسعار من Yahoo Chart
# =========================
def fetch_yahoo_prices(ticker: str, range_: str = "1y", interval: str = "1d", min_rows: int = 80):
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
                if len(df) >= min_rows:
                    return df

            except Exception:
                continue
        time.sleep(1)

    return None

# =========================
# 2) مؤشرات فنية (بدون مكتبات إضافية)
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
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
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
    d["ret3"] = close.pct_change(3)
    d["ret5"] = close.pct_change(5)
    d["vol20"] = d["ret1"].rolling(20).std()

    d["vol_ma20"] = vol.rolling(20).mean()
    d["vol_ratio"] = vol / (d["vol_ma20"] + 1e-9)

    d = d.dropna().reset_index(drop=True)
    return d

FEATURE_COLS = [
    "sma10", "sma20", "sma50",
    "ema10", "ema20", "ema50",
    "rsi14",
    "macd", "macd_signal", "macd_hist",
    "bb_width",
    "atr14",
    "ret1", "ret3", "ret5", "vol20",
    "vol_ratio",
]

def latest_feature_vector(feat_df: pd.DataFrame):
    row = feat_df.iloc[-1][FEATURE_COLS].astype(float)
    X = row.values.reshape(1, -1)
    return X

# =========================
# 3) تحميل النموذج
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
# 4) قواعد الدخول
# =========================
def is_consolidating(d: pd.DataFrame, days: int = CONSOL_DAYS) -> bool:
    if len(d) < days + 5:
        return False
    last = d.tail(days)
    hi = float(last["high"].max())
    lo = float(last["low"].min())
    mid = float(last["close"].mean())
    if mid <= 0:
        return False
    rng = (hi - lo) / mid
    return rng <= CONSOL_RANGE_MAX

def trend_score(d: pd.DataFrame) -> int:
    last = d.iloc[-1]
    score = 0
    close = float(last["close"])
    ema20 = float(last["ema20"])
    ema50 = float(last["ema50"])
    r5 = float(last["ret5"])
    mh = float(last["macd_hist"])

    if close >= ema20:
        score += 1
    if ema20 >= ema50:
        score += 1
    if r5 > -0.03:
        score += 1
    if mh >= -0.001:
        score += 1
    return score

def broke_recent_low(df_ohlc: pd.DataFrame, lookback: int = LOOKBACK_LOW_DAYS) -> bool:
    if len(df_ohlc) < lookback + 5:
        return False
    recent = df_ohlc.tail(lookback)
    recent_low = float(recent["low"].min())
    last_close = float(df_ohlc["close"].iloc[-1])
    return last_close < recent_low * 0.995

def spike_without_base(d: pd.DataFrame) -> bool:
    last = d.iloc[-1]
    ret1 = float(last["ret1"])
    return bool(ret1 >= SPIKE_RET1 and (not is_consolidating(d, CONSOL_DAYS)))

def risk_check_stop(df_ohlc: pd.DataFrame, stop: float) -> bool:
    if len(df_ohlc) < 10:
        return False
    recent_low = float(df_ohlc.tail(5)["low"].min())
    return recent_low >= stop

def passes_rules(df_ohlc: pd.DataFrame, feat: pd.DataFrame):
    reasons = []
    d = feat
    last = d.iloc[-1]
    close = float(last["close"])
    ema20 = float(last["ema20"])
    ema50 = float(last["ema50"])
    rsi14 = float(last["rsi14"])

    if broke_recent_low(df_ohlc, LOOKBACK_LOW_DAYS):
        reasons.append("Rejected: recent low broken.")

    ts = trend_score(d)
    if ts < MIN_TREND_SCORE:
        reasons.append("Rejected: weak trend context.")

    if spike_without_base(d):
        reasons.append("Rejected: spike without base.")

    if rsi14 > RSI_OVERBOUGHT:
        reasons.append("Rejected: RSI overbought (>70).")

    if not (RSI_MIN <= rsi14 <= RSI_MAX):
        if not (rsi14 >= 35 and is_consolidating(d, CONSOL_DAYS)):
            reasons.append("Rejected: RSI not in safe band and no base exception.")

    if ema20 > 0 and (close - ema20) / ema20 > MAX_ABOVE_EMA20:
        reasons.append("Rejected: overextended above EMA20.")
    if ema50 > 0 and (close - ema50) / ema50 > MAX_ABOVE_EMA50:
        reasons.append("Rejected: overextended above EMA50.")

    if len(d) >= 4:
        h = d["macd_hist"].tail(4).values
        if not (h[-1] >= h[-2] or h[-1] >= 0):
            reasons.append("Rejected: MACD hist not improving.")

    base_ok = is_consolidating(d, CONSOL_DAYS)
    if not base_ok and ts < 3:
        reasons.append("Rejected: no consolidation/base.")

    ok = (len(reasons) == 0)
    return ok, reasons

# =========================
# 5) تحليل سهم واحد
# =========================
def analyze_one(ticker: str):
    t = (ticker or "").strip().upper()
    if not t:
        return {"error": "Ticker is required."}
    if not t.endswith(".SR"):
        t += ".SR"

    df = fetch_yahoo_prices(t, range_="1y", interval="1d")
    if df is None:
        return {"ticker": t, "recommendation": "NO_TRADE", "confidence": 0.0, "reason": "No price data."}

    feat_df = build_features(df)
    if len(feat_df) < 10:
        return {"ticker": t, "recommendation": "NO_TRADE", "confidence": 0.0, "reason": "Not enough data."}

    entry = float(feat_df["close"].iloc[-1])
    take_profit = entry * (1.0 + TP_PCT)
    stop_loss = entry * (1.0 - SL_PCT)

    ok, reasons = passes_rules(df, feat_df)
    if ok and (not risk_check_stop(df, stop_loss)):
        ok = False
        reasons.append("Rejected: stop too tight vs recent lows.")

    conf = 0.0
    if model is not None:
        try:
            X = latest_feature_vector(feat_df)
            conf = float(model.predict_proba(X)[0, 1])
        except Exception:
            conf = 0.0
    else:
        # Rule-based confidence (تقريبي)
        score = 0
        last = feat_df.iloc[-1]
        score += 1 if float(last["close"]) >= float(last["ema20"]) else 0
        score += 1 if float(last["ema20"]) >= float(last["ema50"]) else 0
        score += 1 if (RSI_MIN <= float(last["rsi14"]) <= RSI_MAX) else 0
        score += 1 if float(last["macd_hist"]) >= 0 else 0
        score += 1 if is_consolidating(feat_df, CONSOL_DAYS) else 0
        conf = min(0.50 + 0.08 * score, 0.85)

    recommendation = "BUY" if (ok and conf >= 0.60) else "NO_TRADE"
    if recommendation == "BUY":
        reason = "Top10: Rules+Model/Score confirmed."
    else:
        reason = " | ".join(reasons[:3]) if reasons else "Probability below threshold"

    return {
        "ticker": t,
        "recommendation": recommendation,
        "confidence": round(conf, 4),
        "entry": round(entry, 4),
        "take_profit": round(take_profit, 4),
        "stop_loss": round(stop_loss, 4),
        "reason": reason,
        "last_close": round(entry, 4),
    }

# =========================
# 6) Tickers
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
            if not t.endswith(".SR"):
                t += ".SR"
            items.append(t)

    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

# =========================
# 7) Endpoints
# =========================
@app.get("/health")
def health():
    return {
        "status": "OK",
        "model_loaded": bool(model),
        "rules_version": "v1.0",
        "tp_pct": TP_PCT,
        "sl_pct": SL_PCT,
        "tickers_file": TICKERS_PATH,
        "cache_ttl_min": CACHE_TTL_MIN,
    }

@app.get("/predict")
def predict(ticker: str):
    return analyze_one(ticker)

@app.get("/check_tickers")
def check_tickers():
    path = os.path.abspath(TICKERS_PATH)
    if not os.path.exists(TICKERS_PATH):
        return {"exists": False, "path": path, "count": 0, "sample": []}
    tickers = load_tickers_from_file(TICKERS_PATH)
    return {"exists": True, "path": path, "count": len(tickers), "sample": tickers[:10]}

@app.get("/top10")
def top10(max_workers: int = 8, force: int = 0):
    """
    يحلل السوق كامل (tickers_sa.txt) ويرجع أفضل 10 مباشرة.
    - cache 15 دقيقة لتسريع التجربة
    - force=1 لتجاوز الكاش
    """
    if force == 0:
        cached = cache_get()
        if cached is not None:
            return cached

    tickers = load_tickers_from_file(TICKERS_PATH)
    if not tickers:
        raise HTTPException(status_code=400, detail=f"Tickers file not found or empty: {TICKERS_PATH}")

    # حدود معقولة
    max_workers = int(max_workers)
    if max_workers < 2:
        max_workers = 2
    if max_workers > 12:
        max_workers = 12

    results = []
    errors = 0

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(analyze_one, t): t for t in tickers}
        for fut in as_completed(futures):
            try:
                results.append(fut.result())
            except Exception:
                errors += 1

    def score(item):
        rec = item.get("recommendation", "NO_TRADE")
        conf = float(item.get("confidence", 0.0) or 0.0)
        return (1 if rec == "BUY" else 0, conf)

    results_sorted = sorted(results, key=score, reverse=True)
    top_items = results_sorted[:10]

    payload = {
        "items": top_items,
        "total": len(results),
        "errors": errors,
        "source": TICKERS_PATH,
        "note": "Direct Top10. Cached for 15 min (use force=1 to bypass)."
    }

    cache_set(payload)
    return payload
