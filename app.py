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
# Tadawul MVP API (Rules v1.2)
# هدف: +5% | وقف: -2% | تاسي فقط
# ML ALWAYS computed -> ml_confidence is NEVER null if model loaded
# =========================

app = FastAPI(title="Tadawul MVP API", version="2.2.0-rules-v1.2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "model.joblib"
TICKERS_PATH = "tickers_sa.txt"

# إعدادات ثابتة للصفقة
TP_PCT = 0.05
SL_PCT = 0.02

# حدود ومحددات قواعد الدخول (قابلة للتعديل)
RSI_MIN = 40
RSI_MAX = 65
RSI_OVERBOUGHT = 70

MAX_ABOVE_EMA20 = 0.06   # 6%
MAX_ABOVE_EMA50 = 0.08   # 8%

SPIKE_RET1 = 0.08        # 8% يوم واحد اندفاع
CONSOL_DAYS = 4
CONSOL_RANGE_MAX = 0.06  # 6% مدى

LOOKBACK_LOW_DAYS = 10
MIN_TREND_SCORE = 2

ML_THRESHOLD = 0.60  # عتبة قرار BUY من ML
TOP10_WORKERS = 6    # لتقليل تعليق top10 على Render

model = None

# =========================
# Cache بسيط لتقليل ضغط Yahoo أثناء /top10
# =========================
_prices_cache = {}  # key: (ticker, range_, interval) -> {"ts": ..., "df": ...}
CACHE_TTL_SEC = 60  # دقيقة واحدة

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

# =========================
# 1) جلب الأسعار من Yahoo Chart
# =========================
def fetch_yahoo_prices(ticker: str, range_: str = "1y", interval: str = "1d", min_rows: int = 80):
    """
    Returns DataFrame columns: open, high, low, close, volume
    """
    key = (ticker, range_, interval)
    cached = _cache_get(key)
    if cached is not None:
        return cached

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
                    _cache_set(key, df)
                    return df

            except Exception:
                continue

        time.sleep(1)

    return None

# =========================
# 2) مؤشرات فنية
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
    return X, row.to_dict()

# =========================
# 3) تحميل النموذج
# =========================
@app.on_event("startup")
def load_model():
    global model
    model = None
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
        except Exception:
            model = None

# =========================
# 4) قواعد الدخول
# =========================
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

def trend_score(feat_df: pd.DataFrame) -> int:
    last = feat_df.iloc[-1]
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

def spike_without_base(feat_df: pd.DataFrame) -> bool:
    last = feat_df.iloc[-1]
    ret1 = float(last["ret1"])
    if ret1 >= SPIKE_RET1 and (not is_consolidating(feat_df, CONSOL_DAYS)):
        return True
    return False

def risk_check_stop(df_ohlc: pd.DataFrame, stop: float) -> bool:
    if len(df_ohlc) < 10:
        return False
    recent = df_ohlc.tail(5)
    recent_low = float(recent["low"].min())
    # إذا قاع آخر 5 أيام أقل من وقفنا -> احتمال ضرب الوقف عالي
    if recent_low < stop:
        return False
    return True

def passes_rules(df_ohlc: pd.DataFrame, feat_df: pd.DataFrame):
    reasons = []
    last = feat_df.iloc[-1]

    close = float(last["close"])
    ema20 = float(last["ema20"])
    ema50 = float(last["ema50"])
    rsi14 = float(last["rsi14"])

    # 1) منع سياق هابط قوي
    if broke_recent_low(df_ohlc, LOOKBACK_LOW_DAYS):
        reasons.append("Rejected: recent low broken (bearish context).")

    ts = trend_score(feat_df)
    if ts < MIN_TREND_SCORE:
        reasons.append("Rejected: weak trend context (down/weak structure).")

    # 2) منع الاندفاع
    if spike_without_base(feat_df):
        reasons.append("Rejected: strong 1-day spike without consolidation (chasing).")

    # 3) RSI
    if rsi14 > RSI_OVERBOUGHT:
        reasons.append("Rejected: RSI overbought (>70).")

    if not (RSI_MIN <= rsi14 <= RSI_MAX):
        # استثناء: RSI بين 35-40 + تماسك
        if not (rsi14 >= 35 and is_consolidating(feat_df, CONSOL_DAYS)):
            reasons.append("Rejected: RSI not in safe band (40-65) and no consolidation exception.")

    # 4) مبالغة فوق المتوسطات
    if ema20 > 0 and (close - ema20) / ema20 > MAX_ABOVE_EMA20:
        reasons.append("Rejected: price too far above EMA20 (overextended).")
    if ema50 > 0 and (close - ema50) / ema50 > MAX_ABOVE_EMA50:
        reasons.append("Rejected: price too far above EMA50 (overextended).")

    # 5) MACD تحسن بسيط
    if len(feat_df) >= 4:
        h = feat_df["macd_hist"].tail(4).values
        if not (h[-1] >= h[-2] or h[-1] >= 0):
            reasons.append("Rejected: MACD histogram not improving.")

    # 6) قاعدة base/تماسك لتقليل الإشارات
    base_ok = is_consolidating(feat_df, CONSOL_DAYS)
    if not base_ok and ts < 3:
        reasons.append("Rejected: no clear consolidation/base before entry.")

    ok = (len(reasons) == 0)
    return ok, reasons

def rules_score(feat_df: pd.DataFrame) -> int:
    """
    score من 0 إلى 100 تقريبًا (ليس ضمان).
    نستخدمه كـ confidence_pct إذا ML غير متاح.
    """
    if len(feat_df) < 5:
        return 30
    last = feat_df.iloc[-1]
    score = 0

    # trend
    score += 15 if float(last["close"]) >= float(last["ema20"]) else 0
    score += 15 if float(last["ema20"]) >= float(last["ema50"]) else 0

    # RSI
    r = float(last["rsi14"])
    if RSI_MIN <= r <= RSI_MAX:
        score += 20
    elif 35 <= r < RSI_MIN:
        score += 10

    # MACD hist
    score += 15 if float(last["macd_hist"]) >= 0 else 5

    # base
    score += 20 if is_consolidating(feat_df, CONSOL_DAYS) else 5

    # overextension penalty
    close = float(last["close"])
    ema20 = float(last["ema20"])
    if ema20 > 0 and (close - ema20) / ema20 > MAX_ABOVE_EMA20:
        score -= 10

    score = max(0, min(100, score))
    return int(score)

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
        return {
            "ticker": t,
            "recommendation": "NO_TRADE",
            "status": "NO_DATA",
            "confidence_pct": 0,
            "ml_confidence": None,
            "rules_score": 0,
            "entry": None,
            "take_profit": None,
            "stop_loss": None,
            "reason": "No price data returned from provider.",
            "last_close": None,
        }

    feat_df = build_features(df)
    if len(feat_df) < 10:
        return {
            "ticker": t,
            "recommendation": "NO_TRADE",
            "status": "NO_DATA",
            "confidence_pct": 0,
            "ml_confidence": None,
            "rules_score": 0,
            "entry": None,
            "take_profit": None,
            "stop_loss": None,
            "reason": "Not enough data after feature engineering.",
            "last_close": None,
        }

    last_close = float(feat_df["close"].iloc[-1])
    entry = last_close
    take_profit = entry * (1.0 + TP_PCT)
    stop_loss = entry * (1.0 - SL_PCT)

    # RULES
    ok, reasons = passes_rules(df, feat_df)
    if ok and (not risk_check_stop(df, stop_loss)):
        ok = False
        reasons.append("Rejected: -2% stop is too tight vs recent lows.")

    r_score = rules_score(feat_df)

    # ML ALWAYS attempt if model exists
    ml_conf = None
    if model is not None:
        try:
            X, _ = latest_feature_vector(feat_df)
            ml_conf = float(model.predict_proba(X)[0, 1])
        except Exception:
            ml_conf = None

    # Decide final
    if not ok:
        recommendation = "NO_TRADE"
        status = "REJECTED"
        # confidence_pct: نعطي قواعد فقط (أو ML لو موجود لكن ما يغيّر الرفض)
        conf_pct = int(round(r_score))
        reason = " | ".join(reasons[:3]) if reasons else "Rejected by rules."
    else:
        # rules OK => now ML (if available) decides BUY/NO_TRADE
        if ml_conf is not None:
            recommendation = "BUY" if ml_conf >= ML_THRESHOLD else "NO_TRADE"
            status = "APPROVED" if recommendation == "BUY" else "APPROVED_BUT_LOW_ML"
            conf_pct = int(round(ml_conf * 100))
            reason = "Rules+ML confirmed entry." if recommendation == "BUY" else "Rules OK but ML below threshold."
        else:
            # no model -> rules-based only
            recommendation = "BUY" if r_score >= 70 else "NO_TRADE"
            status = "APPROVED_RULES_ONLY" if recommendation == "BUY" else "APPROVED_BUT_LOW_SCORE"
            conf_pct = int(round(r_score))
            reason = "Rules-based decision (ML model not available)."

    return {
        "ticker": t,
        "recommendation": recommendation,
        "status": status,
        "confidence_pct": conf_pct,          # هذا اللي تعرضه في الواجهة %
        "ml_confidence": ml_conf,            # رقم بين 0 و1 أو null إذا ML فشل
        "rules_score": int(r_score),          # 0..100
        "entry": round(entry, 4),
        "take_profit": round(take_profit, 4),
        "stop_loss": round(stop_loss, 4),
        "reason": reason,
        "last_close": round(last_close, 4),
    }

# =========================
# 6) تحميل قائمة الأسهم من الملف
# =========================
def load_tickers_from_file(path: str):
    if not os.path.exists(path):
        return []
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip().upper()
            if not s:
                continue
            if not s.endswith(".SR"):
                s += ".SR"
            items.append(s)

    # إزالة التكرار
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
        "rules_version": "v1.2",
        "tp_pct": TP_PCT,
        "sl_pct": SL_PCT,
        "tickers_file": TICKERS_PATH,
        "ml_threshold": ML_THRESHOLD,
        "top10_mode": "ranking+decision",
    }

@app.get("/predict")
def predict(ticker: str):
    return analyze_one(ticker)

@app.get("/debug_prices")
def debug_prices(ticker: str):
    t = ticker.strip().upper()
    if not t.endswith(".SR"):
        t += ".SR"
    df = fetch_yahoo_prices(t, range_="1y", interval="1d")
    if df is None:
        return {"ticker": t, "ok": False, "rows": 0}
    return {"ticker": t, "ok": True, "rows": int(len(df)), "last_close": float(df["close"].iloc[-1])}

@app.get("/check_tickers")
def check_tickers():
    path = os.path.abspath(TICKERS_PATH)
    tickers = load_tickers_from_file(TICKERS_PATH)
    return {
        "exists": os.path.exists(TICKERS_PATH),
        "path": path,
        "count": len(tickers),
        "sample": tickers[:10],
    }

@app.get("/top10")
def top10(max_workers: int = TOP10_WORKERS):
    """
    يحلل كل أسهم تاسي من tickers_sa.txt ويرجع أفضل 10
    الترتيب:
      1) BUY أولاً
      2) confidence_pct أعلى
      3) ml_confidence أعلى (إن وجد)
    """
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

    def score(item):
        rec = item.get("recommendation", "NO_TRADE")
        confp = int(item.get("confidence_pct", 0) or 0)
        mlc = item.get("ml_confidence")
        mlc = float(mlc) if isinstance(mlc, (int, float)) else -1.0
        return (1 if rec == "BUY" else 0, confp, mlc)

    results_sorted = sorted(results, key=score, reverse=True)
    top_items = results_sorted[:10]

    return {
        "items": top_items,
        "total": len(results),
        "errors": errors,
        "source": TICKERS_PATH,
        "note": "Rules v1.2 + ML always computed (if model loaded)."
    }
