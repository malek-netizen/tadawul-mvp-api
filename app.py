from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import requests
import pandas as pd
import numpy as np
import time
import joblib
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# =========================
# Tadawul MVP API (Rules v1.1 + Top10 Cache)
# هدف: +5% | وقف: -2% | تاسي فقط
# =========================

app = FastAPI(title="Tadawul MVP API", version="2.1.0-rules-v1.1-cache")

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
# إعدادات النظام (تقدر تعدلها لاحقًا)
# =========================
TP_PCT = 0.05
SL_PCT = 0.02

# فلترة RSI
RSI_MIN = 40
RSI_MAX = 65
RSI_OVERBOUGHT = 70

# قرب السعر من المتوسطات
MAX_ABOVE_EMA20 = 0.06   # 6% فوق EMA20 = مبالغة غالباً
MAX_ABOVE_EMA50 = 0.08   # 8% فوق EMA50

# فلترة الاندفاع (شمعة قوية بدون تماسك)
SPIKE_RET1 = 0.08        # 8% يوم واحد تعتبر اندفاع قوي
CONSOL_DAYS = 4
CONSOL_RANGE_MAX = 0.06  # تماسك = مدى 4 أيام <= 6%

# فلترة الهبوط/كسر قاع (اتجاه سلبي نشط)
LOOKBACK_LOW_DAYS = 10
MIN_TREND_SCORE = 2      # لازم يحقق حد أدنى من شروط الاتجاه

# ML decision threshold
ML_THRESHOLD = 0.60

# =========================
# Top10 Cache (الكاش فقط)
# =========================
TOP10_CACHE_TTL_SEC = int(os.getenv("TOP10_CACHE_TTL_SEC", "900"))  # 15 دقيقة افتراضياً
_top10_cache = {
    "ts": 0.0,
    "payload": None,
}
_top10_lock = Lock()

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
    return X, row.to_dict()

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
# 4) قواعد الدخول (Rules v1.1)
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
    if ret1 >= SPIKE_RET1 and (not is_consolidating(d, CONSOL_DAYS)):
        return True
    return False

def risk_check_stop(df_ohlc: pd.DataFrame, entry: float, stop: float) -> bool:
    if len(df_ohlc) < 10:
        return False
    recent = df_ohlc.tail(5)
    recent_low = float(recent["low"].min())
    if recent_low < stop:
        return False
    return True

def passes_rules(df_ohlc: pd.DataFrame, feat: pd.DataFrame):
    reasons = []
    d = feat

    last = d.iloc[-1]
    close = float(last["close"])
    ema20 = float(last["ema20"])
    ema50 = float(last["ema50"])
    rsi14 = float(last["rsi14"])

    if broke_recent_low(df_ohlc, LOOKBACK_LOW_DAYS):
        reasons.append("Rejected: recent low broken (bearish context).")

    ts = trend_score(d)
    if ts < MIN_TREND_SCORE:
        reasons.append("Rejected: weak trend context (down/weak structure).")

    if spike_without_base(d):
        reasons.append("Rejected: strong 1-day spike without consolidation (chasing).")

    if rsi14 > RSI_OVERBOUGHT:
        reasons.append("Rejected: RSI overbought (>70).")
    if not (RSI_MIN <= rsi14 <= RSI_MAX):
        if not (rsi14 >= 35 and is_consolidating(d, CONSOL_DAYS)):
            reasons.append("Rejected: RSI not in safe band (40-65) and no consolidation exception.")

    if ema20 > 0 and (close - ema20) / ema20 > MAX_ABOVE_EMA20:
        reasons.append("Rejected: price too far above EMA20 (overextended).")
    if ema50 > 0 and (close - ema50) / ema50 > MAX_ABOVE_EMA50:
        reasons.append("Rejected: price too far above EMA50 (overextended).")

    if len(d) >= 4:
        h = d["macd_hist"].tail(4).values
        if not (h[-1] >= h[-2] or h[-1] >= 0):
            reasons.append("Rejected: MACD histogram not improving.")

    base_ok = is_consolidating(d, CONSOL_DAYS)
    if not base_ok and ts < 3:
        reasons.append("Rejected: no clear consolidation/base before entry.")

    ok = (len(reasons) == 0)
    return ok, reasons, ts

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def calc_rules_score_pct(ok: bool, reasons: list, ts: int) -> int:
    # score بسيط: يبدأ 70 لو ok، وإلا يبدأ 50 ثم ينقص حسب عدد الرفض
    if ok:
        base = 70
        base += 5 * (ts - 2)  # trend_score يقوي
        return int(clamp(base, 55, 90))
    else:
        base = 50
        base -= 7 * min(len(reasons), 5)
        base += 3 * (ts - 2)
        return int(clamp(base, 5, 60))

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
        last_close_raw = float(df["close"].iloc[-1]) if len(df) else None
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
            "last_close": last_close_raw,
        }

    last_close = float(feat_df["close"].iloc[-1])
    entry = last_close
    take_profit = entry * (1.0 + TP_PCT)
    stop_loss = entry * (1.0 - SL_PCT)

    ok, reasons, ts = passes_rules(df, feat_df)

    if ok and (not risk_check_stop(df, entry, stop_loss)):
        ok = False
        reasons.append("Rejected: -2% stop is too tight vs recent lows.")

    rules_score = calc_rules_score_pct(ok, reasons, ts)

    ml_conf = None
    if model is not None:
        try:
            X, _ = latest_feature_vector(feat_df)
            ml_conf = float(model.predict_proba(X)[0, 1])
        except Exception:
            ml_conf = None

    # قرار نهائي
    if model is not None and ml_conf is not None:
        recommendation = "BUY" if (ok and ml_conf >= ML_THRESHOLD) else "NO_TRADE"
        # ثقة نهائية: مزج بسيط (Rules 70% + ML 30%)
        confidence_pct = int(round(0.7 * rules_score + 0.3 * (ml_conf * 100)))
    else:
        recommendation = "BUY" if ok else "NO_TRADE"
        confidence_pct = int(rules_score)

    status = "ACCEPTED" if recommendation == "BUY" else ("REJECTED" if not ok else "FILTERED")

    reason = " | ".join(reasons[:5]) if reasons else ("Rules+Model confirmed entry." if recommendation == "BUY" else "Probability below threshold")

    return {
        "ticker": t,
        "recommendation": recommendation,
        "status": status,
        "confidence_pct": int(clamp(confidence_pct, 0, 99)),
        "ml_confidence": (round(ml_conf, 4) if ml_conf is not None else None),
        "rules_score": int(clamp(rules_score, 0, 99)),
        "entry": round(entry, 4),
        "take_profit": round(take_profit, 4),
        "stop_loss": round(stop_loss, 4),
        "reason": reason,
        "last_close": round(last_close, 4),
    }

# =========================
# 6) Utilities + Endpoints
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

@app.get("/health")
def health():
    return {
        "status": "OK",
        "model_loaded": bool(model),
        "rules_version": "v1.1",
        "tp_pct": TP_PCT,
        "sl_pct": SL_PCT,
        "tickers_file": TICKERS_PATH,
        "ml_threshold": ML_THRESHOLD,
        "top10_cache_ttl_sec": TOP10_CACHE_TTL_SEC,
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
    if not os.path.exists(TICKERS_PATH):
        return {"exists": False, "path": path, "count": 0, "sample": []}
    tickers = load_tickers_from_file(TICKERS_PATH)
    return {"exists": True, "path": path, "count": len(tickers), "sample": tickers[:10]}

# =========================
# 7) Top10 + CACHE
# =========================
def compute_top10(max_workers: int = 8):
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

    # ترتيب: BUY أولاً، ثم confidence_pct أعلى
    def score(item):
        rec = item.get("recommendation", "NO_TRADE")
        conf = float(item.get("confidence_pct", 0) or 0)
        return (1 if rec == "BUY" else 0, conf)

    results_sorted = sorted(results, key=score, reverse=True)
    top_items = results_sorted[:10]

    return {
        "items": top_items,
        "total": len(results),
        "errors": errors,
        "source": TICKERS_PATH,
        "note": "Rules v1.1 + Top10 cache: returns cached result quickly, use refresh=1 to recompute.",
    }

@app.get("/top10")
def top10(max_workers: int = 8, refresh: int = 0):
    """
    كاش لنتيجة Top10 لتجنب Timeout على Safari/Render.
    - refresh=0: يرجع من الكاش إذا صالح
    - refresh=1: يعيد الحساب ويحدّث الكاش
    """
    now = time.time()

    # 1) إذا ما طلب Refresh وحصلنا كاش صالح
    if not refresh:
        with _top10_lock:
            payload = _top10_cache.get("payload")
            ts = float(_top10_cache.get("ts") or 0.0)
            if payload is not None and (now - ts) < TOP10_CACHE_TTL_SEC:
                payload_out = dict(payload)
                payload_out["cached"] = True
                payload_out["cache_age_sec"] = int(now - ts)
                payload_out["cache_ttl_sec"] = TOP10_CACHE_TTL_SEC
                return payload_out

    # 2) احسب جديد (خارج القفل)
    fresh = compute_top10(max_workers=max_workers)
    fresh["cached"] = False
    fresh["cache_age_sec"] = 0
    fresh["cache_ttl_sec"] = TOP10_CACHE_TTL_SEC

    # 3) خزّن النتيجة
    with _top10_lock:
        _top10_cache["ts"] = now
        _top10_cache["payload"] = fresh

    return fresh
