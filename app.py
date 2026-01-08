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
# Tadawul MVP API (Rules v1.1)
# هدف: +5% | وقف: -2% | تاسي فقط
# تعديلين:
# 1) confidence_pct حقيقي (مش 0 دائماً)
# 2) status لحالة السهم (READY/NEAR_SETUP/WATCH/REJECTED)
# =========================

app = FastAPI(title="Tadawul MVP API", version="2.1.0-rules-v1.1")

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

# RSI
RSI_MIN = 40
RSI_MAX = 65
RSI_OVERBOUGHT = 70

# قرب السعر من المتوسطات
MAX_ABOVE_EMA20 = 0.06   # 6% فوق EMA20
MAX_ABOVE_EMA50 = 0.08   # 8% فوق EMA50

# فلترة الاندفاع
SPIKE_RET1 = 0.08        # 8% يوم واحد تعتبر اندفاع قوي
CONSOL_DAYS = 4
CONSOL_RANGE_MAX = 0.06  # تماسك = مدى 4 أيام <= 6%

# فلترة الاتجاه السلبي
LOOKBACK_LOW_DAYS = 10
MIN_TREND_SCORE = 2

# عتبة ML
ML_THRESHOLD = 0.60

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
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
        except Exception:
            model = None
    else:
        model = None

# =========================
# 4) قواعد الدخول + scoring + status
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

def risk_check_stop(df_ohlc: pd.DataFrame, stop: float) -> bool:
    if len(df_ohlc) < 10:
        return False
    recent = df_ohlc.tail(5)
    recent_low = float(recent["low"].min())
    return recent_low >= stop

def rules_score_and_reasons(df_ohlc: pd.DataFrame, feat: pd.DataFrame):
    """
    يرجع:
      rules_ok (bool)
      rules_score (0..100)
      reasons (list[str])
      status (READY/NEAR_SETUP/WATCH/REJECTED) بناء على القواعد
    """
    d = feat
    last = d.iloc[-1]
    close = float(last["close"])
    ema20 = float(last["ema20"])
    ema50 = float(last["ema50"])
    rsi14 = float(last["rsi14"])

    reasons = []
    hard_reject = False

    ts = trend_score(d)
    base_ok = is_consolidating(d, CONSOL_DAYS)

    # بنظام نقاط (100)
    score = 100

    # 1) سياق سلبي قوي = خصم كبير
    if broke_recent_low(df_ohlc, LOOKBACK_LOW_DAYS):
        reasons.append("Rejected: recent low broken (bearish context).")
        score -= 30
        hard_reject = True

    if ts < MIN_TREND_SCORE:
        reasons.append("Rejected: weak trend context (down/weak structure).")
        score -= 25

    # 2) اندفاع بدون قاعدة
    if spike_without_base(d):
        reasons.append("Rejected: strong 1-day spike without consolidation (chasing).")
        score -= 25
        hard_reject = True

    # 3) RSI
    if rsi14 > RSI_OVERBOUGHT:
        reasons.append("Rejected: RSI overbought (>70).")
        score -= 25
        hard_reject = True

    in_safe_band = (RSI_MIN <= rsi14 <= RSI_MAX)
    consol_exception = (rsi14 >= 35 and base_ok)

    if not (in_safe_band or consol_exception):
        reasons.append("Rejected: RSI not in safe band (40-65) and no consolidation exception.")
        score -= 15

    # 4) Overextension
    if ema20 > 0 and (close - ema20) / ema20 > MAX_ABOVE_EMA20:
        reasons.append("Rejected: price too far above EMA20 (overextended).")
        score -= 15
        hard_reject = True

    if ema50 > 0 and (close - ema50) / ema50 > MAX_ABOVE_EMA50:
        reasons.append("Rejected: price too far above EMA50 (overextended).")
        score -= 10
        hard_reject = True

    # 5) MACD تحسين
    if len(d) >= 4:
        h = d["macd_hist"].tail(4).values
        if not (h[-1] >= h[-2] or h[-1] >= 0):
            reasons.append("Rejected: MACD histogram not improving.")
            score -= 10

    # 6) قاعدة (تماسك)
    if not base_ok and ts < 3:
        reasons.append("Rejected: no clear consolidation/base before entry.")
        score -= 20

    score = max(0, min(100, int(round(score))))

    # rules_ok = جاهز للدخول (بدون أسباب حاسمة)
    rules_ok = (score >= 80) and (not hard_reject)

    # status بناءً على score + طبيعة الرفض
    if rules_ok:
        status = "READY"
    else:
        if hard_reject or score < 50:
            status = "REJECTED"
        elif score >= 70:
            status = "NEAR_SETUP"
        else:
            status = "WATCH"

    return rules_ok, score, reasons, status

def combined_confidence(rules_score: int, ml_conf: float | None):
    """
    confidence_pct 0..100:
    - لو ML موجود: مزيج (60% قواعد + 40% ML)
    - لو ما فيه ML: يعتمد على القواعد فقط
    """
    rules_part = float(rules_score)
    if ml_conf is None:
        return int(round(rules_part))
    ml_part = float(ml_conf) * 100.0
    return int(round(0.60 * rules_part + 0.40 * ml_part))

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
            "status": "REJECTED",
            "confidence_pct": 0,
            "ml_confidence": None,
            "rules_score": 0,
            "reason": "No price data returned from provider."
        }

    feat_df = build_features(df)
    if len(feat_df) < 10:
        return {
            "ticker": t,
            "recommendation": "NO_TRADE",
            "status": "REJECTED",
            "confidence_pct": 0,
            "ml_confidence": None,
            "rules_score": 0,
            "reason": "Not enough data after feature engineering."
        }

    last_close = float(feat_df["close"].iloc[-1])
    entry = last_close
    take_profit = entry * (1.0 + TP_PCT)
    stop_loss = entry * (1.0 - SL_PCT)

    # قواعد + scoring
    rules_ok, rscore, reasons, status = rules_score_and_reasons(df, feat_df)

    # فحص وقف -2%
    if rules_ok and (not risk_check_stop(df, stop_loss)):
        rules_ok = False
        rscore = max(0, rscore - 20)
        reasons.append("Rejected: -2% stop is too tight vs recent lows.")
        status = "WATCH" if rscore >= 60 else "REJECTED"

    # ML
    ml_conf = None
    if model is not None:
        try:
            X, _ = latest_feature_vector(feat_df)
            ml_conf = float(model.predict_proba(X)[0, 1])
        except Exception:
            ml_conf = None

    # recommendation للتوافق (BUY فقط إذا قواعد جاهزة + ML (إن وجد) فوق العتبة)
    if ml_conf is None:
        recommendation = "BUY" if rules_ok else "NO_TRADE"
    else:
        recommendation = "BUY" if (rules_ok and ml_conf >= ML_THRESHOLD) else "NO_TRADE"

    # confidence_pct دائماً يظهر
    conf_pct = combined_confidence(rscore, ml_conf)

    # سبب مختصر
    if recommendation == "BUY":
        reason = "Rules+Confidence confirmed entry."
        status = "READY"
    else:
        if reasons:
            reason = " | ".join(reasons[:3])
        else:
            reason = "Not enough confirmation."

    return {
        "ticker": t,
        "recommendation": recommendation,
        "status": status,
        "confidence_pct": int(conf_pct),
        "ml_confidence": (None if ml_conf is None else round(ml_conf, 4)),
        "rules_score": int(rscore),
        "entry": round(entry, 4),
        "take_profit": round(take_profit, 4),
        "stop_loss": round(stop_loss, 4),
        "reason": reason,
        "last_close": round(last_close, 4),
    }

# =========================
# 6) Endpoints
# =========================
@app.get("/health")
def health():
    return {
        "status": "OK",
        "model_loaded": bool(model),
        "rules_version": "v1.1",
        "tp_pct": TP_PCT,
        "sl_pct": SL_PCT,
        "tickers_file": TICKERS_PATH,
        "ml_threshold": ML_THRESHOLD
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

# =========================
# 7) Top10 (تاسي فقط عبر tickers_sa.txt)
# =========================
def load_tickers_from_file(path: str):
    if not os.path.exists(path):
        return []
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            x = line.strip().upper()
            if not x:
                continue
            if not x.endswith(".SR"):
                x += ".SR"
            items.append(x)

    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

@app.get("/top10")
def top10(max_workers: int = 8):
    """
    يقرأ tickers_sa.txt ويحلل كل الأسهم ويرجع أفضل 10
    الترتيب:
      1) READY أولاً
      2) NEAR_SETUP
      3) WATCH
      4) REJECTED
    ثم confidence_pct أعلى
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

    status_rank = {"READY": 3, "NEAR_SETUP": 2, "WATCH": 1, "REJECTED": 0}

    def score(item):
        st = item.get("status", "REJECTED")
        conf = float(item.get("confidence_pct", 0) or 0)
        return (status_rank.get(st, 0), conf)

    results_sorted = sorted(results, key=score, reverse=True)
    top_items = results_sorted[:10]

    return {
        "items": top_items,
        "total": len(results),
        "errors": errors,
        "source": TICKERS_PATH,
        "note": "Rules v1.1: adds status + confidence_pct (rules_score + optional ML)"
    }
