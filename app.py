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
# Tadawul MVP API (Rules v1.3 - Enhanced)
# هدف: +5% | وقف: ديناميكي (2*ATR) مع حد أقصى -2% | تاسي فقط
# إضافات: ADX, Stochastic, CMF, حجم, نسبة مخاطرة/مكافأة
# ML + Rules combined score
# =========================

app = FastAPI(title="Tadawul MVP API", version="2.3.0-rules-v1.3-enhanced")

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
SL_PCT = 0.02  # سيصبح حدًا أقصى، مع وقف ديناميكي

# حدود ومحددات قواعد الدخول (محدثة)
RSI_MIN = 45
RSI_MAX = 60
RSI_OVERBOUGHT = 70

ADX_MIN = 25           # NEW: قوة اتجاه
STOCH_MAX = 80         # NEW: تجنب التشبع الشرائي
CMF_MIN = 0.0          # NEW: تدفق نقدي إيجابي
VOL_RATIO_MIN = 1.2    # NEW: حجم فوق المتوسط
RR_MIN = 2.0           # NEW: نسبة مخاطرة/مكافأة

MAX_ABOVE_EMA20 = 0.06   # 6%
MAX_ABOVE_EMA50 = 0.08   # 8%

SPIKE_RET1 = 0.08        # 8% يوم واحد اندفاع
CONSOL_DAYS = 4
CONSOL_RANGE_MAX = 0.06  # 6% مدى

LOOKBACK_LOW_DAYS = 10
MIN_TREND_SCORE = 2

ML_THRESHOLD = 0.60  # سيستخدم في الدرجة المركبة مع القواعد
TOP10_WORKERS = 6

model = None

# =========================
# Cache للأسعار لتقليل ضغط Yahoo
# =========================
_prices_cache = {}  # key: (ticker, range_, interval) -> {"ts": ..., "df": ...}
CACHE_TTL_SEC = 600  # 10 دقائق

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
# Cache خاص بنتيجة /top10
# =========================
_top10_cache = {"ts": 0, "data": None}
TOP10_CACHE_TTL_SEC = 600  # 10 دقائق

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
# 2) مؤشرات فنية (تمت إضافة دوال جديدة)
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

# NEW: ADX
def adx(high, low, close, period=14):
    high_low = high - low
    high_close = (high - close.shift()).abs()
    low_close = (low - close.shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    up_move = high - high.shift()
    down_move = low.shift() - low
    pos_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    neg_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    atr_ = tr.ewm(span=period, adjust=False).mean()
    pos_di = 100 * (pd.Series(pos_dm).ewm(span=period, adjust=False).mean() / atr_)
    neg_di = 100 * (pd.Series(neg_dm).ewm(span=period, adjust=False).mean() / atr_)

    dx = 100 * (abs(pos_di - neg_di) / (pos_di + neg_di + 1e-9))
    adx_ = dx.ewm(span=period, adjust=False).mean()
    return adx_

# NEW: Stochastic
def stochastic(close, high, low, k_period=14, d_period=3):
    low_min = low.rolling(k_period).min()
    high_max = high.rolling(k_period).max()
    k = 100 * (close - low_min) / (high_max - low_min + 1e-9)
    d = k.rolling(d_period).mean()
    return k, d

# NEW: Chaikin Money Flow
def chaikin_mf(high, low, close, volume, period=20):
    mfm = ((close - low) - (high - close)) / (high - low + 1e-9)
    mfv = mfm * volume
    cmf = mfv.rolling(period).sum() / volume.rolling(period).sum()
    return cmf

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

    # NEW: المؤشرات الجديدة
    d["adx14"] = adx(high, low, close, 14)
    d["stoch_k14"], d["stoch_d14"] = stochastic(close, high, low, 14, 3)
    d["cmf20"] = chaikin_mf(high, low, close, vol, 20)

    d["ret1"] = close.pct_change(1)
    d["ret3"] = close.pct_change(3)
    d["ret5"] = close.pct_change(5)
    d["vol20"] = d["ret1"].rolling(20).std()

    d["vol_ma20"] = vol.rolling(20).mean()
    d["vol_ratio"] = vol / (d["vol_ma20"] + 1e-9)

    d = d.dropna().reset_index(drop=True)
    return d

# تحديث قائمة الخصائص لتشمل المؤشرات الجديدة (اختياري للنموذج)
FEATURE_COLS = [
    "sma10", "sma20", "sma50",
    "ema10", "ema20", "ema50",
    "rsi14",
    "macd", "macd_signal", "macd_hist",
    "bb_width",
    "atr14",
    "adx14", "stoch_k14", "stoch_d14", "cmf20",   # NEW
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
# 4) قواعد الدخول (محدثة)
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
    return ret1 >= SPIKE_RET1 and (not is_consolidating(feat_df, CONSOL_DAYS))

def risk_check_stop(df_ohlc: pd.DataFrame, stop: float) -> bool:
    if len(df_ohlc) < 10:
        return False
    recent = df_ohlc.tail(5)
    recent_low = float(recent["low"].min())
    return not (recent_low < stop)

# NEW: دالة لحساب نسبة المخاطرة/المكافأة وإضافتها للأسباب
def passes_rules(df_ohlc: pd.DataFrame, feat_df: pd.DataFrame, stop_loss: float = None):
    reasons = []
    last = feat_df.iloc[-1]

    close = float(last["close"])
    ema20 = float(last["ema20"])
    ema50 = float(last["ema50"])
    rsi14 = float(last["rsi14"])

    # القواعد الحالية
    if broke_recent_low(df_ohlc, LOOKBACK_LOW_DAYS):
        reasons.append("Rejected: recent low broken (bearish context).")

    ts = trend_score(feat_df)
    if ts < MIN_TREND_SCORE:
        reasons.append("Rejected: weak trend context (down/weak structure).")

    if spike_without_base(feat_df):
        reasons.append("Rejected: strong 1-day spike without consolidation (chasing).")

    if rsi14 > RSI_OVERBOUGHT:
        reasons.append("Rejected: RSI overbought (>70).")

    # تحديث شرط RSI (نطاق 45-60 بدون استثناءات)
    if not (RSI_MIN <= rsi14 <= RSI_MAX):
        reasons.append(f"Rejected: RSI not in safe band ({RSI_MIN}-{RSI_MAX}).")

    if ema20 > 0 and (close - ema20) / ema20 > MAX_ABOVE_EMA20:
        reasons.append("Rejected: price too far above EMA20 (overextended).")
    if ema50 > 0 and (close - ema50) / ema50 > MAX_ABOVE_EMA50:
        reasons.append("Rejected: price too far above EMA50 (overextended).")

    if len(feat_df) >= 4:
        h = feat_df["macd_hist"].tail(4).values
        if not (h[-1] >= h[-2] or h[-1] >= 0):
            reasons.append("Rejected: MACD histogram not improving.")

    base_ok = is_consolidating(feat_df, CONSOL_DAYS)
    if not base_ok and ts < 3:
        reasons.append("Rejected: no clear consolidation/base before entry.")

    # NEW: شروط إضافية
    adx14 = float(last["adx14"])
    if adx14 < ADX_MIN:
        reasons.append(f"Rejected: ADX below {ADX_MIN} (weak trend).")

    stoch_k = float(last["stoch_k14"])
    if stoch_k > STOCH_MAX:
        reasons.append(f"Rejected: Stochastic overbought (>{STOCH_MAX}).")

    cmf = float(last["cmf20"])
    if cmf < CMF_MIN:
        reasons.append(f"Rejected: Chaikin Money Flow negative (<{CMF_MIN}).")

    vol_ratio = float(last["vol_ratio"])
    if vol_ratio < VOL_RATIO_MIN:
        reasons.append(f"Rejected: Volume below average (ratio < {VOL_RATIO_MIN}).")

    # NEW: نسبة المخاطرة/المكافأة إذا تم تمرير stop_loss
    if stop_loss is not None:
        risk_pct = (close - stop_loss) / close  # الخسارة كنسبة مئوية من السعر الحالي
        reward_pct = TP_PCT
        if reward_pct / (risk_pct + 1e-9) < RR_MIN:
            reasons.append(f"Rejected: Risk/Reward ratio < {RR_MIN} ({reward_pct/(risk_pct+1e-9):.2f})")

    return (len(reasons) == 0), reasons

def rules_score(feat_df: pd.DataFrame) -> int:
    if len(feat_df) < 5:
        return 30
    last = feat_df.iloc[-1]
    score = 0

    score += 15 if float(last["close"]) >= float(last["ema20"]) else 0
    score += 15 if float(last["ema20"]) >= float(last["ema50"]) else 0

    r = float(last["rsi14"])
    if RSI_MIN <= r <= RSI_MAX:
        score += 20
    elif 35 <= r < RSI_MIN:
        score += 10

    score += 15 if float(last["macd_hist"]) >= 0 else 5
    score += 20 if is_consolidating(feat_df, CONSOL_DAYS) else 5

    close = float(last["close"])
    ema20 = float(last["ema20"])
    if ema20 > 0 and (close - ema20) / ema20 > MAX_ABOVE_EMA20:
        score -= 10

    score = max(0, min(100, score))
    return int(score)

# =========================
# 5) تحليل سهم واحد (محدث)
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
            "atr_pct": None,          # NEW
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
            "atr_pct": None,
            "reason": "Not enough data after feature engineering.",
            "last_close": None,
        }

    last_close = float(feat_df["close"].iloc[-1])
    entry = last_close

    # NEW: وقف خسارة ديناميكي بناءً على ATR مع حد أقصى SL_PCT
    atr_val = float(feat_df["atr14"].iloc[-1])
    stop_loss_dynamic = entry - (2 * atr_val)
    stop_loss_fixed = entry * (1.0 - SL_PCT)
    # نختار الوقف الأكثر تقييدًا (الأعلى سعرًا) لحماية رأس المال
    stop_loss = max(stop_loss_fixed, stop_loss_dynamic)
    take_profit = entry * (1.0 + TP_PCT)

    # تمرير stop_loss إلى passes_rules
    ok, reasons = passes_rules(df, feat_df, stop_loss)
    if ok and (not risk_check_stop(df, stop_loss)):
        ok = False
        reasons.append("Rejected: stop loss is too tight vs recent lows.")

    r_score = rules_score(feat_df)

    ml_conf = None
    if model is not None:
        try:
            X, _ = latest_feature_vector(feat_df)
            ml_conf = float(model.predict_proba(X)[0, 1])
        except Exception:
            ml_conf = None

    # NEW: دمج القرارات بدرجة مركبة
    if not ok:
        recommendation = "NO_TRADE"
        status = "REJECTED"
        conf_pct = int(round(r_score))
        reason = " | ".join(reasons[:3]) if reasons else "Rejected by rules."
    else:
        if ml_conf is not None:
            # درجة مركبة: 30% قواعد + 70% ML (محولة إلى مقياس 0-100)
            combined = (r_score * 0.3) + (ml_conf * 0.7 * 100)
            if combined >= 65:   # عتبة مركبة
                recommendation = "BUY"
                status = "APPROVED"
                conf_pct = int(round(combined))
                reason = "Combined score OK (rules+ML)."
            else:
                recommendation = "NO_TRADE"
                status = "APPROVED_BUT_LOW_COMBINED"
                conf_pct = int(round(combined))
                reason = f"Combined score too low ({combined:.1f} < 65)."
        else:
            # بدون ML، نعتمد على القواعد فقط
            if r_score >= 70:
                recommendation = "BUY"
                status = "APPROVED_RULES_ONLY"
                conf_pct = int(round(r_score))
                reason = "Rules-based decision (ML model not available)."
            else:
                recommendation = "NO_TRADE"
                status = "APPROVED_BUT_LOW_SCORE"
                conf_pct = int(round(r_score))
                reason = "Rules OK but score below 70."

    # NEW: حساب atr_pct للمساعدة في ترتيب top10
    atr_pct = (atr_val / entry) if atr_val else 0.0

    return {
        "ticker": t,
        "recommendation": recommendation,
        "status": status,
        "confidence_pct": conf_pct,
        "ml_confidence": ml_conf,
        "rules_score": int(r_score),
        "entry": round(entry, 4),
        "take_profit": round(take_profit, 4),
        "stop_loss": round(stop_loss, 4),
        "atr_pct": round(atr_pct * 100, 2),   # نسبة مئوية
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
        "rules_version": "v1.3-enhanced",
        "tp_pct": TP_PCT,
        "sl_pct": SL_PCT,
        "tickers_file": TICKERS_PATH,
        "ml_threshold": ML_THRESHOLD,
        "top10_workers": TOP10_WORKERS,
        "prices_cache_ttl_sec": CACHE_TTL_SEC,
        "top10_cache_ttl_sec": TOP10_CACHE_TTL_SEC,
        "rsi_min": RSI_MIN,
        "rsi_max": RSI_MAX,
        "adx_min": ADX_MIN,
        "stoch_max": STOCH_MAX,
        "cmf_min": CMF_MIN,
        "vol_ratio_min": VOL_RATIO_MIN,
        "rr_min": RR_MIN,
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
      3) ml_confidence أعلى
      مع معامل تخفيض للأسهم عالية التقلب (atr_pct)
    """

    now = time.time()
    if _top10_cache["data"] is not None and (now - _top10_cache["ts"] < TOP10_CACHE_TTL_SEC):
        cached_payload = dict(_top10_cache["data"])
        cached_payload["cached"] = True
        cached_payload["cached_age_sec"] = int(now - _top10_cache["ts"])
        return cached_payload

    tickers = load_tickers_from_file(TICKERS_PATH)
    if not tickers:
        return {"items": [], "error": f"Tickers file not found or empty: {TICKERS_PATH}", "cached": False}

    results = []
    errors = 0

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(analyze_one, t): t for t in tickers}
        for fut in as_completed(futures):
            try:
                res = fut.result()
                # نتأكد أن النتيجة تحتوي على الحقول الأساسية على الأقل
                if isinstance(res, dict) and "ticker" in res:
                    results.append(res)
                else:
                    errors += 1
            except Exception as e:
                errors += 1
                # يمكن تسجيل الخطأ هنا في ملف لوج
                print(f"Error processing ticker: {e}")

    # دالة ترتيب آمنة
    def safe_score(item):
        # القيم الافتراضية الآمنة
        rec = item.get("recommendation", "NO_TRADE")
        try:
            confp = float(item.get("confidence_pct", 0))
        except (TypeError, ValueError):
            confp = 0.0
        
        try:
            mlc = item.get("ml_confidence")
            if mlc is None:
                mlc = -1.0
            else:
                mlc = float(mlc)
        except (TypeError, ValueError):
            mlc = -1.0
        
        try:
            atr_pct = float(item.get("atr_pct", 5.0))
            if atr_pct <= 0:
                atr_pct = 5.0  # قيمة افتراضية لمنع القسمة على صفر
        except (TypeError, ValueError):
            atr_pct = 5.0
        
        # معامل التخفيض: نريد تقليل الوزن كلما زاد ATR
        # صيغة: penalty = max(0.5, atr_pct/2)  (أو أي صيغة مناسبة)
        penalty = max(0.5, atr_pct / 2.0)
        
        if rec == "BUY":
            # نعطي أولوية للـ BUY، ثم نستخدم الثقة بعد تطبيق penalty
            return (2, confp / penalty, mlc / penalty)
        else:
            return (1, confp / penalty, mlc / penalty)

    # تصفية النتائج الغير صالحة (مثل التي ليس بها توصية أو entry)
    valid_results = [r for r in results if r.get("entry") is not None]
    # إذا أردت، يمكنك الاحتفاظ بالكل ولكن الترتيب سيعالجها.

    try:
        results_sorted = sorted(valid_results, key=safe_score, reverse=True)
    except Exception as e:
        # في حالة فشل الفرز، نعيد قائمة فارغة مع الخطأ
        return {"items": [], "error": f"Sorting failed: {str(e)}", "total": len(results), "errors": errors}

    top_items = results_sorted[:10]

    payload = {
        "items": top_items,
        "total": len(results),
        "errors": errors,
        "source": TICKERS_PATH,
        "note": "Rules v1.3-enhanced + ML combined score. Top10 cached for 10 minutes.",
        "cached": False,
        "computed_at_unix": int(time.time()),
    }

    _top10_cache["ts"] = time.time()
    _top10_cache["data"] = payload

    return payload
