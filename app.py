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
# Tadawul MVP API (Rules v1.0 + Top10 Ranking)
# هدف: +5% | وقف: -2% | تاسي فقط
# =========================

app = FastAPI(title="Tadawul MVP API", version="2.1.0-rules-v1-top10-ranking")

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
MAX_ABOVE_EMA20 = 0.06   # 6% فوق EMA20 = مبالغة غالباً
MAX_ABOVE_EMA50 = 0.08   # 8% فوق EMA50

# فلترة الاندفاع (شمعة قوية بدون تماسك)
SPIKE_RET1 = 0.08        # 8% يوم واحد تعتبر اندفاع قوي
CONSOL_DAYS = 4
CONSOL_RANGE_MAX = 0.06  # تماسك = مدى 4 أيام <= 6%

# فلترة الهبوط/كسر قاع (اتجاه سلبي نشط)
LOOKBACK_LOW_DAYS = 10
MIN_TREND_SCORE = 2      # لازم يحقق حد أدنى من شروط الاتجاه

# ML threshold (إن وجد)
MODEL_BUY_THRESHOLD = 0.60

# =========================
# Cache بسيط لتخفيف ضغط Yahoo أثناء Top10
# =========================
_CACHE_TTL_SEC = 60 * 10  # 10 دقائق
_prices_cache = {}
_cache_lock = Lock()

def _cache_get(key: str):
    now = time.time()
    with _cache_lock:
        item = _prices_cache.get(key)
        if not item:
            return None
        ts, df = item
        if (now - ts) > _CACHE_TTL_SEC:
            _prices_cache.pop(key, None)
            return None
        return df

def _cache_set(key: str, df: pd.DataFrame):
    with _cache_lock:
        _prices_cache[key] = (time.time(), df)

# =========================
# 1) جلب الأسعار من Yahoo Chart
# =========================
def fetch_yahoo_prices(ticker: str, range_: str = "1y", interval: str = "1d", min_rows: int = 80):
    cache_key = f"{ticker}|{range_}|{interval}"
    cached = _cache_get(cache_key)
    if cached is not None and len(cached) >= min_rows:
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

    # محاولات قليلة مع انتظار بسيط
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
                    _cache_set(cache_key, df)
                    return df

            except Exception:
                continue

        time.sleep(0.8)

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
# 4) قواعد الدخول (Rules v1.0)
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
        # استثناء: RSI >=35 مع تماسك واضح
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
    return ok, reasons

# =========================
# 4.1) Top10 Ranking Score (حتى لو NO_TRADE)
# =========================
def setup_score(df_ohlc: pd.DataFrame, feat: pd.DataFrame):
    """
    يرجّع (score 0-100, notes[])
    هذا "ترشيح/Ranking" وليس ضمان دخول.
    """
    notes = []
    d = feat
    last = d.iloc[-1]

    close = float(last["close"])
    ema20 = float(last["ema20"])
    ema50 = float(last["ema50"])
    rsi14 = float(last["rsi14"])
    ret1 = float(last["ret1"])
    ts = trend_score(d)
    base_ok = is_consolidating(d, CONSOL_DAYS)

    score = 50  # نقطة بداية

    # 1) RSI قرب النطاق الآمن
    if RSI_MIN <= rsi14 <= RSI_MAX:
        score += 18
        notes.append("RSI in safe band (40-65)")
    elif 35 <= rsi14 < RSI_MIN and base_ok:
        score += 10
        notes.append("RSI slightly low but consolidation present")
    elif rsi14 > RSI_OVERBOUGHT:
        score -= 18
        notes.append("RSI overbought risk")

    # 2) قرب السعر من EMA20 (عدم مطاردة)
    if ema20 > 0:
        dist20 = (close - ema20) / ema20
        if dist20 <= 0.02:
            score += 14
            notes.append("Price close to EMA20")
        elif dist20 <= MAX_ABOVE_EMA20:
            score += 6
            notes.append("Price moderately above EMA20")
        else:
            score -= 16
            notes.append("Overextended above EMA20")

    # 3) Trend score
    score += (ts * 5)  # 0..20
    notes.append(f"Trend score={ts}/4")

    # 4) Base / consolidation
    if base_ok:
        score += 10
        notes.append("Has base/consolidation")

    # 5) تجنّب spike يوم واحد
    if ret1 >= SPIKE_RET1 and (not base_ok):
        score -= 18
        notes.append("1-day spike without base")

    # 6) كسر قاع حديث
    if broke_recent_low(df_ohlc, LOOKBACK_LOW_DAYS):
        score -= 20
        notes.append("Recent low broken")

    # ضبط
    score = max(0, min(100, score))
    return score, notes

# =========================
# 5) تحليل سهم واحد (قرار دخول صارم)
# =========================
def normalize_ticker(ticker: str) -> str:
    t = (ticker or "").strip().upper()
    if not t:
        return ""
    if not t.endswith(".SR"):
        t += ".SR"
    return t

def analyze_core(ticker: str):
    t = normalize_ticker(ticker)
    if not t:
        return None, None, None, {"error": "Ticker is required."}

    df = fetch_yahoo_prices(t, range_="1y", interval="1d")
    if df is None:
        return t, None, None, {
            "ticker": t,
            "recommendation": "NO_TRADE",
            "confidence": 0.0,
            "reason": "No price data returned from provider."
        }

    feat_df = build_features(df)
    if len(feat_df) < 10:
        return t, df, feat_df, {
            "ticker": t,
            "recommendation": "NO_TRADE",
            "confidence": 0.0,
            "reason": "Not enough data after feature engineering."
        }

    last_close = float(feat_df["close"].iloc[-1])
    return t, df, feat_df, None

def analyze_one(ticker: str):
    t, df, feat_df, early = analyze_core(ticker)
    if early is not None:
        return early

    last_close = float(feat_df["close"].iloc[-1])
    entry = last_close
    take_profit = entry * (1.0 + TP_PCT)
    stop_loss = entry * (1.0 - SL_PCT)

    ok, reasons = passes_rules(df, feat_df)

    if ok and (not risk_check_stop(df, entry, stop_loss)):
        ok = False
        reasons.append("Rejected: -2% stop is too tight vs recent lows.")

    conf = 0.0
    if model is not None:
        try:
            X, _ = latest_feature_vector(feat_df)
            conf = float(model.predict_proba(X)[0, 1])
        except Exception:
            conf = 0.0

        recommendation = "BUY" if (ok and conf >= MODEL_BUY_THRESHOLD) else "NO_TRADE"
        if recommendation == "NO_TRADE":
            if not ok and reasons:
                reason = " | ".join(reasons[:3])
            else:
                reason = "Probability below threshold"
        else:
            reason = "Rules+Model confirmed entry."

        return {
            "ticker": t,
            "recommendation": recommendation,
            "confidence": round(conf, 4),
            "entry": round(entry, 4),
            "take_profit": round(take_profit, 4),
            "stop_loss": round(stop_loss, 4),
            "reason": reason,
            "last_close": round(last_close, 4),
        }

    # Rule-based only
    if ok:
        last = feat_df.iloc[-1]
        score = 0
        score += 1 if float(last["close"]) >= float(last["ema20"]) else 0
        score += 1 if float(last["ema20"]) >= float(last["ema50"]) else 0
        score += 1 if (RSI_MIN <= float(last["rsi14"]) <= RSI_MAX) else 0
        score += 1 if float(last["macd_hist"]) >= 0 else 0
        score += 1 if is_consolidating(feat_df, CONSOL_DAYS) else 0
        conf = min(0.50 + 0.08 * score, 0.85)
        recommendation = "BUY"
        reason = "Rules confirmed: safe entry zone."
    else:
        recommendation = "NO_TRADE"
        conf = 0.35
        reason = " | ".join(reasons[:3]) if reasons else "Rules rejected."

    return {
        "ticker": t,
        "recommendation": recommendation,
        "confidence": round(conf, 4),
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
        "rules_version": "v1.0",
        "tp_pct": TP_PCT,
        "sl_pct": SL_PCT,
        "tickers_file": TICKERS_PATH,
        "top10_mode": "ranking+decision",
    }

@app.get("/predict")
def predict(ticker: str):
    return analyze_one(ticker)

@app.get("/debug_prices")
def debug_prices(ticker: str):
    t = normalize_ticker(ticker)
    if not t:
        return {"ok": False, "error": "Ticker is required."}
    df = fetch_yahoo_prices(t, range_="1y", interval="1d")
    if df is None:
        return {"ticker": t, "ok": False, "rows": 0}
    return {"ticker": t, "ok": True, "rows": int(len(df)), "last_close": float(df["close"].iloc[-1])}

# =========================
# 7) Tickers file loader + check
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

@app.get("/check_tickers")
def check_tickers():
    path = os.path.abspath(TICKERS_PATH)
    if not os.path.exists(TICKERS_PATH):
        return {"exists": False, "path": path, "count": 0, "sample": []}
    tickers = load_tickers_from_file(TICKERS_PATH)
    return {"exists": True, "path": path, "count": len(tickers), "sample": tickers[:10]}

# =========================
# 8) Top10 (Ranking السوق كامل + قرار دخول داخل كل عنصر)
# =========================
def analyze_one_for_top10(ticker: str):
    """
    يرجّع عنصر غني:
      - setup_score (Ranking)
      - decision (BUY/NO_TRADE) بناء على القواعد (ومع ML إن وجد)
      - entry/tp/sl + أسباب مختصرة
    """
    t, df, feat_df, early = analyze_core(ticker)
    if early is not None:
        # نخليه يرجع بشكل موحد للـ Top10
        return {
            "ticker": early.get("ticker", normalize_ticker(ticker)),
            "setup_score": 0,
            "decision": "NO_TRADE",
            "confidence": float(early.get("confidence", 0.0) or 0.0),
            "entry": None,
            "take_profit": None,
            "stop_loss": None,
            "reason": early.get("reason", "No data"),
            "notes": ["No data/early exit"],
        }

    last_close = float(feat_df["close"].iloc[-1])
    entry = last_close
    take_profit = entry * (1.0 + TP_PCT)
    stop_loss = entry * (1.0 - SL_PCT)

    # Ranking score (حتى لو NO_TRADE)
    sc, notes = setup_score(df, feat_df)

    # قرار صارم (نفس منطق analyze_one تقريباً)
    ok, reasons = passes_rules(df, feat_df)
    if ok and (not risk_check_stop(df, entry, stop_loss)):
        ok = False
        reasons.append("Rejected: -2% stop is too tight vs recent lows.")

    conf = 0.0
    decision = "BUY" if ok else "NO_TRADE"

    if model is not None:
        try:
            X, _ = latest_feature_vector(feat_df)
            conf = float(model.predict_proba(X)[0, 1])
        except Exception:
            conf = 0.0
        # حتى لو القواعد ok، لازم ML يوافق للـ BUY النهائي
        if not (ok and conf >= MODEL_BUY_THRESHOLD):
            decision = "NO_TRADE"

    # reason مختصر (للواجهة)
    if decision == "BUY":
        reason = "Top setup + Rules (and Model if enabled) confirm."
    else:
        if reasons:
            reason = " | ".join(reasons[:2])
        else:
            reason = "Not qualified for entry now"

    return {
        "ticker": t,
        "setup_score": int(sc),
        "decision": decision,
        "confidence": round(conf, 4),
        "entry": round(entry, 4),
        "take_profit": round(take_profit, 4),
        "stop_loss": round(stop_loss, 4),
        "reason": reason,
        "notes": notes[:4],
        "last_close": round(last_close, 4),
    }

@app.get("/top10")
def top10(max_workers: int = 6):
    """
    يقرأ tickers_sa.txt ويحلل السوق كامل ثم يرجع أفضل 10 مرشحين (Ranking)
    - لا يشترط وجود BUY، الهدف "أفضل 10" دائماً (Watchlist + Entry decision)
    ترتيب الأفضلية:
      1) BUY أولاً
      2) setup_score أعلى
      3) confidence أعلى (إذا ML شغال)
    """
    tickers = load_tickers_from_file(TICKERS_PATH)
    if not tickers:
        return {"items": [], "error": f"Tickers file not found or empty: {TICKERS_PATH}"}

    results = []
    errors = 0

    # لتقليل الضغط على Yahoo: لا ترفع max_workers كثير
    max_workers = max(2, min(int(max_workers), 10))

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(analyze_one_for_top10, t): t for t in tickers}
        for fut in as_completed(futures):
            try:
                results.append(fut.result())
            except Exception:
                errors += 1

    def score(item):
        decision = item.get("decision", "NO_TRADE")
        sc = int(item.get("setup_score", 0) or 0)
        conf = float(item.get("confidence", 0.0) or 0.0)
        return (1 if decision == "BUY" else 0, sc, conf)

    results_sorted = sorted(results, key=score, reverse=True)
    top_items = results_sorted[:10]

    return {
        "items": top_items,
        "total": len(results),
        "errors": errors,
        "source": TICKERS_PATH,
        "note": "Top10 is ranking (watchlist) + decision. Entry target +5% stop -2% (educational tool)."
    }
