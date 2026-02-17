from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import requests
import pandas as pd
import numpy as np
import time
import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict, Any, List
import uvicorn

# ======================== إعداد التسجيل ========================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("tadawul_sniper.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ======================== الإعدادات الثابتة ========================
TICKERS_PATH = os.getenv("TICKERS_PATH", "tickers_sa.txt")
TP_PCT = float(os.getenv("TP_PCT", "0.05"))
TOP10_WORKERS = int(os.getenv("TOP10_WORKERS", "10"))
CACHE_TTL_SEC = int(os.getenv("CACHE_TTL_SEC", "600"))
MIN_VOLUME = int(os.getenv("MIN_VOLUME", "250000"))      # تم التخفيض
MIN_PRICE = float(os.getenv("MIN_PRICE", "5.0"))         # تم التخفيض
ATR_EXCLUDE_PCT = float(os.getenv("ATR_EXCLUDE_PCT", "4.0"))  # تم الرفع
MAX_5DAY_GAIN = float(os.getenv("MAX_5DAY_GAIN", "0.15"))      # تم الرفع

# ======================== كاش بسيط يدوي ========================
_prices_cache = {}

# ======================== إنشاء تطبيق FastAPI ========================
app = FastAPI(
    title="Tadawul Sniper Pro",
    description="محلل فني لأسهم السوق السعودي",
    version="4.0.3"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================== التأكد من وجود ملف التيكرات ========================
if not os.path.exists(TICKERS_PATH):
    logger.warning(f"ملف {TICKERS_PATH} غير موجود، سيتم إنشاؤه فارغًا.")
    with open(TICKERS_PATH, "w") as f:
        f.write("# أضف رموز الأسهم هنا سطرًا بسطر\n")

# ======================== دوال جلب البيانات من Yahoo Finance ========================
def fetch_yahoo_prices(ticker: str, range_: str = "1y", interval: str = "1d") -> Optional[pd.DataFrame]:
    key = (ticker, range_, interval)
    if key in _prices_cache and time.time() - _prices_cache[key]["ts"] < CACHE_TTL_SEC:
        logger.info(f"استخدام البيانات المخزنة لـ {ticker}")
        return _prices_cache[key]["df"]

    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    headers = {"User-Agent": "Mozilla/5.0"}
    params = {"range": range_, "interval": interval}
    try:
        r = requests.get(url, params=params, headers=headers, timeout=15)
        r.raise_for_status()
        js = r.json()
        result = js['chart']['result'][0]
        quote = result['indicators']['quote'][0]
        if not quote.get('open') or not quote.get('high') or not quote.get('low') or not quote.get('close') or not quote.get('volume'):
            logger.warning(f"بيانات غير مكتملة لـ {ticker}")
            return None
        df = pd.DataFrame({
            "open": quote["open"],
            "high": quote["high"],
            "low": quote["low"],
            "close": quote["close"],
            "volume": quote["volume"]
        })
        df = df.dropna(subset=["close"]).reset_index(drop=True)
        if len(df) < 50:
            logger.warning(f"عدد أيام قليل جداً لـ {ticker}: {len(df)}")
            return None
        _prices_cache[key] = {"ts": time.time(), "df": df}
        logger.info(f"تم جلب بيانات {ticker} بنجاح ({len(df)} يوم)")
        return df
    except Exception as e:
        logger.error(f"خطأ في جلب {ticker}: {str(e)}")
        return None

# ======================== حساب المؤشرات الفنية يدويًا ========================
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    close = d["close"]
    high = d["high"]
    low = d["low"]
    volume = d["volume"]

    d["ema20"] = close.ewm(span=20, adjust=False).mean()
    d["sma20"] = close.rolling(20).mean()
    d["sma50"] = close.rolling(50).mean()

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    d["macd"] = ema12 - ema26
    d["macd_signal"] = d["macd"].ewm(span=9, adjust=False).mean()
    d["macd_hist"] = d["macd"] - d["macd_signal"]

    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    d["rsi14"] = 100 - (100 / (1 + (gain / (loss + 1e-9))))

    d["bb_mid"] = d["sma20"]
    d["bb_std"] = close.rolling(20).std()
    d["bb_upper"] = d["bb_mid"] + 2 * d["bb_std"]

    obv = [0]
    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i-1]:
            obv.append(obv[-1] + volume.iloc[i])
        elif close.iloc[i] < close.iloc[i-1]:
            obv.append(obv[-1] - volume.iloc[i])
        else:
            obv.append(obv[-1])
    d["obv"] = obv

    low14 = low.rolling(14).min()
    high14 = high.rolling(14).max()
    d["stoch_k"] = 100 * ((close - low14) / (high14 - low14 + 1e-9))
    d["stoch_d"] = d["stoch_k"].rolling(3).mean()

    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    d["atr14"] = tr.rolling(14).mean()
    d["atr_pct"] = d["atr14"] / close * 100

    d["vol_ma20"] = volume.rolling(20).mean()
    d["vol_std"] = volume.rolling(20).std()

    d["candle_green"] = close > d["open"]
    d["body"] = abs(close - d["open"])
    d["upper_shadow"] = high - d[["close", "open"]].max(axis=1)
    d["lower_shadow"] = d[["close", "open"]].min(axis=1) - low

    d = d.dropna().reset_index(drop=True)
    return d

# ======================== دوال الأنماط السعرية ========================
def is_bullish_engulfing(prev, curr):
    prev_red = prev["close"] < prev["open"]
    curr_green = curr["close"] > curr["open"]
    if not (prev_red and curr_green):
        return False
    return curr["open"] < prev["close"] and curr["close"] > prev["open"]

def is_hammer(candle):
    body = candle["body"]
    lower = candle["lower_shadow"]
    upper = candle["upper_shadow"]
    return lower > 2 * body and upper < 0.3 * body

def is_morning_star(c1, c2, c3):
    c1_red = c1["close"] < c1["open"]
    c3_green = c3["close"] > c3["open"]
    c2_small = c2["body"] < 0.1 * (c1["high"] - c1["low"])
    if not (c1_red and c3_green and c2_small):
        return False
    midpoint = (c1["open"] + c1["close"]) / 2
    return c3["close"] > midpoint

def is_piercing(prev, curr):
    if not (prev["close"] < prev["open"] and curr["close"] > curr["open"]):
        return False
    midpoint = (prev["open"] + prev["close"]) / 2
    return curr["close"] > midpoint and curr["open"] < prev["close"]

def get_candle_pattern_score(feat_df):
    if len(feat_df) < 3:
        return 0
    last = feat_df.iloc[-1]
    prev = feat_df.iloc[-2]
    prev2 = feat_df.iloc[-3]

    score = 0
    if is_bullish_engulfing(prev, last):
        score += 15
    elif is_morning_star(prev2, prev, last):
        score += 15
    elif is_piercing(prev, last):
        score += 12
    elif is_hammer(last) and (prev["close"] < prev["open"]):
        score += 12
    elif last["candle_green"]:
        body_pct = last["body"] / (last["high"] - last["low"]) if (last["high"]-last["low"])>0 else 0
        if body_pct > 0.5:
            score += 5
        else:
            score += 2

    if last["upper_shadow"] > 2 * last["body"]:
        score -= 3

    return max(0, min(15, score))

def is_bearish_engulfing(prev, curr):
    prev_green = prev["close"] > prev["open"]
    curr_red = curr["close"] < curr["open"]
    if not (prev_green and curr_red):
        return False
    return curr["open"] > prev["close"] and curr["close"] < prev["open"]

def is_shooting_star(candle):
    body = candle["body"]
    upper = candle["upper_shadow"]
    lower = candle["lower_shadow"]
    return upper > 2 * body and lower < 0.3 * body

def is_dark_cloud(prev, curr):
    if not (prev["close"] > prev["open"] and curr["close"] < curr["open"]):
        return False
    midpoint = (prev["open"] + prev["close"]) / 2
    return curr["open"] > prev["high"] and curr["close"] < midpoint

def is_hanging_man(candle):
    body = candle["body"]
    lower = candle["lower_shadow"]
    upper = candle["upper_shadow"]
    return lower > 2 * body and upper < 0.3 * body

def is_evening_star(c1, c2, c3):
    c1_green = c1["close"] > c1["open"]
    c3_red = c3["close"] < c3["open"]
    c2_small = c2["body"] < 0.1 * (c1["high"] - c1["low"])
    if not (c1_green and c3_red and c2_small):
        return False
    midpoint = (c1["open"] + c1["close"]) / 2
    return c3["close"] < midpoint

def has_bearish_pattern(feat_df):
    if len(feat_df) < 3:
        return False
    last = feat_df.iloc[-1]
    prev = feat_df.iloc[-2]
    prev2 = feat_df.iloc[-3]

    if is_bearish_engulfing(prev, last):
        return True
    if is_dark_cloud(prev, last):
        return True
    if is_shooting_star(last):
        return True
    if is_hanging_man(last):
        return True
    if is_evening_star(prev2, prev, last):
        return True
    return False

# ======================== شروط الاستبعاد ========================
def should_exclude(feat_df: pd.DataFrame) -> tuple[bool, str]:
    """تحديد ما إذا كان السهم مستبعداً بناءً على معايير المخاطرة."""
    curr = feat_df.iloc[-1]
    if curr["atr_pct"] > ATR_EXCLUDE_PCT:
        return True, f"تقلب عالي (ATR% = {curr['atr_pct']:.1f}%)"
    if len(feat_df) >= 6:
        close_5 = feat_df["close"].iloc[-6]
        gain_5 = (curr["close"] / close_5 - 1)
        if gain_5 > MAX_5DAY_GAIN:
            return True, f"ارتفاع كبير في 5 أيام ({gain_5*100:.1f}%)"
    if curr["volume"] < MIN_VOLUME:
        return True, f"سيولة منخفضة ({curr['volume']:,.0f})"
    if curr["close"] < MIN_PRICE:
        return True, f"سعر منخفض جداً ({curr['close']:.2f})"
    if has_bearish_pattern(feat_df.tail(4)):
        return True, "وجود نمط هابط"
    return False, ""

# ======================== الشروط الأساسية ========================
def passes_core_rules(feat_df: pd.DataFrame) -> tuple[bool, list[str]]:
    """الشروط الأساسية التي يجب توفرها للنظر في السهم."""
    reasons = []
    curr = feat_df.iloc[-1]
    if not (curr["close"] > curr["ema20"]):
        reasons.append("السعر تحت EMA20")
    if not (curr["close"] > curr["sma50"]):
        reasons.append("السعر تحت SMA50")
    if not (curr["macd"] > curr["macd_signal"]):
        reasons.append("MACD أقل من Signal")
    if not (curr["volume"] > 1.2 * curr["vol_ma20"]):
        reasons.append("حجم التداول أقل من 1.2x المتوسط")
    if not (30 < curr["rsi14"] < 75):
        reasons.append(f"RSI خارج النطاق ({curr['rsi14']:.1f})")
    dist = (curr["close"] - curr["ema20"]) / curr["ema20"]
    if dist > 0.07:
        reasons.append("السعر بعيد جداً عن المتوسط (>7%)")
    passed = len(reasons) == 0
    return passed, reasons

# ======================== حساب نقاط المجموعات ========================
def calculate_group_scores(feat_df: pd.DataFrame) -> dict:
    """حساب نقاط كل مجموعة (الاتجاه، الزخم، السيولة، الأنماط، المستويات)."""
    curr = feat_df.iloc[-1]
    prev = feat_df.iloc[-2]
    prev2 = feat_df.iloc[-3] if len(feat_df) >= 3 else prev
    scores = {}

    # 1. مجموعة الاتجاه (وزن 20)
    # شرط القمم والقيعان الصاعدة: آخر 5 قمم متزايدة وآخر 5 قيعان متزايدة
    highs_5 = feat_df["high"].iloc[-5:].values
    lows_5 = feat_df["low"].iloc[-5:].values
    trend_hh = all(highs_5[i] > highs_5[i-1] for i in range(1, len(highs_5)))
    trend_hl = all(lows_5[i] > lows_5[i-1] for i in range(1, len(lows_5)))
    
    trend_conditions = [
        curr["close"] > curr["ema20"],
        curr["close"] > curr["sma50"],
        trend_hh and trend_hl
    ]
    scores["trend"] = round((sum(trend_conditions) / len(trend_conditions)) * 20, 2)

    # 2. مجموعة الزخم (وزن 30)
    momentum_conditions = [
        curr["macd"] > curr["macd_signal"],
        40 < curr["rsi14"] < 70,
        curr["macd_hist"] > prev["macd_hist"],
        curr["stoch_k"] > curr["stoch_d"]
    ]
    scores["momentum"] = round((sum(momentum_conditions) / len(momentum_conditions)) * 30, 2)

    # 3. مجموعة السيولة (وزن 25)
    volume_conditions = [
        curr["volume"] > 1.2 * curr["vol_ma20"],
        curr["obv"] > prev["obv"],
        curr["volume"] > prev["volume"]
    ]
    scores["volume"] = round((sum(volume_conditions) / len(volume_conditions)) * 25, 2)

    # 4. مجموعة الأنماط (وزن 15)
    scores["pattern"] = get_candle_pattern_score(feat_df)

    # 5. مجموعة المستويات (وزن 10)
    fib_score = 0
    if len(feat_df) > 60:
        recent_high = feat_df["high"].iloc[-60:].max()
        recent_low = feat_df["low"].iloc[-60:].min()
        diff = recent_high - recent_low
        fib_levels = [recent_high - 0.382*diff, recent_high - 0.5*diff, recent_high - 0.618*diff]
        for level in fib_levels:
            if abs(curr["close"] - level) / level < 0.01:
                fib_score = 1
                break
    gap_up = 0
    if len(feat_df) > 1:
        prev_close = feat_df["close"].iloc[-2]
        if curr["open"] > prev_close * 1.01:
            gap_up = 1
    near_ema = 1 if abs((curr["close"] - curr["ema20"]) / curr["ema20"]) < 0.02 else 0
    level_conditions = [fib_score, gap_up, near_ema]
    scores["levels"] = round((sum(level_conditions) / len(level_conditions)) * 10, 2)

    scores["total"] = round(scores["trend"] + scores["momentum"] + scores["volume"] + scores["pattern"] + scores["levels"], 2)
    return scores
    
# ======================== دالة التحليل الرئيسية ========================
def analyze_one(ticker: str) -> Optional[Dict[str, Any]]:
    t = ticker.strip().upper()
    if not t.endswith(".SR"):
        t += ".SR"

    df = fetch_yahoo_prices(t)
    if df is None:
        return None

    try:
        feat_df = build_features(df)
    except Exception as e:
        logger.error(f"خطأ في بناء المؤشرات لـ {t}: {e}")
        return None

    excluded, exclude_reason = should_exclude(feat_df)
    if excluded:
        return {
            "ticker": t,
            "status": "EXCLUDED",
            "reason": exclude_reason,
            "confidence": 0,
            "entry": round(feat_df.iloc[-1]["close"], 2),
            "tp": 0,
            "sl": 0,
            "lastClose": round(feat_df.iloc[-1]["close"], 2)
        }

    passed, core_reasons = passes_core_rules(feat_df)
    if not passed:
        return {
            "ticker": t,
            "status": "REJECTED",
            "reason": " | ".join(core_reasons),
            "confidence": 0,
            "entry": round(feat_df.iloc[-1]["close"], 2),
            "tp": 0,
            "sl": 0,
            "lastClose": round(feat_df.iloc[-1]["close"], 2)
        }

    scores = calculate_group_scores(feat_df)
    confidence = scores["total"]

    curr = feat_df.iloc[-1]
    entry = round(curr["close"], 2)
    tp = round(entry * (1 + TP_PCT), 2)
    recent_low = feat_df["low"].tail(3).min()
    sl_candidate1 = recent_low * 0.99
    sl_candidate2 = entry - 2 * curr["atr14"]
    sl = round(min(sl_candidate1, sl_candidate2), 2)

    return {
        "ticker": t,
        "status": "APPROVED",
        "confidence": confidence,
        "entry": entry,
        "tp": tp,
        "sl": sl,
        "reason": "اجتاز الفلاتر الفنية",
        "lastClose": entry
    }

# ======================== Endpoints العامة ========================
@app.get("/health")
async def health_check():
    return {"status": "ok", "timestamp": time.time()}

@app.get("/predict")
async def predict(ticker: str = Query(..., description="رمز السهم مثال: 2222.SR")):
    result = analyze_one(ticker)
    if result is None:
        raise HTTPException(status_code=404, detail="لا توجد بيانات كافية لهذا السهم")
    return result

@app.get("/top10")
async def top10():
    if not os.path.exists(TICKERS_PATH):
        raise HTTPException(status_code=500, detail=f"ملف {TICKERS_PATH} غير موجود")

    with open(TICKERS_PATH, "r", encoding="utf-8") as f:
        tickers = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    results = []
    total_scanned = 0

    with ThreadPoolExecutor(max_workers=TOP10_WORKERS) as executor:
        futures = {executor.submit(analyze_one, t): t for t in tickers}
        for future in as_completed(futures):
            res = future.result()
            if res:
                total_scanned += 1
                if res["status"] == "APPROVED":
                    results.append(res)

    results.sort(key=lambda x: x["confidence"], reverse=True)
    return {
        "items": results[:10],
        "total_scanned": total_scanned,
        "timestamp": time.time()
    }

# ======================== نقاط نهاية التشخيص ========================
@app.get("/debug/ticker/{ticker}")
async def debug_ticker(ticker: str):
    """إرجاع معلومات تشخيصية مفصلة عن سهم واحد."""
    t = ticker.strip().upper()
    if not t.endswith(".SR"):
        t += ".SR"

    df = fetch_yahoo_prices(t)
    if df is None:
        return {"error": "لا توجد بيانات لهذا السهم"}

    try:
        feat_df = build_features(df)
    except Exception as e:
        return {"error": f"فشل بناء المؤشرات: {str(e)}"}

    if feat_df is None or len(feat_df) == 0:
        return {"error": "البيانات غير كافية لحساب المؤشرات"}

    raw_last5 = df.tail(5).to_dict(orient="records")
    curr = feat_df.iloc[-1].to_dict()
    for k, v in curr.items():
        if isinstance(v, (np.integer, np.floating)):
            curr[k] = float(v)
        elif isinstance(v, np.bool_):
            curr[k] = bool(v)

    passed, reasons = passes_core_rules(feat_df)
    excluded, exclude_reason = should_exclude(feat_df)
    scores = calculate_group_scores(feat_df)

    return {
        "ticker": t,
        "raw_last_5": raw_last5,
        "indicators_last": curr,
        "core_rules_passed": passed,
        "core_rules_reasons": reasons,
        "excluded": excluded,
        "exclude_reason": exclude_reason,
        "group_scores": scores
    }

def analyze_one_debug(ticker: str):
    """نسخة مبسطة من analyze_one لإرجاع الحالة والسبب فقط."""
    t = ticker.strip().upper()
    if not t.endswith(".SR"):
        t += ".SR"

    df = fetch_yahoo_prices(t)
    if df is None:
        return None

    try:
        feat_df = build_features(df)
    except Exception:
        return None

    excluded, exclude_reason = should_exclude(feat_df)
    if excluded:
        return {"status": "EXCLUDED", "reason": exclude_reason}

    passed, reasons = passes_core_rules(feat_df)
    if not passed:
        return {"status": "REJECTED", "reason": " | ".join(reasons)}

    return {"status": "APPROVED", "reason": ""}

@app.get("/debug/summary")
async def debug_summary():
    """تحليل جميع الأسهم وإرجاع إحصائيات حول أسباب الرفض والاستبعاد."""
    if not os.path.exists(TICKERS_PATH):
        raise HTTPException(status_code=500, detail=f"ملف {TICKERS_PATH} غير موجود")

    with open(TICKERS_PATH, "r", encoding="utf-8") as f:
        tickers = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    stats = {
        "total": 0,
        "excluded_count": 0,
        "excluded_reasons": {},
        "core_failed_count": 0,
        "core_failed_reasons": {},
        "approved_count": 0
    }

    with ThreadPoolExecutor(max_workers=TOP10_WORKERS) as executor:
        futures = {executor.submit(analyze_one_debug, t): t for t in tickers}
        for future in as_completed(futures):
            res = future.result()
            if res is None:
                continue
            stats["total"] += 1
            if res["status"] == "EXCLUDED":
                stats["excluded_count"] += 1
                reason = res["reason"]
                stats["excluded_reasons"][reason] = stats["excluded_reasons"].get(reason, 0) + 1
            elif res["status"] == "REJECTED":
                stats["core_failed_count"] += 1
                reasons_list = res["reason"].split(" | ")
                for r in reasons_list:
                    stats["core_failed_reasons"][r] = stats["core_failed_reasons"].get(r, 0) + 1
            elif res["status"] == "APPROVED":
                stats["approved_count"] += 1

    return stats

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
