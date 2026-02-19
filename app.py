from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import requests
import pandas as pd
import numpy as np
import time
import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict, Any, List, Tuple
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

# ======================== إعدادات استراتيجية الصعود (العودة إلى الحالة القديمة) ========================
UPTREND_MIN_VOLUME = int(os.getenv("UPTREND_MIN_VOLUME", "200000"))
UPTREND_MIN_PRICE = float(os.getenv("UPTREND_MIN_PRICE", "5.0"))
UPTREND_ATR_LIMIT = float(os.getenv("UPTREND_ATR_LIMIT", "6.5"))  # العودة إلى 6.5 كما كانت سابقاً
UPTREND_RSI_MIN = float(os.getenv("UPTREND_RSI_MIN", "20"))       # العودة إلى 20
UPTREND_RSI_MAX = float(os.getenv("UPTREND_RSI_MAX", "80"))       # العودة إلى 80
UPTREND_VOL_RATIO = float(os.getenv("UPTREND_VOL_RATIO", "0.5"))  # العودة إلى 0.5
UPTREND_DIST_MIN = float(os.getenv("UPTREND_DIST_MIN", "-10.0"))  # العودة إلى -10%
UPTREND_DIST_MAX = float(os.getenv("UPTREND_DIST_MAX", "11.0"))   # العودة إلى +11%
UPTREND_REQUIRE_MACD = os.getenv("UPTREND_REQUIRE_MACD", "false").lower() == "true"  # العودة إلى False

# ======================== إعدادات استراتيجية القيعان (كما هي) ========================
BOTTOM_MIN_VOLUME = int(os.getenv("BOTTOM_MIN_VOLUME", "200000"))
BOTTOM_MIN_PRICE = float(os.getenv("BOTTOM_MIN_PRICE", "5.0"))
BOTTOM_ATR_LIMIT = float(os.getenv("BOTTOM_ATR_LIMIT", "6.0"))
BOTTOM_RSI_MAX = float(os.getenv("BOTTOM_RSI_MAX", "45"))
BOTTOM_DIST_MIN = float(os.getenv("BOTTOM_DIST_MIN", "-12.0"))
BOTTOM_DIST_MAX = float(os.getenv("BOTTOM_DIST_MAX", "-2.0"))
BOTTOM_MIN_VOL_RATIO = float(os.getenv("BOTTOM_MIN_VOL_RATIO", "0.8"))
BOTTOM_PATTERN_WEIGHT = int(os.getenv("BOTTOM_PATTERN_WEIGHT", "30"))
BOTTOM_RSI_WEIGHT = int(os.getenv("BOTTOM_RSI_WEIGHT", "20"))
BOTTOM_VOL_WEIGHT = int(os.getenv("BOTTOM_VOL_WEIGHT", "15"))
BOTTOM_DIST_WEIGHT = int(os.getenv("BOTTOM_DIST_WEIGHT", "15"))
BOTTOM_BB_WEIGHT = int(os.getenv("BOTTOM_BB_WEIGHT", "10"))

# ======================== كاش بسيط يدوي ========================
_prices_cache = {}

# ======================== إنشاء تطبيق FastAPI ========================
app = FastAPI(
    title="Tadawul Sniper Pro",
    description="محلل فني لأسهم السوق السعودي - يدعم استراتيجيتي الصعود واقتناص القيعان",
    version="5.0.1"  # تحديث الإصدار
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

# ======================== حساب المؤشرات الفنية ========================
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

    # Bollinger Bands
    d["bb_mid"] = d["sma20"]
    d["bb_std"] = close.rolling(20).std()
    d["bb_upper"] = d["bb_mid"] + 2 * d["bb_std"]
    d["bb_lower"] = d["bb_mid"] - 2 * d["bb_std"]

    # OBV
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
    d["vol_ratio"] = volume / d["vol_ma20"]

    d["candle_green"] = close > d["open"]
    d["body"] = abs(close - d["open"])
    d["upper_shadow"] = high - d[["close", "open"]].max(axis=1)
    d["lower_shadow"] = d[["close", "open"]].min(axis=1) - low
    d["dist_from_ema20"] = (close - d["ema20"]) / d["ema20"] * 100

    # أنماط انعكاسية مبسطة
    d["prev_close"] = close.shift(1)
    d["prev_open"] = d["open"].shift(1)

    # مطرقة (Hammer)
    body = d["body"]
    lower = d["lower_shadow"]
    upper = d["upper_shadow"]
    d["is_hammer"] = (lower > 2 * body) & (upper < 0.3 * body)

    # ابتلاع شرائي (Bullish Engulfing)
    d["is_bullish_engulfing"] = (
        (d["prev_close"] < d["prev_open"]) &          # شمعة سابقة حمراء
        (close > d["open"]) &                         # شمعة حالية خضراء
        (d["open"] < d["prev_close"]) &               # تفتح تحت إغلاق السابقة
        (close > d["prev_open"])                       # تغلق فوق افتتاح السابقة
    )

    # دوجي (Doji)
    d["is_doji"] = d["body"] < 0.1 * (high - low)

    d = d.dropna().reset_index(drop=True)
    return d

# ======================== دوال الأنماط الهابطة (للاستبعاد) ========================
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

# ======================== دوال حساب الثقة للصعود (العودة إلى الإصدار القديم) ========================
def get_candle_pattern_score(feat_df):
    if len(feat_df) < 3:
        return 0
    last = feat_df.iloc[-1]
    prev = feat_df.iloc[-2]
    prev2 = feat_df.iloc[-3]

    score = 0
    if 'is_bullish_engulfing' in last and last['is_bullish_engulfing']:
        score += 15
    elif 'is_hammer' in last and last['is_hammer'] and (prev["close"] < prev["open"]):
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

def calculate_uptrend_confidence(feat_df: pd.DataFrame) -> float:
    """حساب الثقة للاستراتيجية الصاعدة (كما كانت سابقاً)"""
    curr = feat_df.iloc[-1]
    prev = feat_df.iloc[-2]
    scores = {}

    # مجموعة الاتجاه (20)
    highs_5 = feat_df["high"].iloc[-5:].values
    lows_5 = feat_df["low"].iloc[-5:].values
    trend_hh = all(highs_5[i] > highs_5[i-1] for i in range(1, len(highs_5)))
    trend_hl = all(lows_5[i] > lows_5[i-1] for i in range(1, len(lows_5)))
    trend_conditions = [
        curr["close"] > curr["ema20"],
        curr["close"] > curr["sma50"],
        trend_hh and trend_hl
    ]
    scores["trend"] = (sum(trend_conditions) / 3) * 20

    # مجموعة الزخم (30)
    momentum_conditions = [
        curr["macd"] > curr["macd_signal"],
        UPTREND_RSI_MIN < curr["rsi14"] < UPTREND_RSI_MAX,
        curr["macd_hist"] > prev["macd_hist"],
        curr["stoch_k"] > curr["stoch_d"]
    ]
    scores["momentum"] = (sum(momentum_conditions) / 4) * 30

    # مجموعة السيولة (25)
    volume_conditions = [
        curr["volume"] > UPTREND_VOL_RATIO * curr["vol_ma20"],
        curr["obv"] > prev["obv"],
        curr["volume"] > prev["volume"]
    ]
    scores["volume"] = (sum(volume_conditions) / 3) * 25

    # مجموعة الأنماط (15)
    scores["pattern"] = get_candle_pattern_score(feat_df)

    # مجموعة المستويات (10)
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
    near_ema = 1 if abs(curr["dist_from_ema20"]) < 2 else 0
    level_conditions = [fib_score, gap_up, near_ema]
    scores["levels"] = (sum(level_conditions) / 3) * 10

    total = scores["trend"] + scores["momentum"] + scores["volume"] + scores["pattern"] + scores["levels"]
    return total

# ======================== حساب الثقة لاستراتيجية القيعان (كما هي) ========================
def calculate_bottom_confidence(feat_df: pd.DataFrame) -> float:
    curr = feat_df.iloc[-1]

    score = 0
    # 1. النمط الانعكاسي
    pattern_score = 0
    if curr.get("is_hammer", False):
        pattern_score = 30
    elif curr.get("is_bullish_engulfing", False):
        pattern_score = 30
    elif curr.get("is_doji", False):
        pattern_score = 15
    score += pattern_score

    # 2. RSI
    rsi = curr["rsi14"]
    if rsi < BOTTOM_RSI_MAX:
        rsi_points = max(0, BOTTOM_RSI_WEIGHT * (1 - (rsi - 20) / (BOTTOM_RSI_MAX - 20))) if rsi > 20 else BOTTOM_RSI_WEIGHT
        score += rsi_points

    # 3. حجم يوم النمط
    vol_ratio = curr.get("vol_ratio", 0)
    if vol_ratio >= BOTTOM_MIN_VOL_RATIO:
        vol_points = min(BOTTOM_VOL_WEIGHT, vol_ratio * 5)
        score += vol_points

    # 4. المسافة تحت EMA20
    dist = curr["dist_from_ema20"]
    if dist < 0:
        dist_points = min(BOTTOM_DIST_WEIGHT, abs(dist) * 1.5)
        score += dist_points

    # 5. ملامسة الحد السفلي لبولينجر
    if curr["close"] <= curr["bb_lower"]:
        score += BOTTOM_BB_WEIGHT
    elif curr["close"] < curr["bb_mid"]:
        score += BOTTOM_BB_WEIGHT // 2

    return min(100, max(0, score))

# ======================== شروط القبول للصعود (العودة إلى الإصدار القديم) ========================
def passes_uptrend(feat_df: pd.DataFrame) -> Tuple[bool, List[str]]:
    reasons = []
    curr = feat_df.iloc[-1]
    prev = feat_df.iloc[-2] if len(feat_df) > 1 else curr

    # الشروط الحالية (RSI، حجم، مسافة، SMA50، MACD، ...)
    if not (UPTREND_RSI_MIN < curr["rsi14"] < UPTREND_RSI_MAX):
        reasons.append(f"RSI خارج {UPTREND_RSI_MIN}-{UPTREND_RSI_MAX}")
    if not (curr["volume"] > UPTREND_VOL_RATIO * curr["vol_ma20"]):
        reasons.append(f"حجم < {UPTREND_VOL_RATIO}x المتوسط")
    dist = curr["dist_from_ema20"]
    if not (UPTREND_DIST_MIN <= dist <= UPTREND_DIST_MAX):
        reasons.append(f"المسافة عن EMA20 خارج [{UPTREND_DIST_MIN},{UPTREND_DIST_MAX}]%")
    if not (curr["close"] > curr["sma50"]):
        reasons.append("السعر تحت SMA50")
    if UPTREND_REQUIRE_MACD and not (curr["macd"] > curr["macd_signal"]):
        reasons.append("MACD أقل من Signal")

    # 🆕 شروط اتجاه المؤشرات
    # 1. RSI في ارتفاع (أو على الأقل ليس في هبوط حاد)
    if curr["rsi14"] < prev["rsi14"] - 5:  # انخفض بأكثر من 5 نقاط
        reasons.append(f"RSI في هبوط حاد ({prev['rsi14']:.1f} → {curr['rsi14']:.1f})")
    
    # 2. MACD Histogram في ارتفاع
    if curr["macd_hist"] < prev["macd_hist"]:
        reasons.append("MACD Histogram في هبوط")
    
    # 3. Stochastic في حالة إيجابية (%K فوق %D)
    if curr["stoch_k"] < curr["stoch_d"]:
        reasons.append("تقاطع Stochastic سلبي (%K تحت %D)")
    
    # 4. OBV في ارتفاع (تدفق سيولة إيجابي)
    if curr["obv"] < prev["obv"]:
        reasons.append("OBV في هبوط")

    # شروط استبعاد أساسية
    if curr["volume"] < UPTREND_MIN_VOLUME:
        reasons.append(f"حجم < {UPTREND_MIN_VOLUME}")
    if curr["close"] < UPTREND_MIN_PRICE:
        reasons.append(f"سعر < {UPTREND_MIN_PRICE}")
    if curr["atr_pct"] > UPTREND_ATR_LIMIT:
        reasons.append(f"تقلب > {UPTREND_ATR_LIMIT}%")
    if has_bearish_pattern(feat_df.tail(4)):
        reasons.append("نمط هابط")

    passed = len(reasons) == 0
    return passed, reasons

# ======================== شروط القبول للقيعان (كما هي) ========================
def passes_bottom(feat_df: pd.DataFrame) -> Tuple[bool, List[str]]:
    reasons = []
    curr = feat_df.iloc[-1]

    if curr["rsi14"] > BOTTOM_RSI_MAX:
        reasons.append(f"RSI > {BOTTOM_RSI_MAX}")

    dist = curr["dist_from_ema20"]
    if not (BOTTOM_DIST_MIN <= dist <= BOTTOM_DIST_MAX):
        reasons.append(f"المسافة عن EMA20 خارج [{BOTTOM_DIST_MIN},{BOTTOM_DIST_MAX}]%")

    has_pattern = curr.get("is_hammer", False) or curr.get("is_bullish_engulfing", False) or curr.get("is_doji", False)
    if not has_pattern:
        reasons.append("لا يوجد نمط انعكاسي")

    if curr["vol_ratio"] < BOTTOM_MIN_VOL_RATIO:
        reasons.append(f"حجم يوم النمط < {BOTTOM_MIN_VOL_RATIO}x المتوسط")

    if curr["volume"] < BOTTOM_MIN_VOLUME:
        reasons.append(f"حجم < {BOTTOM_MIN_VOLUME}")
    if curr["close"] < BOTTOM_MIN_PRICE:
        reasons.append(f"سعر < {BOTTOM_MIN_PRICE}")
    if curr["atr_pct"] > BOTTOM_ATR_LIMIT:
        reasons.append(f"تقلب > {BOTTOM_ATR_LIMIT}%")

    passed = len(reasons) == 0
    return passed, reasons

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

    curr = feat_df.iloc[-1]
    entry = round(curr["close"], 2)
    tp = round(entry * (1 + TP_PCT), 2)
    recent_low = feat_df["low"].tail(3).min()
    sl_candidate1 = recent_low * 0.99
    sl_candidate2 = entry - 2 * curr["atr14"]
    sl = round(min(sl_candidate1, sl_candidate2), 2)

    # تحليل الصعود
    uptrend_passed, uptrend_reasons = passes_uptrend(feat_df)
    uptrend_conf = calculate_uptrend_confidence(feat_df) / 100.0 if uptrend_passed else 0.0
    uptrend_result = {
        "status": "APPROVED" if uptrend_passed else "REJECTED",
        "confidence": uptrend_conf,
        "entry": entry,
        "tp": tp,
        "sl": sl,
        "reason": "اجتاز فلاتر الصعود" if uptrend_passed else " | ".join(uptrend_reasons)
    }

    # تحليل القيعان
    bottom_passed, bottom_reasons = passes_bottom(feat_df)
    bottom_conf = calculate_bottom_confidence(feat_df) / 100.0 if bottom_passed else 0.0
    bottom_result = {
        "status": "APPROVED" if bottom_passed else "REJECTED",
        "confidence": bottom_conf,
        "entry": entry,
        "tp": tp,
        "sl": sl,
        "reason": "اجتاز فلاتر القيعان" if bottom_passed else " | ".join(bottom_reasons)
    }

    return {
        "ticker": t,
        "lastClose": entry,
        "uptrend": uptrend_result,
        "bottom": bottom_result
    }

# ======================== نقطة النهاية القديمة (للتوافق) ========================
@app.get("/predict")
async def predict(ticker: str = Query(..., description="رمز السهم مثال: 2222.SR")):
    """نقطة النهاية القديمة - تعيد بيانات استراتيجية الصعود فقط (بنفس التنسيق السابق)"""
    result = analyze_one(ticker)
    if result is None:
        raise HTTPException(status_code=404, detail="لا توجد بيانات كافية لهذا السهم")
    # نعيد فقط بيانات الصعود مع الحقول المطلوبة من الواجهة القديمة
    uptrend = result["uptrend"]
    return {
        "ticker": result["ticker"],
        "status": uptrend["status"],
        "recommendation": "BUY" if uptrend["status"] == "APPROVED" else "NO_TRADE",
        "confidence": uptrend["confidence"],
        "entry": uptrend["entry"],
        "tp": uptrend["tp"],
        "sl": uptrend["sl"],
        "reason": uptrend["reason"],
        "lastClose": result["lastClose"]
    }

# ======================== نقطة نهاية جديدة للاستراتيجيتين ========================
@app.get("/predict/dual")
async def predict_dual(ticker: str = Query(..., description="رمز السهم مثال: 2222.SR")):
    """نقطة نهاية جديدة تعيد بيانات كلتا الاستراتيجيتين"""
    result = analyze_one(ticker)
    if result is None:
        raise HTTPException(status_code=404, detail="لا توجد بيانات كافية لهذا السهم")
    return result

# ======================== نقطة نهاية top10 المعدلة ========================
@app.get("/top10")
async def top10():
    if not os.path.exists(TICKERS_PATH):
        raise HTTPException(status_code=500, detail=f"ملف {TICKERS_PATH} غير موجود")

    with open(TICKERS_PATH, "r", encoding="utf-8") as f:
        tickers = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    uptrend_results = []
    bottom_results = []
    total_scanned = 0

    with ThreadPoolExecutor(max_workers=TOP10_WORKERS) as executor:
        futures = {executor.submit(analyze_one, t): t for t in tickers}
        for future in as_completed(futures):
            res = future.result()
            if res:
                total_scanned += 1
                if res["uptrend"]["status"] == "APPROVED":
                    uptrend_results.append({
                        "ticker": res["ticker"],
                        "confidence": res["uptrend"]["confidence"],
                        "entry": res["uptrend"]["entry"],
                        "tp": res["uptrend"]["tp"],
                        "sl": res["uptrend"]["sl"],
                        "reason": res["uptrend"]["reason"],
                        "lastClose": res["lastClose"]
                    })
                if res["bottom"]["status"] == "APPROVED":
                    bottom_results.append({
                        "ticker": res["ticker"],
                        "confidence": res["bottom"]["confidence"],
                        "entry": res["bottom"]["entry"],
                        "tp": res["bottom"]["tp"],
                        "sl": res["bottom"]["sl"],
                        "reason": res["bottom"]["reason"],
                        "lastClose": res["lastClose"]
                    })

    # ترتيب تنازلي حسب الثقة
    uptrend_results.sort(key=lambda x: x["confidence"], reverse=True)
    bottom_results.sort(key=lambda x: x["confidence"], reverse=True)

    return {
        "uptrend": uptrend_results[:10],
        "bottom": bottom_results[:10],
        "total_scanned": total_scanned,
        "timestamp": time.time()
    }

# ======================== نقاط التشخيص (اختياري) ========================
@app.get("/debug/ticker/{ticker}")
async def debug_ticker(ticker: str):
    t = ticker.strip().upper()
    if not t.endswith(".SR"):
        t += ".SR"

    df = fetch_yahoo_prices(t)
    if df is None:
        return {"error": "لا توجد بيانات"}

    feat_df = build_features(df)
    curr = feat_df.iloc[-1].to_dict()
    for k, v in curr.items():
        if isinstance(v, (np.integer, np.floating)):
            curr[k] = float(v)
        elif isinstance(v, np.bool_):
            curr[k] = bool(v)

    return {
        "ticker": t,
        "last_indicators": curr,
        "uptrend": passes_uptrend(feat_df)[0],
        "uptrend_confidence": calculate_uptrend_confidence(feat_df),
        "bottom": passes_bottom(feat_df)[0],
        "bottom_confidence": calculate_bottom_confidence(feat_df)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
