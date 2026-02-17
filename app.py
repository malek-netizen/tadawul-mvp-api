# main.py
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import requests
import pandas as pd
import pandas_ta as ta
import numpy as np
import time
import os
import logging
from cachetools import TTLCache
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, List, Dict, Any
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
TP_PCT = float(os.getenv("TP_PCT", "0.05"))          # هدف الربح 5%
TOP10_WORKERS = int(os.getenv("TOP10_WORKERS", "10"))
CACHE_TTL_SEC = int(os.getenv("CACHE_TTL_SEC", "600"))  # 10 دقائق
MIN_VOLUME = int(os.getenv("MIN_VOLUME", "500000"))     # الحد الأدنى لحجم التداول
MIN_PRICE = float(os.getenv("MIN_PRICE", "10.0"))       # الحد الأدنى للسعر
ATR_EXCLUDE_PCT = float(os.getenv("ATR_EXCLUDE_PCT", "3.0"))  # استبعاد إذا ATR% > 3%
MAX_5DAY_GAIN = float(os.getenv("MAX_5DAY_GAIN", "0.10"))      # استبعاد إذا الارتفاع في 5 أيام > 10%

# ======================== الكاش (للبيانات) ========================
cache = TTLCache(maxsize=200, ttl=CACHE_TTL_SEC)

# ======================== إنشاء تطبيق FastAPI ========================
app = FastAPI(
    title="Tadawul Sniper Pro",
    description="محلل فني لأسهم السوق السعودي – يحدد أفضل 10 فرص للمضاربة",
    version="4.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================== دوال جلب البيانات من Yahoo Finance ========================
def fetch_yahoo_prices(ticker: str, range_: str = "1y", interval: str = "1d") -> Optional[pd.DataFrame]:
    """
    جلب بيانات السهم من Yahoo Finance مع تخزين مؤقت.
    تعيد DataFrame يحتوي على open, high, low, close, volume أو None عند الفشل.
    """
    key = (ticker, range_, interval)
    if key in cache:
        logger.info(f"استخدام البيانات المخزنة لـ {ticker}")
        return cache[key]

    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    headers = {"User-Agent": "Mozilla/5.0"}
    params = {"range": range_, "interval": interval}
    try:
        r = requests.get(url, params=params, headers=headers, timeout=15)
        r.raise_for_status()
        js = r.json()
        result = js['chart']['result'][0]
        quote = result['indicators']['quote'][0]
        # التأكد من وجود البيانات المطلوبة
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
        cache[key] = df
        logger.info(f"تم جلب بيانات {ticker} بنجاح ({len(df)} يوم)")
        return df
    except Exception as e:
        logger.error(f"خطأ في جلب {ticker}: {str(e)}")
        return None

# ======================== حساب المؤشرات الفنية باستخدام pandas_ta ========================
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """إضافة المؤشرات الفنية إلى DataFrame."""
    d = df.copy()
    # استخدام مكتبة pandas_ta لحساب المؤشرات
    d.ta.ema(length=20, append=True)          # EMA_20
    d.ta.sma(length=20, append=True)          # SMA_20
    d.ta.sma(length=50, append=True)          # SMA_50
    d.ta.macd(append=True)                    # MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
    d.ta.rsi(length=14, append=True)          # RSI_14
    d.ta.bbands(length=20, std=2, append=True) # BBL_20_2.0, BBM_20_2.0, BBU_20_2.0
    d.ta.obv(append=True)                      # OBV
    d.ta.stoch(append=True)                    # STOCHk_14_3_3, STOCHd_14_3_3
    d.ta.atr(length=14, append=True)           # ATR_14
    d.ta.adx(length=14, append=True)           # ADX_14

    # إعادة تسمية الأعمدة لتكون قصيرة ومتسقة
    rename_dict = {
        "EMA_20": "ema20",
        "SMA_20": "sma20",
        "SMA_50": "sma50",
        "RSI_14": "rsi14",
        "BBM_20_2.0": "bb_mid",
        "BBU_20_2.0": "bb_upper",
        "ATR_14": "atr14",
        "OBV": "obv",
        "STOCHk_14_3_3": "stoch_k",
        "STOCHd_14_3_3": "stoch_d",
        "ADX_14": "adx"
    }
    d.rename(columns=rename_dict, inplace=True)

    # أعمدة MACD
    if "MACD_12_26_9" in d.columns:
        d.rename(columns={"MACD_12_26_9": "macd"}, inplace=True)
    if "MACDh_12_26_9" in d.columns:
        d.rename(columns={"MACDh_12_26_9": "macd_hist"}, inplace=True)
    if "MACDs_12_26_9" in d.columns:
        d.rename(columns={"MACDs_12_26_9": "macd_signal"}, inplace=True)

    # إضافة vol_ma20 و vol_std يدوياً
    d["vol_ma20"] = d["volume"].rolling(20).mean()
    d["vol_std"] = d["volume"].rolling(20).std()
    d["atr_pct"] = d["atr14"] / d["close"] * 100

    # إضافة عمود لون الشمعة
    d["candle_green"] = d["close"] > d["open"]
    d["body"] = abs(d["close"] - d["open"])
    d["upper_shadow"] = d["high"] - d[["close", "open"]].max(axis=1)
    d["lower_shadow"] = d[["close", "open"]].min(axis=1) - d["low"]

    # حذف الصفوف الأولى التي تحتوي NaN
    d.dropna(inplace=True)
    d.reset_index(drop=True, inplace=True)
    return d

# ======================== دوال الأنماط السعرية ========================
def is_bullish_engulfing(prev: pd.Series, curr: pd.Series) -> bool:
    """نمط ابتلاع شرائي: شمعة حمراء تتبعها شمعة خضراء تبتلعها بالكامل."""
    prev_red = prev["close"] < prev["open"]
    curr_green = curr["close"] > curr["open"]
    if not (prev_red and curr_green):
        return False
    return curr["open"] < prev["close"] and curr["close"] > prev["open"]

def is_hammer(candle: pd.Series) -> bool:
    """نمط مطرقة: ظل سفلي طويل (ضعف الجسم على الأقل) وظل علوي قصير."""
    body = candle["body"]
    lower = candle["lower_shadow"]
    upper = candle["upper_shadow"]
    return lower > 2 * body and upper < 0.3 * body

def is_morning_star(c1: pd.Series, c2: pd.Series, c3: pd.Series) -> bool:
    """نمط نجمة صباحية مبسطة: شمعة حمراء، شمعة صغيرة، شمعة خضراء تغلق فوق منتصف الأولى."""
    c1_red = c1["close"] < c1["open"]
    c3_green = c3["close"] > c3["open"]
    c2_small = c2["body"] < 0.1 * (c1["high"] - c1["low"])
    if not (c1_red and c3_green and c2_small):
        return False
    midpoint = (c1["open"] + c1["close"]) / 2
    return c3["close"] > midpoint

def is_piercing(prev: pd.Series, curr: pd.Series) -> bool:
    """نمط الاختراق الصاعد: شمعة حمراء تليها شمعة خضراء تغلق فوق منتصف الحمراء."""
    if not (prev["close"] < prev["open"] and curr["close"] > curr["open"]):
        return False
    midpoint = (prev["open"] + prev["close"]) / 2
    return curr["close"] > midpoint and curr["open"] < prev["close"]

def get_candle_pattern_score(feat_df: pd.DataFrame) -> int:
    """تقييم أنماط الشموع في آخر 3 أيام، وإرجاع نقاط إضافية (بحد أقصى 15)."""
    if len(feat_df) < 3:
        return 0
    last = feat_df.iloc[-1]
    prev = feat_df.iloc[-2]
    prev2 = feat_df.iloc[-3]

    score = 0
    # الأنماط الصاعدة القوية
    if is_bullish_engulfing(prev, last):
        score += 15
    elif is_morning_star(prev2, prev, last):
        score += 15
    elif is_piercing(prev, last):
        score += 12
    elif is_hammer(last) and (prev["close"] < prev["open"]):  # مطرقة بعد هبوط
        score += 12
    elif last["candle_green"]:
        # شمعة خضراء عادية: نعطي نقاطاً حسب حجم الجسم
        body_pct = last["body"] / (last["high"] - last["low"]) if (last["high"]-last["low"])>0 else 0
        if body_pct > 0.5:
            score += 5
        else:
            score += 2

    # خصم بسيط إذا كان هناك ظل علوي طويل
    if last["upper_shadow"] > 2 * last["body"]:
        score -= 3

    return max(0, min(15, score))

# ======================== دوال الأنماط الهابطة (للاستبعاد) ========================
def is_bearish_engulfing(prev: pd.Series, curr: pd.Series) -> bool:
    prev_green = prev["close"] > prev["open"]
    curr_red = curr["close"] < curr["open"]
    if not (prev_green and curr_red):
        return False
    return curr["open"] > prev["close"] and curr["close"] < prev["open"]

def is_shooting_star(candle: pd.Series) -> bool:
    body = candle["body"]
    upper = candle["upper_shadow"]
    lower = candle["lower_shadow"]
    return upper > 2 * body and lower < 0.3 * body

def is_dark_cloud(prev: pd.Series, curr: pd.Series) -> bool:
    if not (prev["close"] > prev["open"] and curr["close"] < curr["open"]):
        return False
    midpoint = (prev["open"] + prev["close"]) / 2
    return curr["open"] > prev["high"] and curr["close"] < midpoint

def is_hanging_man(candle: pd.Series) -> bool:
    body = candle["body"]
    lower = candle["lower_shadow"]
    upper = candle["upper_shadow"]
    return lower > 2 * body and upper < 0.3 * body

def is_evening_star(c1: pd.Series, c2: pd.Series, c3: pd.Series) -> bool:
    c1_green = c1["close"] > c1["open"]
    c3_red = c3["close"] < c3["open"]
    c2_small = c2["body"] < 0.1 * (c1["high"] - c1["low"])
    if not (c1_green and c3_red and c2_small):
        return False
    midpoint = (c1["open"] + c1["close"]) / 2
    return c3["close"] < midpoint

def has_bearish_pattern(feat_df: pd.DataFrame) -> bool:
    """التحقق من وجود أي نمط هابط قوي في آخر يومين."""
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
    # 1. التقلب العالي (ATR%)
    if curr["atr_pct"] > ATR_EXCLUDE_PCT:
        return True, f"تقلب عالي (ATR% = {curr['atr_pct']:.1f}%)"

    # 2. ارتفاع كبير في آخر 5 أيام
    if len(feat_df) >= 6:
        close_5 = feat_df["close"].iloc[-6]
        gain_5 = (curr["close"] / close_5 - 1)
        if gain_5 > MAX_5DAY_GAIN:
            return True, f"ارتفاع كبير في 5 أيام ({gain_5*100:.1f}%)"

    # 3. سيولة منخفضة جداً
    if curr["volume"] < MIN_VOLUME:
        return True, f"سيولة منخفضة ({curr['volume']:,.0f})"

    # 4. سعر منخفض جداً
    if curr["close"] < MIN_PRICE:
        return True, f"سعر منخفض جداً ({curr['close']:.2f})"

    # 5. وجود نمط هابط
    if has_bearish_pattern(feat_df.tail(4)):
        return True, "وجود نمط هابط"

    return False, ""

# ======================== الشروط الأساسية (إلزامية) ========================
def passes_core_rules(feat_df: pd.DataFrame) -> tuple[bool, list[str]]:
    """الشروط الأساسية التي يجب توفرها للنظر في السهم."""
    reasons = []
    curr = feat_df.iloc[-1]

    # 1. السعر فوق EMA20
    if not (curr["close"] > curr["ema20"]):
        reasons.append("السعر تحت EMA20")

    # 2. السعر فوق SMA50
    if not (curr["close"] > curr["sma50"]):
        reasons.append("السعر تحت SMA50")

    # 3. MACD > Signal
    if not (curr["macd"] > curr["macd_signal"]):
        reasons.append("MACD أقل من Signal")

    # 4. حجم التداول > 1.5 × متوسط 20 يوم
    if not (curr["volume"] > 1.5 * curr["vol_ma20"]):
        reasons.append("حجم التداول أقل من 1.5x المتوسط")

    # 5. RSI بين 40 و 70
    if not (40 < curr["rsi14"] < 70):
        reasons.append(f"RSI خارج النطاق ({curr['rsi14']:.1f})")

    # 6. البعد عن EMA20 لا يزيد عن 5%
    dist = (curr["close"] - curr["ema20"]) / curr["ema20"]
    if dist > 0.05:
        reasons.append("السعر بعيد جداً عن المتوسط (>5%)")

    passed = len(reasons) == 0
    return passed, reasons

# ======================== حساب الثقة حسب المجموعات الخمس ========================
def calculate_group_scores(feat_df: pd.DataFrame) -> dict:
    """
    حساب نقاط كل مجموعة (0-100) بناءً على الشروط المحققة داخل المجموعة.
    المجموعات: الاتجاه، الزخم، السيولة، الأنماط، المستويات.
    """
    curr = feat_df.iloc[-1]
    prev = feat_df.iloc[-2]
    prev2 = feat_df.iloc[-3] if len(feat_df) >= 3 else prev

    scores = {}

    # 1. مجموعة الاتجاه (وزن 20) – 4 شروط
    trend_conditions = [
        curr["close"] > curr["ema20"],
        curr["close"] > curr["sma50"],
        curr["adx"] > 20 if "adx" in curr else False,
        # شرط القمم والقيعان الصاعدة (HH/HL) خلال آخر 5 أيام (تبسيط)
        (feat_df["high"].iloc[-5:].is_monotonic_increasing and feat_df["low"].iloc[-5:].is_monotonic_increasing)
    ]
    trend_score = (sum(trend_conditions) / len(trend_conditions)) * 20
    scores["trend"] = round(trend_score, 2)

    # 2. مجموعة الزخم (وزن 30) – 4 شروط
    momentum_conditions = [
        curr["macd"] > curr["macd_signal"],
        40 < curr["rsi14"] < 70,
        curr["macd_hist"] > prev["macd_hist"],
        curr["stoch_k"] > curr["stoch_d"]
    ]
    momentum_score = (sum(momentum_conditions) / len(momentum_conditions)) * 30
    scores["momentum"] = round(momentum_score, 2)

    # 3. مجموعة السيولة (وزن 25) – 3 شروط
    volume_conditions = [
        curr["volume"] > 1.5 * curr["vol_ma20"],
        curr["obv"] > prev["obv"],
        curr["volume"] > prev["volume"]
    ]
    volume_score = (sum(volume_conditions) / len(volume_conditions)) * 25
    scores["volume"] = round(volume_score, 2)

    # 4. مجموعة الأنماط السعرية (وزن 15) – نستخدم get_candle_pattern_score والتي تعطي 0-15
    pattern_score = get_candle_pattern_score(feat_df)
    scores["pattern"] = pattern_score  # بالفعل من 0 إلى 15

    # 5. مجموعة المستويات (وزن 10) – 3 شروط (فيبوناتشي، فجوة، قرب من المتوسط)
    # (فيبوناتشي سنبسطه: نعتبر أن السعر عند مستوى فيبوناتشي إذا كان ضمن 1% من قمة/قاع سابقة)
    # حساب فيبوناتشي تقريبي: نأخذ آخر 60 يوم ونحسب مستويات 0.382 و 0.618
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
    # فجوة صاعدة
    gap_up = 0
    if "open" in curr and len(feat_df) > 1:
        prev_close = feat_df["close"].iloc[-2]
        if curr["open"] > prev_close * 1.01:
            gap_up = 1
    # قرب من المتوسط (ضمن 2%)
    near_ema = 1 if abs((curr["close"] - curr["ema20"]) / curr["ema20"]) < 0.02 else 0

    level_conditions = [fib_score, gap_up, near_ema]
    level_score = (sum(level_conditions) / len(level_conditions)) * 10
    scores["levels"] = round(level_score, 2)

    # المجموع الكلي
    total = scores["trend"] + scores["momentum"] + scores["volume"] + scores["pattern"] + scores["levels"]
    scores["total"] = round(total, 2)
    return scores

# ======================== دالة التحليل الرئيسية ========================
def analyze_one(ticker: str) -> Optional[Dict[str, Any]]:
    """تحليل سهم واحد وإرجاع النتيجة بالحقول المطلوبة للواجهة."""
    t = ticker.strip().upper()
    if not t.endswith(".SR"):
        t += ".SR"

    df = fetch_yahoo_prices(t)
    if df is None:
        logger.warning(f"لا توجد بيانات لـ {t}")
        return None

    try:
        feat_df = build_features(df)
    except Exception as e:
        logger.error(f"خطأ في بناء المؤشرات لـ {t}: {e}")
        return None

    # التحقق من الاستبعاد أولاً
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

    # الشروط الأساسية
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

    # حساب الثقة حسب المجموعات
    scores = calculate_group_scores(feat_df)
    confidence = scores["total"]

    # إعدادات الهدف ووقف الخسارة
    curr = feat_df.iloc[-1]
    entry = round(curr["close"], 2)
    tp = round(entry * (1 + TP_PCT), 2)
    # وقف الخسارة: أدنى سعر في آخر 3 أيام * 0.99 (أو 2*ATR أيهما أقرب للسعر)
    recent_low = feat_df["low"].tail(3).min()
    sl_candidate1 = recent_low * 0.99
    sl_candidate2 = entry - 2 * curr["atr14"]
    sl = round(min(sl_candidate1, sl_candidate2), 2)  # نختار الأقرب (الأكثر أماناً)

    return {
        "ticker": t,
        "status": "APPROVED",
        "confidence": confidence,
        "entry": entry,
        "tp": tp,
        "sl": sl,
        "reason": "اجتاز الفلاتر الفنية",
        "lastClose": entry,
        "scores": scores  # يمكن إضافته للتصحيح لكن الواجهة لا تطلبه
    }

# ======================== Endpoints ========================
@app.get("/health")
async def health_check():
    """التحقق من حالة الخدمة."""
    return {"status": "ok", "timestamp": time.time()}

@app.get("/predict")
async def predict(ticker: str = Query(..., description="رمز السهم مثال: 2222.SR")):
    """تحليل سهم واحد وإرجاع النتيجة."""
    result = analyze_one(ticker)
    if result is None:
        raise HTTPException(status_code=404, detail="لا توجد بيانات كافية لهذا السهم")
    return result

@app.get("/top10")
async def top10():
    """مسح جميع الأسهم في الملف وإرجاع أفضل 10 فرص."""
    if not os.path.exists(TICKERS_PATH):
        raise HTTPException(status_code=500, detail=f"ملف {TICKERS_PATH} غير موجود")

    with open(TICKERS_PATH, "r", encoding="utf-8") as f:
        tickers = [line.strip() for line in f if line.strip()]

    results = []
    total_scanned = 0

    with ThreadPoolExecutor(max_workers=TOP10_WORKERS) as executor:
        future_to_ticker = {executor.submit(analyze_one, t): t for t in tickers}
        for future in as_completed(future_to_ticker):
            res = future.result()
            if res:
                total_scanned += 1
                if res["status"] == "APPROVED":
                    results.append(res)

    # ترتيب حسب الثقة تنازلياً
    results.sort(key=lambda x: x["confidence"], reverse=True)

    return {
        "items": results[:10],
        "total_scanned": total_scanned,
        "timestamp": time.time()
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
