from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import requests
import pandas as pd
import numpy as np
import time
import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict, Any
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
MIN_VOLUME = int(os.getenv("MIN_VOLUME", "500000"))
MIN_PRICE = float(os.getenv("MIN_PRICE", "10.0"))
ATR_EXCLUDE_PCT = float(os.getenv("ATR_EXCLUDE_PCT", "3.0"))
MAX_5DAY_GAIN = float(os.getenv("MAX_5DAY_GAIN", "0.10"))

# ======================== كاش بسيط يدوي ========================
_prices_cache = {}

# ======================== إنشاء تطبيق FastAPI ========================
app = FastAPI(
    title="Tadawul Sniper Pro",
    description="محلل فني لأسهم السوق السعودي",
    version="4.0.1"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================== التأكد من وجود ملف التيكرات (اختياري) ========================
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

    d["adx"] = 25  # قيمة افتراضية لتجنب الأخطاء

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
# ======================== نقاط نهاية التشخيص (للفحص فقط) ========================

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
    
    # آخر 5 أيام من البيانات الخام (قبل حساب المؤشرات)
    raw_last5 = df.tail(5).to_dict(orient="records")
    
    # آخر يوم من المؤشرات المحسوبة (تحويل القيم numpy إلى float/json)
    curr = feat_df.iloc[-1].to_dict()
    # تحويل أي قيم numpy إلى أنواع python قياسية
    for k, v in curr.items():
        if isinstance(v, (np.integer, np.floating)):
            curr[k] = float(v)
        elif isinstance(v, np.bool_):
            curr[k] = bool(v)
    
    # فحص الشروط الأساسية
    passed, reasons = passes_core_rules(feat_df)
    
    # فحص الاستبعاد
    excluded, exclude_reason = should_exclude(feat_df)
    
    # نقاط المجموعات
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


# دالة مساعدة لجمع إحصائيات debug_summary (دون تخزين النتائج كاملة)
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
    
    # الاستبعاد أولاً
    excluded, exclude_reason = should_exclude(feat_df)
    if excluded:
        return {"status": "EXCLUDED", "reason": exclude_reason}
    
    # الشروط الأساسية
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
                continue  # تجاهل الأسهم التي فشل جلب بياناتها
            stats["total"] += 1
            if res["status"] == "EXCLUDED":
                stats["excluded_count"] += 1
                reason = res["reason"]
                stats["excluded_reasons"][reason] = stats["excluded_reasons"].get(reason, 0) + 1
            elif res["status"] == "REJECTED":
                stats["core_failed_count"] += 1
                # قد يكون هناك عدة أسباب مفصولة بـ " | "
                reasons_list = res["reason"].split(" | ")
                for r in reasons_list:
                    stats["core_failed_reasons"][r] = stats["core_failed_reasons"].get(r, 0) + 1
            elif res["status"] == "APPROVED":
                stats["approved_count"] += 1
    
    return stats
