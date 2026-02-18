"""
backtest.py - أداة الاختبار الخلفي لتحليل أداء شروط التداول
يقرأ نفس الشروط المستخدمة في Tadawul Sniper ويطبقها على بيانات تاريخية
"""

import pandas as pd
import numpy as np
import requests
import time
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# إعداد التسجيل
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# ======================== الإعدادات الثابتة ========================
# نفس الإعدادات المستخدمة في التطبيق الرئيسي
TICKERS_PATH = "tickers_sa.txt"
TP_PCT = 0.05  # هدف 5%
MIN_VOLUME = 250000
MIN_PRICE = 5.0
ATR_EXCLUDE_PCT = 4.0
MAX_5DAY_GAIN = 0.15
MAX_HOLD_DAYS = 10  # أقصى مدة للاحتفاظ بالسهم

# كاش بسيط للبيانات
_prices_cache = {}

# ======================== جلب البيانات التاريخية ========================
def fetch_historical_prices(ticker: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """
    جلب بيانات تاريخية لفترة محددة.
    start_date, end_date: بصيغة YYYY-MM-DD
    """
    # نحول الفترة إلى صيغة range التي تفهمها ياهو (1d, 5d, 1mo, 3mo, 1y, 2y, 5y, 10y)
    # نبسطها: نجيب آخر سنتين ونفلتر بعدين
    key = (ticker, "2y", "1d")
    if key in _prices_cache and time.time() - _prices_cache[key]["ts"] < 3600:  # كاش ساعة
        df = _prices_cache[key]["df"]
    else:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        headers = {"User-Agent": "Mozilla/5.0"}
        params = {"range": "2y", "interval": "1d"}
        try:
            r = requests.get(url, params=params, headers=headers, timeout=15)
            r.raise_for_status()
            js = r.json()
            result = js['chart']['result'][0]
            quote = result['indicators']['quote'][0]
            # جلب التاريخ
            timestamp = result['timestamp']
            dates = pd.to_datetime(timestamp, unit='s')
            df = pd.DataFrame({
                "date": dates,
                "open": quote["open"],
                "high": quote["high"],
                "low": quote["low"],
                "close": quote["close"],
                "volume": quote["volume"]
            })
            df = df.dropna(subset=["close"]).reset_index(drop=True)
            _prices_cache[key] = {"ts": time.time(), "df": df}
        except Exception as e:
            logger.error(f"خطأ في جلب {ticker}: {e}")
            return None

    # فلترة حسب التاريخ
    df = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()
    if len(df) < 30:
        return None
    return df

# ======================== حساب المؤشرات (نفس build_features) ========================
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """حساب المؤشرات الفنية على DataFrame"""
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

    d["vol_ma20"] = volume.rolling(20).mean()
    d["vol_std"] = volume.rolling(20).std()

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

    # ATR
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    d["atr14"] = tr.rolling(14).mean()
    d["atr_pct"] = d["atr14"] / close * 100

    # Stochastic
    low14 = low.rolling(14).min()
    high14 = high.rolling(14).max()
    d["stoch_k"] = 100 * ((close - low14) / (high14 - low14 + 1e-9))
    d["stoch_d"] = d["stoch_k"].rolling(3).mean()

    # شموع
    d["candle_green"] = close > d["open"]
    d["body"] = abs(close - d["open"])
    d["upper_shadow"] = high - d[["close", "open"]].max(axis=1)
    d["lower_shadow"] = d[["close", "open"]].min(axis=1) - low

    d = d.dropna().reset_index(drop=True)
    return d

# ======================== الأنماط السعرية (مبسطة للاختبار) ========================
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

def has_bearish_pattern(feat_df, idx):
    """التحقق من وجود نمط هابط في آخر 3 أيام من النقطة idx"""
    if idx < 2:
        return False
    curr = feat_df.iloc[idx]
    prev = feat_df.iloc[idx-1]
    prev2 = feat_df.iloc[idx-2]

    if is_bearish_engulfing(prev, curr):
        return True
    if is_shooting_star(curr):
        return True
    return False

# ======================== شروط الاستبعاد ========================
def should_exclude_at_row(row, lookback_df):
    """تطبيق شروط الاستبعاد على صف معين"""
    if row["atr_pct"] > ATR_EXCLUDE_PCT:
        return True, f"تقلب عالي ({row['atr_pct']:.1f}%)"
    if row["volume"] < MIN_VOLUME:
        return True, f"سيولة منخفضة ({row['volume']:,.0f})"
    if row["close"] < MIN_PRICE:
        return True, f"سعر منخفض ({row['close']:.2f})"
    # ارتفاع 5 أيام
    if len(lookback_df) >= 6:
        idx = lookback_df.index[-1]
        if idx >= 5:
            close_5 = lookback_df.iloc[idx-5]["close"]
            gain_5 = (row["close"] / close_5 - 1)
            if gain_5 > MAX_5DAY_GAIN:
                return True, f"ارتفاع 5 أيام ({gain_5*100:.1f}%)"
    # أنماط هابطة
    if has_bearish_pattern(lookback_df, len(lookback_df)-1):
        return True, "نمط هابط"
    return False, ""

# ======================== الشروط الأساسية ========================
def passes_core_rules_at_row(row):
    """تطبيق الشروط الأساسية على صف معين"""
    reasons = []
    if not (row["close"] > row["ema20"]):
        reasons.append("تحت EMA20")
    if not (row["close"] > row["sma50"]):
        reasons.append("تحت SMA50")
    if not (row["macd"] > row["macd_signal"]):
        reasons.append("MACD أقل")
    if not (row["volume"] > 1.2 * row["vol_ma20"]):
        reasons.append("حجم < 1.2x")
    if not (30 < row["rsi14"] < 75):
        reasons.append(f"RSI خارج ({row['rsi14']:.1f})")
    dist = (row["close"] - row["ema20"]) / row["ema20"]
    if dist > 0.07:
        reasons.append("بعيد عن المتوسط")
    return len(reasons) == 0, reasons

# ======================== محاكاة صفقة واحدة ========================
def simulate_trade(ticker: str, entry_date: str, entry_price: float, stop_loss: float, 
                   historical_df: pd.DataFrame) -> Dict[str, Any]:
    """
    محاكاة صفقة من تاريخ الدخول إلى أقصى مدة محددة.
    تعيد نتيجة الصفقة (نجاح/فشل/محايد) ومدة تحقيق الهدف.
    """
    # البحث عن تاريخ الدخول في البيانات
    entry_idx = historical_df[historical_df['date'] == entry_date].index
    if len(entry_idx) == 0:
        return {"result": "error", "reason": "تاريخ غير موجود"}
    start_idx = entry_idx[0] + 1
    end_idx = min(start_idx + MAX_HOLD_DAYS, len(historical_df))

    for i in range(start_idx, end_idx):
        row = historical_df.iloc[i]
        high = row["high"]
        low = row["low"]
        
        # هل حقق الهدف؟
        if high >= entry_price * (1 + TP_PCT):
            days = i - start_idx + 1
            return {"result": "success", "days": days, "exit_price": entry_price * (1 + TP_PCT)}
        
        # هل ضرب وقف الخسارة؟
        if low <= stop_loss:
            days = i - start_idx + 1
            return {"result": "fail", "days": days, "exit_price": low}

    # لم يتحقق شيء خلال المدة
    return {"result": "neutral", "days": MAX_HOLD_DAYS, "exit_price": historical_df.iloc[end_idx-1]["close"]}

# ======================== تحليل سهم واحد (تاريخي) ========================
def backtest_one_ticker(ticker: str, start_date: str, end_date: str) -> List[Dict[str, Any]]:
    """تحليل سهم واحد عبر الفترة التاريخية وتسجيل جميع إشارات الدخول"""
    t = ticker.strip().upper()
    if not t.endswith(".SR"):
        t += ".SR"

    df = fetch_historical_prices(t, start_date, end_date)
    if df is None or len(df) < 50:
        return []

    feat_df = calculate_indicators(df)
    trades = []

    # نبدأ من اليوم 50 لضمان وجود جميع المؤشرات
    for i in range(50, len(feat_df)):
        # نأخذ آخر 4 أيام لفحص الأنماط
        window = feat_df.iloc[:i+1]
        curr = feat_df.iloc[i]
        
        # 1. التحقق من الاستبعاد
        excluded, exclude_reason = should_exclude_at_row(curr, window)
        if excluded:
            continue

        # 2. الشروط الأساسية
        passed, reasons = passes_core_rules_at_row(curr)
        if not passed:
            continue

        # 3. حساب وقف الخسارة (أدنى سعر في آخر 3 أيام * 0.99)
        recent_low = feat_df["low"].iloc[i-2:i+1].min()
        sl_candidate1 = recent_low * 0.99
        sl_candidate2 = curr["close"] - 2 * curr["atr14"]
        stop_loss = min(sl_candidate1, sl_candidate2)

        # 4. تسجيل الصفقة
        trades.append({
            "ticker": t,
            "entry_date": curr["date"],
            "entry_price": round(curr["close"], 2),
            "stop_loss": round(stop_loss, 2),
            "target": round(curr["close"] * (1 + TP_PCT), 2)
        })

    return trades

# ======================== تشغيل الاختبار على جميع الأسهم ========================
def run_backtest(start_date: str, end_date: str, max_workers: int = 5) -> pd.DataFrame:
    """
    تشغيل الاختبار الخلفي على جميع الأسهم في الفترة المحددة
    """
    if not os.path.exists(TICKERS_PATH):
        logger.error(f"ملف {TICKERS_PATH} غير موجود")
        return pd.DataFrame()

    with open(TICKERS_PATH, "r", encoding="utf-8") as f:
        tickers = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    all_trades = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(backtest_one_ticker, t, start_date, end_date): t for t in tickers[:20]}  # نبدأ بـ 20 سهماً للتجربة
        for future in as_completed(futures):
            trades = future.result()
            all_trades.extend(trades)
            logger.info(f"تم تحليل {futures[future]}: {len(trades)} صفقة")

    return pd.DataFrame(all_trades)

# ======================== محاكاة نتائج الصفقات ========================
def simulate_all_trades(trades_df: pd.DataFrame, historical_data_func) -> Dict[str, Any]:
    """
    محاكاة جميع الصفقات المسجلة وجمع الإحصائيات
    """
    results = {
        "total_trades": 0,
        "success": 0,
        "fail": 0,
        "neutral": 0,
        "avg_days_success": 0,
        "avg_days_fail": 0,
        "tickers": {}
    }

    for _, trade in trades_df.iterrows():
        ticker = trade["ticker"]
        entry_date = trade["entry_date"]
        entry_price = trade["entry_price"]
        stop_loss = trade["stop_loss"]

        # نجلب البيانات كاملة لهذا السهم
        df = fetch_historical_prices(ticker, entry_date, 
                                     (pd.to_datetime(entry_date) + timedelta(days=MAX_HOLD_DAYS+5)).strftime("%Y-%m-%d"))
        if df is None:
            continue

        feat_df = calculate_indicators(df)
        sim = simulate_trade(ticker, entry_date, entry_price, stop_loss, feat_df)

        results["total_trades"] += 1
        results[sim["result"]] += 1

        if sim["result"] == "success":
            results["avg_days_success"] = (results["avg_days_success"] * (results["success"]-1) + sim["days"]) / results["success"]
        elif sim["result"] == "fail":
            results["avg_days_fail"] = (results["avg_days_fail"] * (results["fail"]-1) + sim["days"]) / results["fail"]

        # تخزين تفاصيل لكل سهم
        if ticker not in results["tickers"]:
            results["tickers"][ticker] = {"success": 0, "fail": 0, "neutral": 0}
        results["tickers"][ticker][sim["result"]] += 1

    return results

# ======================== تقرير نهائي ========================
def print_report(results: Dict[str, Any]):
    """طباعة تقرير جميل عن نتائج الاختبار"""
    print("\n" + "="*60)
    print("📊 تقرير الاختبار الخلفي".center(60))
    print("="*60)

    total = results["total_trades"]
    if total == 0:
        print("\n❌ لا توجد صفقات في هذه الفترة")
        return

    success_rate = (results["success"] / total) * 100
    fail_rate = (results["fail"] / total) * 100
    neutral_rate = (results["neutral"] / total) * 100

    print(f"\n📈 إجمالي الصفقات: {total}")
    print(f"✅ نجاح: {results['success']} ({success_rate:.1f}%)")
    print(f"❌ فشل: {results['fail']} ({fail_rate:.1f}%)")
    print(f"⏸️ محايد: {results['neutral']} ({neutral_rate:.1f}%)")

    if results["success"] > 0:
        print(f"\n⏱️ متوسط أيام النجاح: {results['avg_days_success']:.2f} يوم")
    if results["fail"] > 0:
        print(f"⏱️ متوسط أيام الفشل: {results['avg_days_fail']:.2f} يوم")

    print("\n" + "-"*60)
    print("🏆 أفضل 5 أسهم (حسب عدد الصفقات الناجحة)")
    top_tickers = sorted(results["tickers"].items(), 
                         key=lambda x: x[1]["success"], reverse=True)[:5]
    for ticker, stats in top_tickers:
        if stats["success"] > 0:
            print(f"{ticker}: نجاح {stats['success']} / فشل {stats['fail']}")

    print("\n" + "="*60)

# ======================== التشغيل الرئيسي ========================
if __name__ == "__main__":
    # تحديد الفترة
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")  # آخر 6 أشهر

    print(f"🔍 بدء الاختبار الخلفي من {start} إلى {end}")
    print("📥 جلب البيانات وتحليل الإشارات...")

    # 1. جمع جميع الإشارات
    trades_df = run_backtest(start, end, max_workers=5)

    if len(trades_df) == 0:
        print("❌ لم يتم العثور على أي إشارات في هذه الفترة")
        exit()

    print(f"✅ تم العثور على {len(trades_df)} إشارة محتملة")
    print("📊 جاري محاكاة الصفقات...")

    # 2. محاكاة الصففات
    results = simulate_all_trades(trades_df, fetch_historical_prices)

    # 3. طباعة التقرير
    print_report(results)

    # 4. حفظ النتائج (اختياري)
    trades_df.to_csv("backtest_signals.csv", index=False)
    print("\n💾 تم حفظ الإشارات في backtest_signals.csv")
