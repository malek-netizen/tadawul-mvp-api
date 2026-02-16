from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import requests
import pandas as pd
import numpy as np
import time
import joblib
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

app = FastAPI(title="Tadawul MVP API", version="2.2.3")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- إعدادات ثابتة ---
MODEL_PATH = "model.joblib"
TICKERS_PATH = "tickers_sa.txt"
TP_PCT = 0.05
SL_PCT = 0.02
ML_THRESHOLD = 0.60
RSI_MIN, RSI_MAX, RSI_OVERBOUGHT = 40, 65, 70
MAX_ABOVE_EMA20 = 0.06
CONSOL_DAYS, CONSOL_RANGE_MAX = 4, 0.06
LOOKBACK_LOW_DAYS = 10
MIN_TREND_SCORE = 2

model = None
@app.on_event("startup")
def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        try: model = joblib.load(MODEL_PATH)
        except: model = None

# =========================
# 1) جلب الأسعار (Yahoo Chart)
# =========================
def fetch_yahoo_prices(ticker: str, range_: str = "1y", interval: str = "1d"):
    headers = {"User-Agent": "Mozilla/5.0"}
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    try:
        r = requests.get(url, params={"range": range_, "interval": interval}, headers=headers, timeout=15)
        if r.status_code != 200: return None
        js = r.json()
        res = js['chart']['result'][0]
        q = res['indicators']['quote'][0]
        df = pd.DataFrame({"open": q['open'], "high": q['high'], "low": q['low'], "close": q['close'], "volume": q['volume']})
        return df.dropna().reset_index(drop=True)
    except: return None

# =========================
# 2) المؤشرات الفنية (الأساسيات)
# =========================
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    close = d["close"]
    # المتوسطات
    d["ema20"] = close.ewm(span=20, adjust=False).mean()
    d["ema50"] = close.ewm(span=50, adjust=False).mean()
    # RSI
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    d["rsi14"] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
    # MACD
    exp1 = close.ewm(span=12, adjust=False).mean()
    exp2 = close.ewm(span=26, adjust=False).mean()
    d["macd"] = exp1 - exp2
    d["macd_signal"] = d["macd"].ewm(span=9, adjust=False).mean()
    d["macd_hist"] = d["macd"] - d["macd_signal"]
    # الارتجاع
    d["ret1"] = close.pct_change(1)
    return d.dropna().reset_index(drop=True)

# =========================
# 3) شروطك الخاصة (تأكيد الصعود)
# =========================
def is_reversing_up(feat_df: pd.DataFrame) -> bool:
    """ الشرط المطلوب: السعر الحالي أكبر من السابق وأكبر من قبل يومين """
    if len(feat_df) < 3: return False
    c = feat_df["close"].iloc[-1]
    p1 = feat_df["close"].iloc[-2]
    p2 = feat_df["close"].iloc[-3]
    return (c > p1) and (c > p2)

def is_consolidating(feat_df: pd.DataFrame) -> bool:
    if len(feat_df) < 5: return False
    last = feat_df.tail(4)
    rng = (last["high"].max() - last["low"].min()) / last["close"].mean()
    return rng <= CONSOL_RANGE_MAX

# =========================
# 4) الفلترة النهائية (The Rules)
# =========================
def passes_rules(df_ohlc: pd.DataFrame, feat_df: pd.DataFrame):
    reasons = []
    last = feat_df.iloc[-1]
    
    # الشرط الأساسي الذي طلبته: منع الدخول في مسار هابط
    if not is_reversing_up(feat_df):
        reasons.append("السعر في مسار هابط أو لم يبدأ بالارتداد بعد")

    if last["rsi14"] > RSI_OVERBOUGHT:
        reasons.append("تشبع شرائي RSI > 70")
        
    if last["close"] < last["ema20"] * 0.98: # تحت المتوسط بمسافة كبيرة
        reasons.append("السعر تحت متوسط 20 (ترند ضعيف)")

    return (len(reasons) == 0), reasons

# =========================
# 5) دالة التحليل الرئيسية (المصححة)
# =========================
def analyze_one(ticker: str):
    t = ticker.strip().upper()
    if not t.endswith(".SR"): t += ".SR"
    
    df = fetch_yahoo_prices(t)
    if df is None or len(df) < 30:
        return {"ticker": t, "recommendation": "NO_TRADE", "reason": "بيانات غير كافية"}

    feat_df = build_features(df)
    last_close = float(feat_df["close"].iloc[-1])
    
    # تشغيل الفلاتر
    ok, reasons = passes_rules(df, feat_df)
    
    # حساب النقاط
    score = 0
    if is_reversing_up(feat_df): score += 40
    if last_close > feat_df["ema20"].iloc[-1]: score += 30
    if 40 < feat_df["rsi14"].iloc[-1] < 65: score += 30

    recommendation = "BUY" if (ok and score >= 70) else "NO_TRADE"
    
    return {
        "ticker": t,
        "recommendation": recommendation,
        "confidence_pct": score,
        "entry": round(last_close, 2),
        "reason": " | ".join(reasons) if reasons else "إشارة دخول مؤكدة",
        "last_close": round(last_close, 2)
    }

# إبقاء Endpoints الأخرى كما هي...
@app.get("/predict")
def predict(ticker: str):
    return analyze_one(ticker)
