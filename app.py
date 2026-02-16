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
    prev = feat_df.iloc[-2]
    
    # 1. شرط السعر المرن (الجديد)
    # نسمح بالدخول إذا كان السعر صاعداً، أو إذا كان مستقراً مع زخم فني
    is_price_ok = last["close"] >= prev["close"] * 0.995 # يسمح بهبوط طفيف جداً 0.5% (تذبذب طبيعي)
    
    # 2. فحص الزخم (MACD Histogram)
    # إذا كان الهيستوجرام ينمو للأعلى، فهذا يعني أن البائعين فقدوا السيطرة
    is_momentum_rising = last["macd_hist"] > prev["macd_hist"]

    if not is_price_ok and not is_momentum_rising:
        reasons.append("السعر في هبوط حاد والزخم ضعيف")

    if last["rsi14"] > 70:
        reasons.append("تشبع شرائي كبير - خطر")
        
    if last["rsi14"] < 30:
        reasons.append("السهم في منطقة انهيار - انتظر إشارة ارتداد")

    return (len(reasons) == 0), reasons

def analyze_one(ticker: str):
    # ... (كود جلب البيانات كما هو) ...
    df = fetch_yahoo_prices(t)
    if df is None or len(df) < 30: return {"ticker": t, "status": "بيانات ناقصة"}

    feat_df = build_features(df)
    last = feat_df.iloc[-1]
    prev = feat_df.iloc[-2]

    # حساب النقاط بناءً على الاقتراح الجديد:
    score = 0
    
    # أ) نقاط السعر (30 نقطة)
    if last["close"] > prev["close"]: score += 30
    elif last["close"] >= prev["close"] * 0.998: score += 15 # نقاط أقل للاستقرار
    
    # ب) نقاط MACD (توقع الانفجار) (35 نقطة)
    if last["macd_hist"] > prev["macd_hist"]: score += 35
    
    # ج) نقاط المتوسطات (20 نقطة)
    if last["close"] > last["ema20"]: score += 20
    
    # د) نقاط RSI (15 نقطة)
    if 45 <= last["rsi14"] <= 65: score += 15

    ok, reasons = passes_rules(df, feat_df)
    
    # قرار الدخول: إذا حصل على أكثر من 65 نقطة واجتاز الفلاتر
    recommendation = "BUY" if (ok and score >= 65) else "NO_TRADE"
    
    return {
        "ticker": t,
        "recommendation": recommendation,
        "confidence_pct": score,
        "reason": " | ".join(reasons) if reasons else "تجميع إيجابي وزخم صاعد",
        "last_close": round(float(last["close"]), 2)
    }

# إبقاء Endpoints الأخرى كما هي...
@app.get("/predict")
def predict(ticker: str):
    return analyze_one(ticker)
