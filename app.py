from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import requests
import pandas as pd
import numpy as np
import time
import joblib
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

app = FastAPI(title="Tadawul Sniper Pro", version="2.3.0-Technical-Stop")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "model.joblib"
TICKERS_PATH = "tickers_sa.txt"

# إعدادات الأهداف
TP_PCT = 0.05
# ملاحظة: تم استبدال SL_PCT الثابت بوقف فني متحرك داخل الدالة
TOP10_WORKERS = 10     

model = None
_prices_cache = {}
CACHE_TTL_SEC = 600

# =========================
# 1) جلب البيانات والمؤشرات
# =========================
def fetch_yahoo_prices(ticker: str, range_="1y", interval="1d"):
    key = (ticker, range_, interval)
    if key in _prices_cache and time.time() - _prices_cache[key]["ts"] < CACHE_TTL_SEC:
        return _prices_cache[key]["df"]

    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(url, params={"range": range_, "interval": interval}, headers=headers, timeout=15)
        js = r.json()
        quote = js['chart']['result'][0]['indicators']['quote'][0]
        # إضافة High و Low لحساب الوقف الفني
        df = pd.DataFrame({
            "close": quote["close"],
            "high": quote["high"],
            "low": quote["low"],
            "volume": quote["volume"]
        })
        df = df.dropna(subset=["close"]).reset_index(drop=True)
        _prices_cache[key] = {"ts": time.time(), "df": df}
        return df
    except: return None

def build_features(df: pd.DataFrame):
    d = df.copy()
    close = d["close"]
    d["ema20"] = close.ewm(span=20, adjust=False).mean()
    d["sma20"] = close.rolling(20).mean()
    
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    d["macd"] = ema12 - ema26
    d["macd_signal"] = d["macd"].ewm(span=9, adjust=False).mean()
    
    std20 = close.rolling(20).std()
    d["bb_mid"] = d["sma20"]
    
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    d["rsi14"] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
    
    d["vol_ma20"] = d["volume"].rolling(20).mean()
    return d.dropna().reset_index(drop=True)

# =========================
# 2) منطق القواعد المطور (Rules)
# =========================
def passes_rules(feat_df: pd.DataFrame):
    reasons = []
    curr = feat_df.iloc[-1]
    prev = feat_df.iloc[-2]
    
    # 1. شرط السيولة الانفجارية (أهم شرط لأسهم النسب)
    if not (curr['volume'] > (curr['vol_ma20'] * 1.2)):
        reasons.append("Rejected: Weak Volume (No real surge)")

    # 2. الماكد: استدارة من قاع
    if not (curr['macd'] > curr['macd_signal'] and curr['macd'] < 0.2):
        reasons.append("Rejected: MACD not turning from base")

    # 3. البولينجر: اختراق المنتصف
    cond_bb_break = (curr['close'] > curr['bb_mid']) and (prev['close'] <= prev['bb_mid'])
    was_below = (feat_df['close'].shift(1).tail(5) < feat_df['bb_mid'].shift(1).tail(5)).any()
    if not (cond_bb_break or (curr['close'] > curr['bb_mid'] and was_below)):
        reasons.append("Rejected: No Mid-BB breakout")

    # 4. RSI الأمان (تجنب القمم)
    if curr['rsi14'] > 65:
        reasons.append(f"Rejected: RSI too high ({round(curr['rsi14'],1)})")

    # 5. الأمان من المطاردة (الابتعاد عن المتوسط)
    dist = (curr['close'] - curr['ema20']) / curr['ema20']
    if dist > 0.03:
        reasons.append("Rejected: Price too far from EMA20")

    return (len(reasons) == 0), reasons

def calculate_confidence(feat_df: pd.DataFrame):
    curr = feat_df.iloc[-1]
    score = 0
    
    # --- [1] نقاط القوة (السيولة والزخم) ---
    if curr['volume'] > curr['vol_ma20'] * 1.5: score += 40
    elif curr['volume'] > curr['vol_ma20'] * 1.2: score += 25
    
    if (curr['macd'] > curr['macd_signal']): score += 30
    if (curr['close'] > curr['bb_mid']): score += 30
    
    # --- [2] نقاط الخصم (الأمان والتشبع) ---
    # إذا كان السهم متضخم (RSI > 65)، نخصم 40 نقطة
    if curr['rsi14'] > 65:
        score -= 40
        
    # إذا كان السعر بعيد جداً عن المتوسط (EMA20)، نخصم 50 نقطة
    dist = (curr['close'] - curr['ema20']) / curr['ema20']
    if dist > 0.03:
        score -= 50

    # نضمن أن النسبة لا تخرج عن نطاق 0-100
    return max(0, min(100, score))

# =========================
# 3) المحلل مع الوقف الفني
# =========================
def analyze_one(ticker: str):
    t = ticker.strip().upper()
    if not t.endswith(".SR"): t += ".SR"
    
    df = fetch_yahoo_prices(t)
    if df is None or len(df) < 30: return None
    
    feat_df = build_features(df)
    curr = feat_df.iloc[-1]
    last_close = float(curr["close"])
    
    # حساب الوقف الفني: أدنى سعر في آخر 3 شموع
    recent_low = float(feat_df['low'].tail(3).min())
    technical_stop = round(recent_low * 0.997, 2) # وقف تحت القاع بـ 0.3%
    
    # فلتر أمان للوقف (بين 1.5% و 4%)
    dist_pct = (last_close - technical_stop) / last_close
    if dist_pct > 0.04: technical_stop = round(last_close * 0.96, 2)
    if dist_pct < 0.015: technical_stop = round(last_close * 0.985, 2)
    
    ok, reasons = passes_rules(feat_df)
    conf_pct = calculate_confidence(feat_df)
    
    recommendation = "BUY" if ok else "NO_TRADE"
    
    return {
        "ticker": t,
        "recommendation": recommendation,
        "confidence_pct": conf_pct,
        "entry": round(last_close, 2),
        "take_profit": round(last_close * (1 + TP_PCT), 2),
        "stop_loss": technical_stop,
        "reason": " | ".join(reasons) if reasons else "اختراق قاع بسيولة انفجارية (نموذج النسب)",
        "last_close": round(last_close, 2),
        "status": "APPROVED" if ok else "REJECTED"
    }

# =========================
# 4) الروابط (Endpoints)
# =========================
@app.on_event("startup")
def startup():
    global model
    if os.path.exists(MODEL_PATH): model = joblib.load(MODEL_PATH)

@app.get("/predict")
def predict(ticker: str):
    return analyze_one(ticker)

@app.get("/top10")
def top10():
    # ... (كود جلب التيكرز والتحليل يبقى كما هو) ...

    # التعديل هنا: 
    # x['status'] == 'APPROVED' يضع الـ BUY في الأعلى
    # x['confidence_pct'] يضمن ترتيبهم من 100 إلى 0
    sorted_res = sorted(
        results, 
        key=lambda x: (x['status'] == 'APPROVED', x['confidence_pct']), 
        reverse=True
    )
    
    return {"items": sorted_res[:10], "total_scanned": len(results)}
