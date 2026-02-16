from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import requests
import pandas as pd
import numpy as np
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

app = FastAPI(title="Tadawul MVP API", version="2.2.4")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

TICKERS_PATH = "tickers_sa.txt"
TOP10_WORKERS = 5

# =========================
# 1) جلب الأسعار (المصححة)
# =========================
def fetch_yahoo_prices(ticker: str, range_: str = "1y", interval: str = "1d"):
    t = ticker.strip().upper()
    if not t.endswith(".SR"): t += ".SR"
    
    headers = {"User-Agent": "Mozilla/5.0"}
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{t}"
    try:
        r = requests.get(url, params={"range": range_, "interval": interval}, headers=headers, timeout=15)
        if r.status_code != 200: return None
        js = r.json()
        res = js['chart']['result'][0]
        q = res['indicators']['quote'][0]
        
        # التأكد من وجود بيانات Close قبل بناء DataFrame
        if not q.get('close'): return None
        
        df = pd.DataFrame({
            "open": q.get('open'),
            "high": q.get('high'),
            "low": q.get('low'),
            "close": q.get('close'),
            "volume": q.get('volume')
        })
        return df.dropna().reset_index(drop=True)
    except:
        return None

# =========================
# 2) المؤشرات الفنية
# =========================
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    close = d["close"]
    # المتوسطات
    d["ema20"] = close.ewm(span=20, adjust=False).mean()
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
    return d.dropna().reset_index(drop=True)

# =========================
# 3) قواعد الفلترة (المنطق المرن الجديد)
# =========================
def passes_rules(feat_df: pd.DataFrame):
    reasons = []
    if len(feat_df) < 2: return False, ["بيانات غير كافية"]
    
    last = feat_df.iloc[-1]
    prev = feat_df.iloc[-2]
    
    # السماح بهبوط طفيف (تذبذب) بشرط تحسن الزخم
    is_price_ok = last["close"] >= prev["close"] * 0.995
    is_momentum_rising = last["macd_hist"] > prev["macd_hist"]

    if not is_price_ok and not is_momentum_rising:
        reasons.append("السعر في هبوط والزخم ضعيف")

    if last["rsi14"] > 70:
        reasons.append("تشبع شرائي - مخاطرة عالية")
        
    if last["rsi14"] < 30:
        reasons.append("منطقة انهيار - انتظر الارتداد")

    return (len(reasons) == 0), reasons

# =========================
# 4) دالة التحليل (المصححة بالكامل)
# =========================
def analyze_one(ticker: str):
    # تصحيح الخطأ: تعريف t داخل الدالة
    t_clean = ticker.strip().upper()
    if not t_clean.endswith(".SR"): t_clean += ".SR"
    
    df = fetch_yahoo_prices(t_clean)
    if df is None or len(df) < 30:
        return {"ticker": t_clean, "recommendation": "NO_TRADE", "confidence_pct": 0, "reason": "بيانات تاريخية غير كافية"}

    feat_df = build_features(df)
    last = feat_df.iloc[-1]
    prev = feat_df.iloc[-2]

    # حساب النقاط
    score = 0
    if last["close"] > prev["close"]: score += 30
    elif last["close"] >= prev["close"] * 0.998: score += 15
    
    if last["macd_hist"] > prev["macd_hist"]: score += 35
    if last["close"] > last["ema20"]: score += 20
    if 45 <= last["rsi14"] <= 65: score += 15

    ok, reasons = passes_rules(feat_df)
    
    # قرار الدخول
    recommendation = "BUY" if (ok and score >= 65) else "NO_TRADE"
    
    return {
        "ticker": t_clean,
        "recommendation": recommendation,
        "confidence_pct": int(score),
        "reason": " | ".join(reasons) if reasons else "تجميع إيجابي وزخم صاعد",
        "last_close": round(float(last["close"]), 2)
    }

# =========================
# 5) Endpoints
# =========================
@app.get("/predict")
def predict(ticker: str):
    return analyze_one(ticker)

@app.get("/top10")
def top10():
    if not os.path.exists(TICKERS_PATH):
        return {"error": "ملف الأسهم غير موجود"}
    
    with open(TICKERS_PATH, "r") as f:
        tickers = [line.strip() for line in f if line.strip()]
    
    results = []
    with ThreadPoolExecutor(max_workers=TOP10_WORKERS) as executor:
        futures = [executor.submit(analyze_one, t) for t in tickers]
        for future in as_completed(futures):
            results.append(future.result())
    
    # الترتيب: شراء أولاً ثم الأعلى ثقة
    sorted_results = sorted(results, key=lambda x: (x['recommendation'] == 'BUY', x['confidence_pct']), reverse=True)
    return {"items": sorted_results[:10]}
