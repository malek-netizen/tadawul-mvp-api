from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import requests
import pandas as pd
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

app = FastAPI(title="Tadawul MVP AI", version="2.2.5")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

TICKERS_PATH = "tickers_sa.txt"

# =========================
# 1) جلب الأسعار مع كامل البيانات
# =========================
def fetch_yahoo_prices(ticker: str):
    t = ticker.strip().upper()
    if not t.endswith(".SR"): t += ".SR"
    headers = {"User-Agent": "Mozilla/5.0"}
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{t}?range=1y&interval=1d"
    try:
        r = requests.get(url, headers=headers, timeout=15)
        js = r.json()
        res = js['chart']['result'][0]
        q = res['indicators']['quote'][0]
        df = pd.DataFrame({
            "open": q['open'], "high": q['high'], 
            "low": q['low'], "close": q['close'], 
            "volume": q['volume']
        }).dropna().reset_index(drop=True)
        return df
    except: return None

# =========================
# 2) محرك التحليل المطور
# =========================
def analyze_one(ticker: str):
    t_clean = ticker.strip().upper()
    if not t_clean.endswith(".SR"): t_clean += ".SR"
    
    df = fetch_yahoo_prices(t_clean)
    if df is None or len(df) < 35:
        return {"ticker": t_clean, "recommendation": "NO_TRADE", "confidence_pct": 0, "reason": "بيانات ناقصة"}

    # حساب المؤشرات
    close = df["close"]
    ema20 = close.ewm(span=20, adjust=False).mean()
    
    # RSI
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rsi = 100 - (100 / (1 + (gain / (loss + 1e-9))))
    
    # MACD
    exp1 = close.ewm(span=12, adjust=False).mean()
    exp2 = close.ewm(span=26, adjust=False).mean()
    macd_line = exp1 - exp2
    signal = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - signal

    # --- نظام النقاط الجديد (أكثر صرامة) ---
    score = 0
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    # 1. حركة السعر (Price Action) - 25 نقطة
    if last["close"] > prev["close"]: score += 15
    if last["close"] > last["open"]: score += 10 # شمعة خضراء
    
    # 2. الزخم (Momentum) - 35 نقطة
    if macd_hist.iloc[-1] > macd_hist.iloc[-2]:
        score += 20
        if macd_hist.iloc[-1] > 0: score += 15 # زخم إيجابي حقيقي
        
    # 3. القوة النسبية (RSI) - 20 نقطة
    last_rsi = rsi.iloc[-1]
    if 45 <= last_rsi <= 65: score += 20
    elif 35 <= last_rsi < 45: score += 10

    # 4. السيولة (Volume) - 20 نقطة
    avg_vol = df["volume"].tail(10).mean()
    if last["volume"] > avg_vol: score += 20

    # الفلترة (Rules)
    reasons = []
    if last_rsi > 70: reasons.append("تشبع شرائي")
    if last["close"] < prev["close"] * 0.99: reasons.append("نزيف سعري")
    
    recommendation = "BUY" if (score >= 65 and not reasons) else "NO_TRADE"
    
    return {
        "ticker": t_clean,
        "recommendation": recommendation,
        "confidence_pct": int(score),
        "price_data": {
            "open": round(float(last["open"]), 2),
            "high": round(float(last["high"]), 2),
            "low": round(float(last["low"]), 2),
            "close": round(float(last["close"]), 2),
            "vol_status": "عالي" if last["volume"] > avg_vol else "طبيعي"
        },
        "technical_values": {
            "rsi": round(float(last_rsi), 2),
            "macd_hist": round(float(macd_hist.iloc[-1]), 4)
        },
        "reason": " | ".join(reasons) if reasons else "تجميع إيجابي وزخم صاعد"
    }

# =========================
# 3) الروابط
# =========================
@app.get("/predict")
def predict(ticker: str):
    return analyze_one(ticker)

@app.get("/top10")
def top10():
    if not os.path.exists(TICKERS_PATH): return {"error": "File not found"}
    with open(TICKERS_PATH, "r") as f:
        tickers = [line.strip() for line in f if line.strip()]
    
    results = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(analyze_one, t) for t in tickers]
        for future in as_completed(futures):
            results.append(future.result())
    
    # ترتيب احترافي: BUY أولاً، ثم الأعلى ثقة
    sorted_results = sorted(results, key=lambda x: (x['recommendation'] == 'BUY', x['confidence_pct']), reverse=True)
    return {"items": sorted_results[:10]}
