from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import requests
import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

TICKERS_PATH = "tickers_sa.txt"

def analyze_one(ticker: str):
    t_clean = ticker.strip().upper()
    if not t_clean.endswith(".SR"): t_clean += ".SR"
    
    headers = {"User-Agent": "Mozilla/5.0"}
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{t_clean}?range=1y&interval=1d"
    
    try:
        r = requests.get(url, headers=headers, timeout=10)
        js = r.json()
        res = js['chart']['result'][0]
        q = res['indicators']['quote'][0]
        
        df = pd.DataFrame(q)
        df = df.dropna().reset_index(drop=True)
        
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        # --- الحسابات الفنية المبسطة ---
        close_p = float(last['close'])
        open_p = float(last['open']) if last['open'] > 0 else close_p
        
        # RSI & MACD (بشكل سريع)
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi = 100 - (100 / (1 + (gain / (loss + 1e-9)))).iloc[-1]
        
        # حساب الأهداف (Target) والوقف (Stop) بناءً على حركة السعر
        target = close_p * 1.05  # هدف 5%
        stop = close_p * 0.97    # وقف 3%

        # نظام النقاط (لحل مشكلة الـ 100%)
        score = 0
        if close_p > prev['close']: score += 40
        if rsi < 65: score += 30
        if last['volume'] > df['volume'].tail(10).mean(): score += 30

        recommendation = "BUY" if (score >= 70 and close_p > prev['close']) else "NO_TRADE"

        # هذا التنسيق هو ما تبحث عنه واجهتك (Frontend)
        return {
            "ticker": t_clean.replace(".SR", ""),
            "recommendation": recommendation,
            "confidence_pct": int(score),
            "entry": round(close_p, 2),        # سيظهر في خانة "الدخول"
            "target": round(target, 2),       # سيظهر في خانة "الهدف"
            "stop": round(stop, 2),           # سيظهر في خانة "الوقف"
            "last_close": round(close_p, 2),  # سيظهر في خانة "آخر إغلاق"
            "reason": "تجميع إيجابي وزخم صاعد"
        }
    except:
        return {"ticker": ticker, "recommendation": "ERROR", "confidence_pct": 0}

@app.get("/predict")
def predict(ticker: str):
    return analyze_one(ticker)

@app.get("/top10")
def top10():
    if not os.path.exists(TICKERS_PATH): return {"items": []}
    with open(TICKERS_PATH, "r") as f:
        tickers = [line.strip() for line in f if line.strip()][:50] # فحص أول 50 سهم للسرعة
    
    results = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(analyze_one, t) for t in tickers]
        for future in as_completed(futures):
            res = future.result()
            if res["recommendation"] == "BUY":
                results.append(res)
    
    return {"items": sorted(results, key=lambda x: x['confidence_pct'], reverse=True)[:10]}
