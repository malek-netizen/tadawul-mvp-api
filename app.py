from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import requests
import pandas as pd
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

app = FastAPI(title="Tadawul Short-Term AI")

# حل مشكلة الاتصال بين الواجهة والخلفية
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

TICKERS_PATH = "tickers_sa.txt"

def analyze_logic(ticker: str):
    t_clean = ticker.strip().upper()
    if not t_clean.endswith(".SR"): t_clean += ".SR"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    try:
        # 1. الفلتر المالي (Fundamental)
        f_url = f"https://query1.finance.yahoo.com/v10/finance/quoteSummary/{t_clean}?modules=summaryDetail,financialData"
        f_res = requests.get(f_url, headers=headers, timeout=10).json()
        fin = f_res['quoteSummary']['result'][0]
        
        pe = fin.get('summaryDetail', {}).get('trailingPE', {}).get('value', 999)
        roe = fin.get('financialData', {}).get('returnOnEquity', {}).get('value', 0)
        debt = fin.get('financialData', {}).get('debtToEquity', {}).get('value', 0)

        # 2. الفلتر الفني (Technical)
        c_url = f"https://query1.finance.yahoo.com/v8/finance/chart/{t_clean}?range=1y&interval=1d"
        c_res = requests.get(c_url, headers=headers, timeout=10).json()
        data = c_res['chart']['result'][0]
        ohlc = data['indicators']['quote'][0]
        df = pd.DataFrame({"close": ohlc['close'], "high": ohlc['high'], "low": ohlc['low']}).dropna()

        # حساب RSI و Bollinger
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
        
        ma20 = df['close'].rolling(20).mean()
        std20 = df['close'].rolling(20).std()
        df['lower_band'] = ma20 - (2 * std20)

        last = df.iloc[-1]
        prev = df.iloc[-2]
        price = float(last['close'])

        # تطبيق شروطك (استثمار قصير المدى)
        # مالي: P/E < 25 و ROE > 10%
        is_fin_ok = (pe < 25) and (roe > 0.10)
        # فني: ارتداد RSI أو ملامسة بولينجر سفلي
        is_tech_ok = (last['rsi'] > prev['rsi'] and prev['rsi'] < 40) or (price <= last['lower_band'] * 1.015)

        recommendation = "BUY" if (is_fin_ok and is_tech_ok) else "NO_TRADE"
        
        # توافق الأسماء مع الواجهة (تأكد من هذه الأسماء تحديداً)
        return {
            "ticker": t_clean.replace(".SR", ""),
            "recommendation": recommendation,
            "confidence_pct": 100 if recommendation == "BUY" else 0,
            "entry": round(price, 2),
            "target": round(price * 1.05, 2), # هدف 5%
            "stop": round(price * 0.97, 2),   # وقف 3%
            "last_close": round(price, 2),
            "reason": "تجميع إيجابي وزخم صاعد" if recommendation == "BUY" else "لا توجد إشارة"
        }
    except:
        return {"ticker": ticker, "recommendation": "ERROR", "confidence_pct": 0}

# المسارات التي كانت تسبب خطأ Not Found
@app.get("/predict")
def predict(ticker: str):
    return analyze_logic(ticker)

@app.get("/top10")
def get_top10():
    if not os.path.exists(TICKERS_PATH):
        return {"items": [], "error": "ملف الأسهم غير موجود"}
    
    with open(TICKERS_PATH, "r") as f:
        tickers = [line.strip() for line in f if line.strip()]
    
    results = []
    # تقليل عدد العمال لضمان استقرار ياهو فاينانس
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(analyze_logic, t) for t in tickers]
        for f in as_completed(futures):
            res = f.result()
            if res["recommendation"] == "BUY":
                results.append(res)
    
    # نأخذ أفضل 10 حسب الثقة
    final_list = sorted(results, key=lambda x: x['confidence_pct'], reverse=True)[:10]
    return {"items": final_list}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
