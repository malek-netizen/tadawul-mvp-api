from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import requests
import pandas as pd
import numpy as np
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

app = FastAPI(title="Tadawul Pro Analyzer - Analyst Logic", version="5.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

TICKERS_PATH = "tickers_sa.txt"

def fetch_data(ticker: str, interval="1h", period="60d"):
    bases = ["https://query1.finance.yahoo.com/v8/finance/chart/", "https://query2.finance.yahoo.com/v8/finance/chart/"]
    headers = {"User-Agent": "Mozilla/5.0"}
    for base in bases:
        try:
            url = f"{base}{ticker}"
            r = requests.get(url, params={"range": period, "interval": interval}, headers=headers, timeout=10)
            if r.status_code != 200: continue
            js = r.json()
            quote = js['chart']['result'][0]['indicators']['quote'][0]
            df = pd.DataFrame({
                "open": quote['open'], "high": quote['high'], 
                "low": quote['low'], "close": quote['close'], "volume": quote['volume']
            }).dropna().reset_index(drop=True)
            return df
        except: continue
    return None

def apply_indicators(df):
    # MACD
    df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema12'] - df['ema26']
    df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    # Bollinger Bands
    df['sma20'] = df['close'].rolling(20).mean()
    df['std20'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['sma20'] + (df['std20'] * 2)
    df['bb_mid'] = df['sma20']
    # Others
    df['res_20'] = df['high'].rolling(20).max().shift(1)
    df['vol_avg'] = df['volume'].rolling(20).mean()
    return df

def analyze_one(ticker: str):
    try:
        t = ticker.strip().upper()
        if not t.endswith(".SR"): t += ".SR"
        
        df_hour = fetch_data(t, interval="1h", period="60d")
        if df_hour is None or len(df_hour) < 30: return None

        # --- الفلتر الذكي للبيانات ---
        # إذا كان حجم تداول الشمعة الأخيرة صفر أو ضعيف جداً (وقت إغلاق)
        # ننتقل للشمعة التي قبلها لضمان دقة المؤشرات
        last_vol = df_hour.iloc[-1]['volume']
        avg_vol = df_hour['volume'].tail(20).mean()
        
        if last_vol < (avg_vol * 0.1): # الشمعة الأخيرة "خاملة" (وقت إغلاق)
            df_active = df_hour.iloc[:-1]
        else:
            df_active = df_hour # السوق مفتوح والسيولة تتدفق
            
        df_active = apply_indicators(df_active)
        curr = df_active.iloc[-1]
        prev = df_active.iloc[-2]

        # --- شروط "عينك" (منطق ساسكو) ---
        cond_macd = curr['macd'] > curr['signal']
        # مساحة للصعود (البولينجر)
        cond_bb_space = curr['close'] < (curr['bb_mid'] + (curr['bb_upper'] - curr['bb_mid']) * 0.6)
        # سيولة حقيقية في آخر جلسة نشطة
        cond_vol = curr['volume'] > (df_active['volume'].rolling(20).mean().iloc[-1] * 0.9)
        # RSI مريح
        cond_rsi = curr['rsi'] < 65

        score = 0
        if cond_macd:     score += 40
        if cond_bb_space: score += 20
        if cond_vol:      score += 20
        if cond_rsi:      score += 20

        recommendation = "BUY" if score >= 80 else "NO_TRADE"

        return {
            "ticker": t,
            "recommendation": recommendation,
            "confidence_pct": score,
            "entry": round(float(curr['close']), 2),
            "take_profit": round(float(curr['close']) * 1.05, 2),
            "stop_loss": round(float(curr['close']) * 0.97, 2),
            "last_close": round(float(curr['close']), 2),
            "reason": f"M:{int(cond_macd)}|V:{int(cond_vol)}|B:{int(cond_bb_space)}",
            "status": "APPROVED" if recommendation == "BUY" else "REJECTED"
        }
    except: return None
@app.get("/top10")
def top10():
    if not os.path.exists(TICKERS_PATH): return {"items": [], "error": "Missing file"}
    with open(TICKERS_PATH, "r") as f:
        tickers = [line.strip() for line in f if line.strip()]
    
    results = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(analyze_one, t) for t in tickers]
        for fut in as_completed(futures):
            res = fut.result()
            if res["confidence_pct"] > 0: results.append(res)
    
    # ترتيب حسب القوة
    sorted_res = sorted(results, key=lambda x: x['confidence_pct'], reverse=True)
    return {"items": sorted_res[:10], "total_scanned": len(results)}

@app.get("/predict")
def predict(ticker: str):
    return analyze_one(ticker)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
