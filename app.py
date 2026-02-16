from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import requests
import pandas as pd
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

app = FastAPI(title="Tadawul Pro - Sniper Edition", version="6.0.0")

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
    
    # Bollinger Bands & SMA20
    df['sma20'] = df['close'].rolling(20).mean()
    df['std20'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['sma20'] + (df['std20'] * 2)
    df['bb_mid'] = df['sma20']
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['rsi'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
    
    # Volume Average
    df['vol_avg'] = df['volume'].rolling(20).mean()
    return df

def analyze_one(ticker: str):
    try:
        t = ticker.strip().upper()
        if not t.endswith(".SR"): t += ".SR"
        
        df_raw = fetch_data(t, interval="1h", period="60d")
        if df_raw is None or len(df_raw) < 30: return None

        # الفلتر الذكي: تجاوز شمعة المزاد لو السوق مقفل
        last_vol = df_raw.iloc[-1]['volume']
        avg_vol_l = df_raw['volume'].tail(20).mean()
        df_active = df_raw.iloc[:-1] if last_vol < (avg_vol_l * 0.1) else df_raw
            
        df_active = apply_indicators(df_active)
        curr = df_active.iloc[-1]
        prev = df_active.iloc[-2]

        # --- شروط "منطق ساسكو" مع فلاتر الأمان ---
        
        # 1. الماكد: تقاطع إيجابي + فجوة تتوسع (زخم صاعد)
        macd_gap = curr['macd'] - curr['signal']
        prev_macd_gap = prev['macd'] - prev['signal']
        cond_macd = (curr['macd'] > curr['signal']) and (macd_gap > prev_macd_gap)

        # 2. RSI: صاعد (أعلى من السابق) + تحت الـ 60 (يستبعد الأهلي المتضخم)
        cond_rsi = (curr['rsi'] > prev['rsi']) and (curr['rsi'] < 60)

        # 3. الأمان: السعر غير متضخم (قريب من المتوسط بحد أقصى 3%)
        is_not_extended = (curr['close'] < curr['sma20'] * 1.03)

        # 4. السيولة والمساحة
        cond_vol = curr['volume'] > (curr['vol_avg'] * 0.8) # مرونة بسيطة في السيولة
        cond_bb_space = curr['close'] < (curr['bb_mid'] + (curr['bb_upper'] - curr['bb_mid']) * 0.5)

        score = 0
        if cond_macd:       score += 30
        if cond_rsi:        score += 30
        if is_not_extended: score += 20
        if cond_vol:        score += 10
        if cond_bb_space:   score += 10

        recommendation = "BUY" if score >= 80 else "NO_TRADE"

        return {
            "ticker": t,
            "recommendation": recommendation,
            "confidence_pct": score,
            "entry": round(float(curr['close']), 2),
            "take_profit": round(float(curr['close']) * 1.05, 2),
            "stop_loss": round(float(curr['close']) * 0.97, 2),
            "last_close": round(float(curr['close']), 2),
            "reason": f"M:{int(cond_macd)}|R:{int(cond_rsi)}|Safe:{int(is_not_extended)}",
            "status": "APPROVED" if recommendation == "BUY" else "REJECTED"
        }
    except:
        return None

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
            if res and res.get("confidence_pct", 0) > 0:
                results.append(res)
    
    # ترتيب النتائج حسب النسبة الأعلى
    sorted_res = sorted(results, key=lambda x: x['confidence_pct'], reverse=True)
    return {"items": sorted_res[:10], "total_scanned": len(results)}

@app.get("/predict")
def predict(ticker: str):
    res = analyze_one(ticker)
    return res if res else {"error": "Could not analyze ticker"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
