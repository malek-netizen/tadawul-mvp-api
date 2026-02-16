from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import requests
import pandas as pd
import numpy as np
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

app = FastAPI(title="Tadawul Pro Analyzer", version="3.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

TICKERS_PATH = "tickers_sa.txt"
TP_PCT = 0.05
SL_PCT = 0.02

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
    df['ema12'] = df['close'].ewm(span=12).mean()
    df['ema26'] = df['close'].ewm(span=26).mean()
    df['macd'] = df['ema12'] - df['ema26']
    df['signal'] = df['macd'].ewm(span=9).mean()
    df['sma20'] = df['close'].rolling(20).mean()
    df['std20'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['sma20'] + (df['std20'] * 2)
    df['bb_mid'] = df['sma20']
    df['ema200'] = df['close'].ewm(span=200).mean()
    df['res_20'] = df['high'].rolling(20).max().shift(1)
    df['vol_avg'] = df['volume'].rolling(20).mean()
    return df

def analyze_one(ticker: str):
    t = ticker.strip().upper()
    if not t.endswith(".SR"): t += ".SR"
    
    # 1. تحليل اليومي (فلتر الاتجاه العام)
    df_day = fetch_data(t, interval="1d", period="1y")
    if df_day is None or len(df_day) < 200:
        return {"ticker": t, "recommendation": "NO_TRADE", "reason": "نقص بيانات", "confidence_pct": 0}
    
    df_day = apply_indicators(df_day)
    is_trend_up = df_day.iloc[-1]['close'] > df_day.iloc[-1]['ema200']
    
    # 2. تحليل الساعة (فلتر التأكيد اللحظي)
    df_hour = fetch_data(t, interval="1h", period="60d")
    if df_hour is None or len(df_hour) < 30:
        return {"ticker": t, "recommendation": "NO_TRADE", "reason": "نقص بيانات", "confidence_pct": 0}
    
    # حساب RSI
    delta = df_hour['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df_hour['rsi'] = 100 - (100 / (1 + (gain / loss)))
    
    df_hour = apply_indicators(df_hour)
    curr = df_hour.iloc[-1]
    prev = df_hour.iloc[-2]
    
    # شروط التأكيد الصارمة
    cond_macd = (curr['macd'] > curr['signal']) and (curr['macd'] > prev['macd']) # تقاطع إيجابي حقيقي
    cond_rsi = (curr['rsi'] < 70) and (curr['rsi'] > prev['rsi']) # قوة شرائية صاعدة وغير متضخمة
    cond_vol = curr['volume'] > (curr['vol_avg'] * 1.10) # سيولة تأكيد (أعلى بـ 10%)
    cond_bb = curr['close'] > curr['bb_mid'] # السعر في المنطقة الإيجابية
    cond_res = curr['close'] >= curr['res_20'] # اختراق مقاومة قريبة

    # توزيع النقاط (الخماسية الذهبية)
    score = 0
    if is_trend_up: score += 30
    if cond_macd:   score += 20
    if cond_rsi:    score += 15
    if cond_vol:    score += 15
    if cond_bb:     score += 10
    if cond_res:    score += 10
    
    recommendation = "BUY" if score >= 75 else "NO_TRADE"
    
    return {
        "ticker": t,
        "recommendation": recommendation,
        "confidence_pct": score,
        "entry": round(float(curr['close']), 2),
        "take_profit": round(float(curr['close']) * 1.05, 2),
        "stop_loss": round(float(curr['close']) * 0.98, 2),
        "last_close": round(float(curr['close']), 2), # تم الإصلاح
        "reason": f"Trend:{is_trend_up}|MACD:{cond_macd}|RSI:{cond_rsi}|Vol:{cond_vol}",
        "status": "APPROVED" if recommendation == "BUY" else "REJECTED"
    }
@app.get("/predict")
def predict(ticker: str):
    return analyze_one(ticker)

@app.get("/top10")
def top10():
    if not os.path.exists(TICKERS_PATH): 
        return {"items": [], "error": "ملف الأسهم مفقود"}
    
    with open(TICKERS_PATH, "r") as f:
        tickers = [line.strip() for line in f if line.strip()]
    
    results = []
    # رفعنا عدد العمال لـ 15 لتسريع مسح السوق كاملاً
    with ThreadPoolExecutor(max_workers=15) as executor:
        futures = [executor.submit(analyze_one, t) for t in tickers]
        for fut in as_completed(futures):
            try:
                res = fut.result()
                # نستبعد فقط الأسهم التي فشل جلب بياناتها تماماً
                if "بيانات" not in res.get("reason", ""):
                    results.append(res)
            except:
                continue
    
    # الترتيب: الأعلى ثقة في البداية (حتى لو لم تكن BUY)
    sorted_res = sorted(results, key=lambda x: x['confidence_pct'], reverse=True)
    
    # نأخذ أفضل 10 فرص موجودة في السوق حالياً
    top_items = sorted_res[:10]
    
    return {"items": top_items, "total_scanned": len(results)}
    
@app.get("/health")
def health():
    return {"status": "Online"}
