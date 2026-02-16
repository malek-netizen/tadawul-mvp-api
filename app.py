from fastapi import FastAPI
import requests
import pandas as pd
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

app = FastAPI(title="Tadawul Pro API")

TICKERS_PATH = "tickers_sa.txt"

def fetch_full_data(ticker: str):
    t = ticker.strip().upper()
    if not t.endswith(".SR"): t += ".SR"
    
    headers = {"User-Agent": "Mozilla/5.0"}
    # جلب بيانات السعر والمؤشرات المالية الأساسية
    url = f"https://query1.finance.yahoo.com/v10/finance/quoteSummary/{t}?modules=summaryDetail,financialData,defaultKeyStatistics"
    chart_url = f"https://query1.finance.yahoo.com/v8/finance/chart/{t}?range=1y&interval=1d"
    
    try:
        # 1. سحب بيانات الشارت للمؤشرات الفنية
        r_chart = requests.get(chart_url, headers=headers, timeout=10)
        chart_data = r_chart.json()['chart']['result'][0]
        ohlc = chart_data['indicators']['quote'][0]
        df = pd.DataFrame(ohlc)
        df = df.dropna().tail(100) # نأخذ آخر 100 يوم

        # 2. سحب البيانات المالية (P/E, EPS, etc)
        r_meta = requests.get(url, headers=headers, timeout=10)
        meta = r_meta.json()['quoteSummary']['result'][0]
        
        return df, meta, chart_data['meta']
    except:
        return None, None, None

def analyze_one(ticker: str):
    df, financial, meta = fetch_full_data(ticker)
    if df is None or len(df) < 20:
        return {"ticker": ticker, "recommendation": "ERROR", "confidence_pct": 0}

    # --- الحسابات الفنية ---
    close = df['close']
    last_c = close.iloc[-1]
    prev_c = close.iloc[-2]
    
    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rsi = 100 - (100 / (1 + (gain / (loss + 1e-9)))).iloc[-1]

    # MACD
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    macd_line = ema12 - ema26
    signal = macd_line.ewm(span=9).mean()
    hist = (macd_line - signal).iloc[-1]
    prev_hist = (macd_line - signal).iloc[-2]

    # --- القوائم المالية (Fundamental) ---
    pe_ratio = financial.get('summaryDetail', {}).get('trailingPE', {}).get('value', 0)
    eps = financial.get('defaultKeyStatistics', {}).get('trailingEps', {}).get('value', 0)

    # --- نظام الثقة المعقد (Confidence Scoring) ---
    score = 0
    reasons = []

    # 1. القوة النسبية (RSI) - الوزن: 25%
    if 40 <= rsi <= 60: score += 25
    elif rsi < 40: score += 15 # تشبع بيعي يحتاج تأكيد صعود

    # 2. اتجاه السعر (Price Action) - الوزن: 30%
    if last_c > prev_c: score += 30
    elif last_c >= prev_c * 0.995: score += 10 # استقرار

    # 3. الزخم (MACD) - الوزن: 25%
    if hist > prev_hist and hist > 0: score += 25
    elif hist > prev_hist: score += 15 # تحسن في الزخم

    # 4. الأساس المالي (P/E) - الوزن: 20%
    if 0 < pe_ratio < 20: score += 20 # مكرر ربحية مغري
    elif 20 <= pe_ratio < 35: score += 10

    # --- القرار النهائي ---
    recommendation = "BUY" if score >= 65 else "NO_TRADE"
    
    return {
        "ticker": ticker,
        "recommendation": recommendation,
        "confidence_pct": int(score),
        "price_data": {
            "open": round(df['open'].iloc[-1], 2),
            "high": round(df['high'].iloc[-1], 2),
            "low": round(df['low'].iloc[-1], 2),
            "close": round(last_c, 2),
            "prev_close": round(prev_c, 2)
        },
        "financials": {
            "P/E": pe_ratio,
            "EPS": eps,
            "currency": meta.get('currency', 'SAR')
        },
        "technical_values": {
            "RSI": round(rsi, 2),
            "MACD_Hist": round(hist, 4)
        },
        "reason": "قوة فنية ومالية مشتركة" if score > 80 else "تجميع معتدل"
    }

@app.get("/predict")
def predict(ticker: str):
    return analyze_one(ticker)
