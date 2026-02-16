import pandas as pd
import numpy as np
import requests
from fastapi import FastAPI
from concurrent.futures import ThreadPoolExecutor, as_completed

app = FastAPI()

def analyze_short_term(ticker: str):
    t = ticker.strip().upper()
    if not t.endswith(".SR"): t += ".SR"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    try:
        # 1. سحب البيانات المالية (الأساسيات)
        f_url = f"https://query1.finance.yahoo.com/v10/finance/quoteSummary/{t}?modules=summaryDetail,financialData"
        f_res = requests.get(f_url, headers=headers, timeout=10).json()
        fin = f_res['quoteSummary']['result'][0]
        
        pe = fin.get('summaryDetail', {}).get('trailingPE', {}).get('value', 999)
        roe = fin.get('financialData', {}).get('returnOnEquity', {}).get('value', 0)
        debt = fin.get('financialData', {}).get('debtToEquity', {}).get('value', 0)

        # 2. سحب البيانات الفنية (الحركة السعرية)
        c_url = f"https://query1.finance.yahoo.com/v8/finance/chart/{t}?range=6mo&interval=1d"
        c_res = requests.get(c_url, headers=headers, timeout=10).json()
        data = c_res['chart']['result'][0]
        close_prices = data['indicators']['quote'][0]['close']
        df = pd.DataFrame({"close": close_prices, "high": data['indicators']['quote'][0]['high'], "low": data['indicators']['quote'][0]['low']}).dropna()

        # حساب المؤشرات
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
        
        # Bollinger Bands
        df['ma20'] = df['close'].rolling(20).mean()
        df['std'] = df['close'].rolling(20).std()
        df['lower_band'] = df['ma20'] - (2 * df['std'])

        last = df.iloc[-1]
        prev = df.iloc[-2]
        current_price = float(last['close'])

        # --- شروط القناص (Short-Term Filters) ---
        # مالي: شركة رابحة وديون معقولة
        is_fundamental_ok = (pe < 25) and (roe > 0.10) and (debt < 150)
        
        # فني: ارتداد من قاع البولينجر أو ارتداد RSI
        is_technical_ok = (
            (current_price <= last['lower_band'] * 1.015) or # قريب جداً من النطاق السفلي
            (prev['rsi'] < 35 and last['rsi'] > prev['rsi']) # ارتداد من تشبع بيعي
        )

        recommendation = "BUY" if (is_fundamental_ok and is_technical_ok) else "NO_TRADE"
        
        # حساب الأهداف (تثبيت الـ 5% مكسب كما طلبت)
        target = current_price * 1.05
        stop_loss = current_price * 0.97 # وقف خسارة 3% للمحافظة على رأس المال

        return {
            "ticker": t.replace(".SR", ""),
            "recommendation": recommendation,
            "confidence_pct": 100 if recommendation == "BUY" else 0,
            "entry": round(current_price, 2),
            "target": round(target, 2),
            "stop": round(stop_loss, 2),
            "last_close": round(current_price, 2),
            "reason": "قوة مالية + ارتداد فني وشيك" if recommendation == "BUY" else "لا توجد إشارة دخول"
        }
    except:
        return {"ticker": ticker, "recommendation": "ERROR"}

# ... دالة top10 تستدعي analyze_short_term ...
