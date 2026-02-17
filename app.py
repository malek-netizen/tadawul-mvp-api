from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import requests
import pandas as pd
import numpy as np
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

app = FastAPI(title="Tadawul Sniper Pro", version="2.7.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

TICKERS_PATH = "tickers_sa.txt"
TP_PCT = 0.05
TOP10_WORKERS = 10     
_prices_cache = {}
CACHE_TTL_SEC = 600

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
        df = pd.DataFrame({
            "close": quote["close"], "high": quote["high"], "low": quote["low"], "volume": quote["volume"]
        }).dropna(subset=["close"]).reset_index(drop=True)
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
    d["bb_mid"] = d["sma20"]
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    d["rsi14"] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
    d["vol_ma20"] = d["volume"].rolling(20).mean()
    return d.dropna().reset_index(drop=True)

def analyze_one(ticker: str):
    t = ticker.strip().upper()
    if not t.endswith(".SR"): t += ".SR"
    df = fetch_yahoo_prices(t)
    if df is None or len(df) < 30: return None
    
    feat_df = build_features(df)
    curr = feat_df.iloc[-1]
    prev = feat_df.iloc[-2]
    last_close = float(curr["close"])
    
    # --- منطق الفلترة (الماكد المطور) ---
    reasons = []
    if not (curr['volume'] > (curr['vol_ma20'] * 1.2)): reasons.append("ضعف سيولة")
    
    # الماكد: تقاطع صاعد + (تحت الصفر أو اتساع الفجوة الإيجابية)
    cond_macd_cross = (curr['macd'] > curr['macd_signal'])
    cond_from_bottom = curr['macd'] < 0.005 
    momentum_growing = (curr['macd'] - curr['macd_signal']) > (prev['macd'] - prev['macd_signal'])
    
    if not (cond_macd_cross and (cond_from_bottom or momentum_growing)):
        reasons.append("الماكد لم يستدر")

    if not (curr['close'] > curr['bb_mid']): reasons.append("تحت منتصف البولينجر")
    if curr['rsi14'] > 65: reasons.append("تضخم RSI")

    ok = (len(reasons) == 0)
    
    # حساب الثقة
    score = 0
    if curr['volume'] > curr['vol_ma20'] * 1.5: score += 40
    if cond_macd_cross: score += 30
    if curr['close'] > curr['bb_mid']: score += 30
    if curr['rsi14'] > 65: score -= 40
    conf = max(0, min(100, score))

    # الوقف الفني
    recent_low = float(feat_df['low'].tail(3).min())
    stop_l = round(recent_low * 0.997, 2)

    return {
        "ticker": t,
        "recommendation": "BUY" if ok else "NO_TRADE",
        "confidence": conf,
        "entry": round(last_close, 2),
        "tp": round(last_close * (1 + TP_PCT), 2),
        "sl": stop_l,
        "reason": " | ".join(reasons) if reasons else "نموذج انفجاري: سيولة + ماكد مستد",
        "last_close": round(last_close, 2),
        "status": "APPROVED" if ok else "REJECTED"
    }

@app.get("/top10")
def top10():
    if not os.path.exists(TICKERS_PATH): return []
    with open(TICKERS_PATH, "r") as f:
        tickers = [line.strip() for line in f if line.strip()]
    
    results = []
    with ThreadPoolExecutor(max_workers=TOP10_WORKERS) as executor:
        futures = [executor.submit(analyze_one, t) for t in tickers]
        for fut in as_completed(futures):
            res = fut.result()
            if res: results.append(res)
    
    # الترتيب الرقمي القوي لضمان عدم الانعكاس
    def get_weight(item):
        return (1000 if item['status'] == 'APPROVED' else 0) + item['confidence']

    results.sort(key=get_weight, reverse=True)
    
    top_items = results[:10]
    
    # إضافة إحصائية السوق في أول سهم (ليظهر في الواجهة القديمة)
    if top_items:
        scanned = len(results)
        found = len([r for r in results if r['status'] == 'APPROVED'])
        top_items[0]['reason'] = f"🔍 [تحليل {scanned} سهم | وجدنا {found}] - " + top_items[0]['reason']

    return top_items

@app.get("/predict")
def predict(ticker: str):
    return analyze_one(ticker)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
