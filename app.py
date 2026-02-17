from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import requests
import pandas as pd
import numpy as np
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

app = FastAPI(title="Tadawul Sniper Pro", version="2.5.0-Final-Optimized")

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

# =========================
# 1) جلب البيانات والمؤشرات
# =========================
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
            "close": quote["close"],
            "high": quote["high"],
            "low": quote["low"],
            "volume": quote["volume"]
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

# =========================
# 2) القواعد ونظام الثقة
# =========================
def passes_rules(feat_df: pd.DataFrame):
    reasons = []
    curr = feat_df.iloc[-1]
    prev = feat_df.iloc[-2]
    
    # سيولة انفجارية
    if not (curr['volume'] > (curr['vol_ma20'] * 1.2)):
        reasons.append("Weak Volume")

    # [MACD المطور]: تقاطع صاعد + (تحت الصفر أو زخم متزايد)
    cond_macd_cross = (curr['macd'] > curr['macd_signal'])
    cond_from_bottom = curr['macd'] < 0.001 
    momentum_growing = (curr['macd'] - curr['macd_signal']) > (prev['macd'] - prev['macd_signal'])
    
    if not (cond_macd_cross and (cond_from_bottom or momentum_growing)):
        reasons.append("MACD not in reversal phase")

    # اختراق بولينجر
    cond_bb_break = (curr['close'] > curr['bb_mid']) and (prev['close'] <= prev['bb_mid'])
    was_below = (feat_df['close'].shift(1).tail(5) < feat_df['bb_mid'].shift(1).tail(5)).any()
    if not (cond_bb_break or (curr['close'] > curr['bb_mid'] and was_below)):
        reasons.append("No Mid-BB breakout")

    # أمان RSI و EMA
    if curr['rsi14'] > 65: reasons.append(f"RSI high ({round(curr['rsi14'],1)})")
    if ((curr['close'] - curr['ema20']) / curr['ema20']) > 0.03: reasons.append("Overextended price")

    return (len(reasons) == 0), reasons

def calculate_confidence(feat_df: pd.DataFrame):
    curr = feat_df.iloc[-1]
    score = 0
    if curr['volume'] > curr['vol_ma20'] * 1.5: score += 40
    elif curr['volume'] > curr['vol_ma20'] * 1.2: score += 25
    if (curr['macd'] > curr['macd_signal']): score += 30
    if (curr['close'] > curr['bb_mid']): score += 30
    
    # الخصم (Penalty) للأمان
    if curr['rsi14'] > 65: score -= 40
    if ((curr['close'] - curr['ema20']) / curr['ema20']) > 0.03: score -= 50
    return max(0, min(100, score))

# =========================
# 3) المحلل الرئيسي
# =========================
def analyze_one(ticker: str):
    t = ticker.strip().upper()
    if not t.endswith(".SR"): t += ".SR"
    df = fetch_yahoo_prices(t)
    if df is None or len(df) < 30: return None
    
    feat_df = build_features(df)
    curr = feat_df.iloc[-1]
    last_close = float(curr["close"])
    
    # الوقف الفني: أدنى سعر في آخر 3 شموع
    recent_low = float(feat_df['low'].tail(3).min())
    stop_l = round(recent_low * 0.997, 2)
    dist = (last_close - stop_l) / last_close
    if dist > 0.04: stop_l = round(last_close * 0.96, 2)
    elif dist < 0.015: stop_l = round(last_close * 0.985, 2)
    
    ok, reasons = passes_rules(feat_df)
    conf = calculate_confidence(feat_df)
    
    return {
        "ticker": t,
        "recommendation": "BUY" if ok else "NO_TRADE",
        "confidence_pct": conf,
        "entry": round(last_close, 2),
        "take_profit": round(last_close * (1 + TP_PCT), 2),
        "stop_loss": stop_l,
        "reason": " | ".join(reasons) if reasons else "اختراق قاع بسيولة انفجارية",
        "last_close": round(last_close, 2),
        "status": "APPROVED" if ok else "REJECTED"
    }

# =========================
# 4) الروابط (Endpoints)
# =========================
@app.get("/top10")
def top10():
    if not os.path.exists(TICKERS_PATH): return {"error": "tickers_sa.txt not found"}
    with open(TICKERS_PATH, "r") as f:
        tickers = [line.strip() for line in f if line.strip()]
    
    results = []
    with ThreadPoolExecutor(max_workers=TOP10_WORKERS) as executor:
        futures = [executor.submit(analyze_one, t) for t in tickers]
        for fut in as_completed(futures):
            res = fut.result()
            if res: results.append(res)
    
    # دالة الترتيب لضمان عدم الانعكاس: APPROVED تأخذ وزن 1000
    def final_sorter(item):
        status_weight = 1000 if item['status'] == 'APPROVED' else 0
        return status_weight + item['confidence_pct']

    sorted_res = sorted(results, key=final_sorter, reverse=True)
    buy_signals = [r for r in results if r['status'] == 'APPROVED']
    
    return {
        "market_summary": {
            "total_scanned": len(results),
            "buy_signals_found": len(buy_signals),
            "market_health": "Bullish" if len(buy_signals) > 5 else "Cautious"
        },
        "items": sorted_res[:10]
    }

@app.get("/predict")
def predict(ticker: str):
    return analyze_one(ticker)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
