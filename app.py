from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import requests
import pandas as pd
import numpy as np
import time
import joblib
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

app = FastAPI(title="Tadawul Sniper Pro", version="2.2.1-Integrated")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "model.joblib"
TICKERS_PATH = "tickers_sa.txt"

# إعدادات الأهداف والأمان (ضبط القاع)
TP_PCT = 0.05
SL_PCT = 0.02
RSI_SAFE_MAX = 60      # سقف الأمان (يمنع تضخم الأهلي)
EMA_STRETCH_MAX = 0.03 # أقصى ابتعاد عن المتوسط 3%
TOP10_WORKERS = 10     # سرعة التحليل

model = None
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
        df = pd.DataFrame(quote)
        df = df.dropna(subset=["close"]).reset_index(drop=True)
        _prices_cache[key] = {"ts": time.time(), "df": df}
        return df
    except: return None

def build_features(df: pd.DataFrame):
    d = df.copy()
    close = d["close"]
    # المتوسطات
    d["ema20"] = close.ewm(span=20, adjust=False).mean()
    d["sma20"] = close.rolling(20).mean()
    # الماكد
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    d["macd"] = ema12 - ema26
    d["macd_signal"] = d["macd"].ewm(span=9, adjust=False).mean()
    # البولينجر
    std20 = close.rolling(20).std()
    d["bb_mid"] = d["sma20"]
    d["bb_upper"] = d["sma20"] + (std20 * 2)
    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    d["rsi14"] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
    # السيولة
    d["vol_ma20"] = d["volume"].rolling(20).mean()
    return d.dropna().reset_index(drop=True)

# =========================
# 2) منطق "قناص القيعان" (Core Logic)
# =========================
def passes_rules(feat_df: pd.DataFrame):
    reasons = []
    curr = feat_df.iloc[-1]
    prev = feat_df.iloc[-2]
    
    # 1. الماكد: استدارة من منطقة منخفضة (شرط ساسكو)
    cond_macd_bottom = (curr['macd'] > curr['macd_signal']) and (prev['macd'] <= prev['macd_signal']) and (curr['macd'] < 0.15)
    if not cond_macd_bottom:
        # سماح بسيط لو الزخم لسه في بدايته وقريب من الصفر
        if not (curr['macd'] > curr['macd_signal'] and curr['macd'] < 0.2 and (curr['macd']-curr['macd_signal'] > prev['macd']-prev['macd_signal'])):
            reasons.append("Rejected: MACD not turning from base.")

    # 2. البولينجر: اختراق المنتصف (Pivot point)
    cond_bb_break = (curr['close'] > curr['bb_mid']) and (prev['close'] <= prev['bb_mid'])
    was_below = (feat_df['close'].shift(1).tail(5) < feat_df['bb_mid'].shift(1).tail(5)).any()
    if not (cond_bb_break or (curr['close'] > curr['bb_mid'] and was_below)):
        reasons.append("Rejected: No BB mid-band breakout.")

    # 3. RSI الأمان: (يمنع تضخم الأهلي)
    if curr['rsi14'] > RSI_SAFE_MAX:
        reasons.append(f"Rejected: RSI too high ({round(curr['rsi14'],1)})")
    if not (curr['rsi14'] > prev['rsi14']):
        reasons.append("Rejected: RSI not rising.")

    # 4. استدارة المتوسط (MA Slope)
    is_ma_turning = curr['sma20'] >= feat_df['sma20'].iloc[-4]
    if not is_ma_turning:
        reasons.append("Rejected: SMA20 slope is downward.")

    # 5. الأمان من المطاردة
    dist = (curr['close'] - curr['ema20']) / curr['ema20']
    if dist > EMA_STRETCH_MAX:
        reasons.append("Rejected: Price overextended from EMA20.")

    return (len(reasons) == 0), reasons

def calculate_confidence(feat_df: pd.DataFrame):
    curr = feat_df.iloc[-1]
    prev = feat_df.iloc[-2]
    score = 0
    if (curr['macd'] > curr['macd_signal']) and (curr['macd'] < 0.1): score += 35
    if (curr['close'] > curr['bb_mid']): score += 25
    if (curr['rsi14'] > prev['rsi14'] and curr['rsi14'] < 55): score += 20
    if (curr['volume'] > curr['vol_ma20']): score += 20
    return min(100, score)

# =========================
# 3) تحليل سهم واحد (The Analyzer)
# =========================
def analyze_one(ticker: str):
    t = ticker.strip().upper()
    if not t.endswith(".SR"): t += ".SR"
    
    df = fetch_yahoo_prices(t)
    if df is None or len(df) < 30: return None
    
    feat_df = build_features(df)
    last_close = float(feat_df["close"].iloc[-1])
    
    ok, reasons = passes_rules(feat_df)
    conf_pct = calculate_confidence(feat_df)
    
    # دمج الـ ML إذا كان الملف موجوداً
    ml_conf = None
    if model:
        try:
            # هنا نفترض وجود موديل مدرب على نفس ميزات FEATURE_COLS
            pass 
        except: ml_conf = None

    recommendation = "BUY" if ok else "NO_TRADE"
    
    return {
        "ticker": t,
        "recommendation": recommendation,
        "confidence_pct": conf_pct,
        "entry": round(last_close, 2),
        "take_profit": round(last_close * (1 + TP_PCT), 2),
        "stop_loss": round(last_close * (1 - SL_PCT), 2),
        "reason": " | ".join(reasons) if reasons else "قاع حقيقي + استدارة فنية",
        "last_close": round(last_close, 2),
        "status": "APPROVED" if ok else "REJECTED"
    }

# =========================
# 4) الروابط (Endpoints)
# =========================
@app.on_event("startup")
def startup():
    global model
    if os.path.exists(MODEL_PATH): model = joblib.load(MODEL_PATH)

@app.get("/predict")
def predict(ticker: str):
    return analyze_one(ticker)

@app.get("/top10")
def top10():
    if not os.path.exists(TICKERS_PATH): return {"error": "File not found"}
    with open(TICKERS_PATH, "r") as f:
        tickers = [line.strip() for line in f if line.strip()]
    
    results = []
    with ThreadPoolExecutor(max_workers=TOP10_WORKERS) as executor:
        futures = [executor.submit(analyze_one, t) for t in tickers]
        for fut in as_completed(futures):
            res = fut.result()
            if res: results.append(res)
    
    # الترتيب: المقبول أولاً ثم حسب القوة
    sorted_res = sorted(results, key=lambda x: (x['recommendation'] == 'BUY', x['confidence_pct']), reverse=True)
    return {"items": sorted_res[:10], "total_scanned": len(results)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
