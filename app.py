from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import requests
import joblib
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# ------------------------
MODEL_PATH = "model.joblib"
TICKERS_PATH = "tickers_sa.txt"
ML_THRESHOLD = 0.60
TOP10_WORKERS = 6
FEATURE_COLS = [
    "sma10","sma20","sma50",
    "ema10","ema20","ema50",
    "rsi14","macd","macd_signal","macd_hist",
    "bb_width","atr14","ret1","ret3","ret5","vol20","vol_ratio"
]

# ------------------------
app = FastAPI(title="Tadawul MVP API", version="2.3.0-with-ML")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

model = None
_prices_cache = {}
CACHE_TTL_SEC = 600
_top10_cache = {"ts":0,"data":None}

# ------------------------
def _cache_get(key):
    v = _prices_cache.get(key)
    if not v: return None
    if time.time() - v["ts"] > CACHE_TTL_SEC:
        _prices_cache.pop(key, None)
        return None
    return v["df"]

def _cache_set(key, df):
    _prices_cache[key] = {"ts": time.time(), "df": df}

# ------------------------
def fetch_yahoo_prices(ticker: str, range_="2y", interval="1d"):
    key = (ticker, range_, interval)
    cached = _cache_get(key)
    if cached is not None: return cached
    bases = ["https://query1.finance.yahoo.com/v8/finance/chart/",
             "https://query2.finance.yahoo.com/v8/finance/chart/"]
    headers = {"User-Agent":"Mozilla/5.0"}
    for _ in range(3):
        for base in bases:
            try:
                url = f"{base}{ticker}"
                r = requests.get(url, params={"range": range_, "interval": interval}, headers=headers, timeout=20)
                if r.status_code != 200: continue
                js = r.json()
                result = (js.get("chart", {}).get("result") or [None])[0]
                if not result: continue
                quote = (result.get("indicators", {}).get("quote") or [None])[0]
                if not quote: continue
                df = pd.DataFrame({
                    "open": quote.get("open",[]),
                    "high": quote.get("high",[]),
                    "low": quote.get("low",[]),
                    "close": quote.get("close",[]),
                    "volume": quote.get("volume",[])
                }).dropna(subset=["close"]).reset_index(drop=True)
                if len(df) > 0:
                    _cache_set(key, df)
                    return df
            except: continue
        time.sleep(1)
    return None

# ------------------------
# مؤشرات فنية
def sma(series, window): return series.rolling(window).mean()
def ema(series, span): return series.ewm(span=span, adjust=False).mean()
def rsi(close, period=14):
    delta = close.diff()
    gain = delta.where(delta>0,0.0)
    loss = (-delta).where(delta<0,0.0)
    rs = gain.rolling(period).mean() / (loss.rolling(period).mean()+1e-9)
    return 100-(100/(1+rs))
def macd(close, fast=12, slow=26, signal=9):
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist
def bollinger(close, window=20, num_std=2):
    mid = sma(close, window)
    std = close.rolling(window).std()
    return mid+num_std*std, mid, mid-num_std*std
def atr(high, low, close, period=14):
    prev = close.shift(1)
    tr = pd.concat([high-low, (high-prev).abs(), (low-prev).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def build_features(df):
    d = df.copy()
    close=d["close"]; high=d["high"]; low=d["low"]; vol=d["volume"]
    d["sma10"]=sma(close,10); d["sma20"]=sma(close,20); d["sma50"]=sma(close,50)
    d["ema10"]=ema(close,10); d["ema20"]=ema(close,20); d["ema50"]=ema(close,50)
    d["rsi14"]=rsi(close,14)
    macd_line, sig_line, hist = macd(close)
    d["macd"]=macd_line; d["macd_signal"]=sig_line; d["macd_hist"]=hist
    bb_u, bb_m, bb_l = bollinger(close)
    d["bb_width"]=(bb_u-bb_l)/(bb_m+1e-9)
    d["atr14"]=atr(high, low, close)
    d["ret1"]=close.pct_change(1); d["ret3"]=close.pct_change(3); d["ret5"]=close.pct_change(5)
    d["vol20"]=d["ret1"].rolling(20).std(); d["vol_ma20"]=vol.rolling(20).mean()
    d["vol_ratio"]=vol/(d["vol_ma20"]+1e-9)
    return d.dropna().reset_index(drop=True)

def latest_feature_vector(feat_df):
    row = feat_df.iloc[-1][FEATURE_COLS].astype(float)
    return row.values.reshape(1,-1)

# ------------------------
@app.on_event("startup")
def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            print("Loaded model.joblib")
        except:
            model = None
    else:
        print("No model found. Run train_model.py first.")

# ------------------------
def analyze_one(ticker: str):
    t = ticker.strip().upper()
    if not t.endswith(".SR"): t += ".SR"
    df = fetch_yahoo_prices(t)
    if df is None or len(df)<10: 
        return {"ticker": t,"recommendation":"NO_TRADE","ml_confidence":None,"reason":"No price data"}
    feat_df = build_features(df)
    X = latest_feature_vector(feat_df)
    ml_conf = float(model.predict_proba(X)[0,1]) if model else None
    recommendation = "BUY" if ml_conf and ml_conf>=ML_THRESHOLD else "NO_TRADE"
    return {"ticker": t, "recommendation": recommendation, "ml_confidence": ml_conf, "last_close": float(feat_df["close"].iloc[-1])}

# ------------------------
@app.get("/predict")
def predict(ticker: str):
    return analyze_one(ticker)

@app.get("/top10")
def top10(max_workers: int = TOP10_WORKERS):
    now = time.time()
    if _top10_cache["data"] and now-_top10_cache["ts"] < 600:
        cached_payload = dict(_top10_cache["data"]); cached_payload["cached"]=True
        cached_payload["cached_age_sec"]=int(now-_top10_cache["ts"]); return cached_payload

    tickers=[]
    if os.path.exists(TICKERS_PATH):
        with open(TICKERS_PATH,"r") as f:
            tickers = [l.strip().upper() for l in f if l.strip()]
    if not tickers: return {"items": [], "error":"Tickers file missing","cached":False}

    results=[]; errors=0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures={ex.submit(analyze_one,t):t for t in tickers}
        for fut in as_completed(futures):
            try: results.append(fut.result())
            except: errors+=1

    top_items = sorted(results, key=lambda x: (x["recommendation"]=="BUY", float(x["ml_confidence"] or 0)), reverse=True)[:10]
    payload={"items":top_items,"total":len(results),"errors":errors,"cached":False,"computed_at_unix":int(time.time())}
    _top10_cache["ts"]=time.time(); _top10_cache["data"]=payload
    return payload
