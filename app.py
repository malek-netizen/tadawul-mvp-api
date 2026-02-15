from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import requests
import pandas as pd
import numpy as np
import time
import joblib
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

app = FastAPI(title="Tadawul MVP API", version="2.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "model.joblib"
TICKERS_PATH = "tickers_sa.txt"

TP_PCT = 0.05
SL_PCT = 0.02
RSI_MIN = 40
RSI_MAX = 65
RSI_OVERBOUGHT = 70
MAX_ABOVE_EMA20 = 0.06
MAX_ABOVE_EMA50 = 0.08
LOOKBACK_LOW_DAYS = 10
MIN_TREND_SCORE = 2
ML_THRESHOLD = 0.60
TOP10_WORKERS = 6

model = None
_prices_cache = {}
CACHE_TTL_SEC = 600
_top10_cache = {"ts": 0, "data": None}
TOP10_CACHE_TTL_SEC = 600

# =========================
# Utilities
# =========================
def _cache_get(key):
    v = _prices_cache.get(key)
    if not v: return None
    if time.time() - v["ts"] > CACHE_TTL_SEC:
        _prices_cache.pop(key, None)
        return None
    return v["df"]

def _cache_set(key, df):
    _prices_cache[key] = {"ts": time.time(), "df": df}

def fetch_yahoo_prices(ticker: str, range_="1y", interval="1d", min_rows=80):
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
                r = requests.get(url, params={"range":range_, "interval":interval}, headers=headers, timeout=15)
                if r.status_code != 200: continue
                js = r.json()
                result = (js.get("chart",{}).get("result") or [None])[0]
                if not result: continue
                quote = (result.get("indicators",{}).get("quote") or [None])[0]
                if not quote: continue
                df = pd.DataFrame({
                    "open": quote.get("open",[]),
                    "high": quote.get("high",[]),
                    "low": quote.get("low",[]),
                    "close": quote.get("close",[]),
                    "volume": quote.get("volume",[])
                }).dropna(subset=["close"]).reset_index(drop=True)
                if len(df) >= min_rows:
                    _cache_set(key, df)
                    return df
            except Exception: continue
        time.sleep(1)
    return None

# =========================
# Indicators
# =========================
def sma(series, window): return series.rolling(window).mean()
def ema(series, span): return series.ewm(span=span, adjust=False).mean()
def rsi(close, period=14):
    delta = close.diff()
    gain = delta.where(delta>0,0.0)
    loss = (-delta).where(delta<0,0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain/(avg_loss+1e-9)
    return 100 - (100/(1+rs))

def macd(close, fast=12, slow=26, signal=9):
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast-ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line-signal_line
    return macd_line, signal_line, hist

def bollinger(close, window=20, num_std=2):
    mid = sma(close, window)
    std = close.rolling(window).std()
    return mid+num_std*std, mid, mid-num_std*std

def atr(high, low, close, period=14):
    prev_close = close.shift(1)
    tr = pd.concat([(high-low),(high-prev_close).abs(),(low-prev_close).abs()],axis=1).max(axis=1)
    return tr.rolling(period).mean()

def build_features(df):
    d = df.copy()
    close, high, low, vol = d["close"], d["high"], d["low"], d["volume"].fillna(0)
    d["sma10"],d["sma20"],d["sma50"]=sma(close,10),sma(close,20),sma(close,50)
    d["ema10"],d["ema20"],d["ema50"]=ema(close,10),ema(close,20),ema(close,50)
    d["rsi14"]=rsi(close,14)
    macd_line, signal_line, hist = macd(close)
    d["macd"], d["macd_signal"], d["macd_hist"] = macd_line, signal_line, hist
    bb_u, bb_m, bb_l = bollinger(close)
    d["bb_upper"], d["bb_mid"], d["bb_lower"]=bb_u,bb_m,bb_l
    d["bb_width"]=(bb_u-bb_l)/(bb_m+1e-9)
    d["atr14"]=atr(high,low,close)
    d["ret1"],d["ret3"],d["ret5"]=close.pct_change(1),close.pct_change(3),close.pct_change(5)
    d["vol20"]=d["ret1"].rolling(20).std()
    d["vol_ma20"]=vol.rolling(20).mean()
    d["vol_ratio"]=vol/(d["vol_ma20"]+1e-9)
    return d.dropna().reset_index(drop=True)

FEATURE_COLS = ["sma10","sma20","sma50","ema10","ema20","ema50","rsi14","macd","macd_signal","macd_hist","bb_width","atr14","ret1","ret3","ret5","vol20","vol_ratio"]

def latest_feature_vector(feat_df):
    row = feat_df.iloc[-1][FEATURE_COLS].astype(float)
    return row.values.reshape(1,-1), row.to_dict()

# =========================
# Load Model
# =========================
@app.on_event("startup")
def load_model():
    global model
    model=None
    if os.path.exists(MODEL_PATH):
        try: model=joblib.load(MODEL_PATH)
        except Exception: model=None

# =========================
# Rules
# =========================
def atr_compression(df): return atr(df['high'],df['low'],df['close']).tail(14).mean()<atr(df['high'],df['low'],df['close']).tail(34).head(20).mean()*0.8
def bollinger_compression(df): return (bollinger(df['close'])[0]-bollinger(df['close'])[2]).tail(5).mean()<(bollinger(df['close'])[0]-bollinger(df['close'])[2]).tail(20).head(15).mean()*0.8
def volume_building(df): return df['volume'].tail(5).mean()>df['volume'].tail(15).head(10).mean()*1.1

def passes_rules(df, feat_df):
    reasons=[]
    last = feat_df.iloc[-1]
    close,last_rsi = float(last["close"]), float(last["rsi14"])
    if close<df["low"].tail(10).min()*0.995: reasons.append("Recent low broken")
    if last_rsi>RSI_OVERBOUGHT: reasons.append("RSI overbought")
    if not (atr_compression(df) and bollinger_compression(df)): reasons.append("No ATR/Bollinger compression")
    if not volume_building(df): reasons.append("Volume not building")
    ok = len(reasons)==0
    return ok, reasons

# =========================
# Analyze
# =========================
def analyze_one(ticker):
    t = ticker.strip().upper()
    if not t.endswith(".SR"): t+=".SR"
    df=fetch_yahoo_prices(t)
    if df is None: return {"ticker": t,"recommendation":"NO_TRADE","reason":"No data"}
    feat_df = build_features(df)
    if len(feat_df)<10: return {"ticker": t,"recommendation":"NO_TRADE","reason":"Not enough data"}
    last_close = float(feat_df["close"].iloc[-1])
    entry=last_close; tp=entry*(1+TP_PCT); sl=entry*(1-SL_PCT)
    ok,reasons=passes_rules(df,feat_df)
    ml_conf=None
    if model is not None:
        try: X,_=latest_feature_vector(feat_df); ml_conf=float(model.predict_proba(X)[0,1])
        except Exception: ml_conf=None
    if not ok:
        recommendation="NO_TRADE"; reason=" | ".join(reasons[:3]) if reasons else "Rejected by rules"
    else:
        if ml_conf is not None:
            recommendation="BUY" if ml_conf>=ML_THRESHOLD else "NO_TRADE"
            reason="Rules+ML confirmed entry" if recommendation=="BUY" else "Rules OK but ML low"
        else:
            recommendation="BUY"; reason="Rules OK (ML not available)"
    return {"ticker": t,"recommendation":recommendation,"ml_confidence":ml_conf,"entry":round(entry,4),
            "take_profit":round(tp,4),"stop_loss":round(sl,4),"last_close":round(last_close,4),"reason":reason}

# =========================
# Load tickers
# =========================
def load_tickers():
    if not os.path.exists(TICKERS_PATH): return []
    with open(TICKERS_PATH,"r") as f:
        return [line.strip().upper()+(".SR" if not line.strip().upper().endswith(".SR") else "") for line in f if line.strip()]

# =========================
# Endpoints
# =========================
@app.get("/health")
def health(): return {"status":"OK","model_loaded":bool(model)}

@app.get("/predict")
def predict(ticker:str): return analyze_one(ticker)

@app.get("/top10")
def top10(max_workers:int=TOP10_WORKERS):
    now=time.time()
    if _top10_cache["data"] and now-_top10_cache["ts"]<TOP10_CACHE_TTL_SEC:
        payload=dict(_top10_cache["data"]); payload["cached"]=True; payload["cached_age_sec"]=int(now-_top10_cache["ts"]); return payload
    tickers=load_tickers()
    if not tickers: return {"items":[],"error":"Tickers empty"}
    results=[]; errors=0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures={ex.submit(analyze_one,t):t for t in tickers}
        for fut in as_completed(futures):
            try: results.append(fut.result())
            except: errors+=1
    # Sort by BUY first, then ml_conf
    results_sorted=sorted(results,key=lambda x:(x.get("recommendation")=="BUY", x.get("ml_confidence") or 0), reverse=True)
    top_items=results_sorted[:10]
    payload={"items":top_items,"total":len(results),"errors":errors,"cached":False,"computed_at_unix":int(time.time())}
    _top10_cache["ts"]=time.time(); _top10_cache["data"]=payload
    return payload
