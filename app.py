from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import requests
import pandas as pd
import numpy as np
import time
import joblib
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# =========================
# Tadawul MVP API (Rules v1.2)
# Top10 محسّن حسب الخوارزمية + ML
# =========================

app = FastAPI(title="Tadawul MVP API", version="2.2.1-top10-enhanced")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "model.joblib"
TICKERS_PATH = "tickers_sa.txt"

# إعدادات ثابتة للصفقة
TP_PCT = 0.05
SL_PCT = 0.02

# حدود ومحددات قواعد الدخول
RSI_MIN = 40
RSI_MAX = 65
RSI_OVERBOUGHT = 70

MAX_ABOVE_EMA20 = 0.06
MAX_ABOVE_EMA50 = 0.08

SPIKE_RET1 = 0.08
CONSOL_DAYS = 4
CONSOL_RANGE_MAX = 0.06
LOOKBACK_LOW_DAYS = 10
MIN_TREND_SCORE = 2
ML_THRESHOLD = 0.60
TOP10_WORKERS = 6

model = None

# =========================
# Cache للأسعار
# =========================
_prices_cache = {}
CACHE_TTL_SEC = 600  # 10 دقائق

def _cache_get(key):
    v = _prices_cache.get(key)
    if not v: return None
    if time.time() - v["ts"] > CACHE_TTL_SEC:
        _prices_cache.pop(key, None)
        return None
    return v["df"]

def _cache_set(key, df):
    _prices_cache[key] = {"ts": time.time(), "df": df}

# =========================
# Cache Top10
# =========================
_top10_cache = {"ts": 0, "data": None}
TOP10_CACHE_TTL_SEC = 600

# =========================
# 1) جلب الأسعار من Yahoo
# =========================
def fetch_yahoo_prices(ticker: str, range_: str = "1y", interval: str = "1d", min_rows: int = 80):
    key = (ticker, range_, interval)
    cached = _cache_get(key)
    if cached is not None:
        return cached

    bases = ["https://query1.finance.yahoo.com/v8/finance/chart/",
             "https://query2.finance.yahoo.com/v8/finance/chart/"]
    headers = {"User-Agent": "Mozilla/5.0"}

    for _ in range(3):
        for base in bases:
            try:
                url = f"{base}{ticker}"
                r = requests.get(url, params={"range": range_, "interval": interval}, headers=headers, timeout=20)
                if r.status_code != 200: continue
                js = r.json()
                chart = js.get("chart", {})
                if chart.get("error"): continue
                result = (chart.get("result") or [None])[0]
                if not result: continue
                quote = (result.get("indicators", {}).get("quote") or [None])[0]
                if not quote: continue

                df = pd.DataFrame({
                    "open": quote.get("open", []),
                    "high": quote.get("high", []),
                    "low": quote.get("low", []),
                    "close": quote.get("close", []),
                    "volume": quote.get("volume", []),
                })
                df = df.dropna(subset=["close"]).reset_index(drop=True)
                if len(df) >= min_rows:
                    _cache_set(key, df)
                    return df
            except Exception: continue
        time.sleep(1)
    return None

# =========================
# 2) مؤشرات فنية
# =========================
def sma(series, window): return series.rolling(window).mean()
def ema(series, span): return series.ewm(span=span, adjust=False).mean()
def rsi(close, period=14):
    delta = close.diff()
    gain = delta.where(delta>0, 0.0)
    loss = (-delta).where(delta<0, 0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))

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
    upper = mid + num_std*std
    lower = mid - num_std*std
    return upper, mid, lower

def atr(high, low, close, period=14):
    prev_close = close.shift(1)
    tr = pd.concat([(high-low), (high-prev_close).abs(), (low-prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    close = d["close"]; high=d["high"]; low=d["low"]; vol=d["volume"].fillna(0)
    d["sma10"], d["sma20"], d["sma50"] = sma(close,10), sma(close,20), sma(close,50)
    d["ema10"], d["ema20"], d["ema50"] = ema(close,10), ema(close,20), ema(close,50)
    d["rsi14"] = rsi(close,14)
    macd_line, signal_line, hist = macd(close)
    d["macd"], d["macd_signal"], d["macd_hist"] = macd_line, signal_line, hist
    bb_u, bb_m, bb_l = bollinger(close)
    d["bb_upper"], d["bb_mid"], d["bb_lower"] = bb_u, bb_m, bb_l
    d["bb_width"] = (bb_u - bb_l)/(bb_m + 1e-9)
    d["atr14"] = atr(high, low, close)
    d["ret1"], d["ret3"], d["ret5"] = close.pct_change(1), close.pct_change(3), close.pct_change(5)
    d["vol20"] = d["ret1"].rolling(20).std()
    d["vol_ma20"] = vol.rolling(20).mean()
    d["vol_ratio"] = vol/(d["vol_ma20"] + 1e-9)
    d = d.dropna().reset_index(drop=True)
    return d

FEATURE_COLS = ["sma10","sma20","sma50","ema10","ema20","ema50","rsi14",
                "macd","macd_signal","macd_hist","bb_width","atr14",
                "ret1","ret3","ret5","vol20","vol_ratio"]

def latest_feature_vector(feat_df: pd.DataFrame):
    row = feat_df.iloc[-1][FEATURE_COLS].astype(float)
    return row.values.reshape(1,-1), row.to_dict()

# =========================
# 3) تحميل النموذج
# =========================
@app.on_event("startup")
def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        try: model = joblib.load(MODEL_PATH)
        except: model=None

# =========================
# 4) قواعد الدخول وتحليل سهم واحد
# =========================
def is_consolidating(feat_df: pd.DataFrame, days: int = CONSOL_DAYS) -> bool:
    if len(feat_df) < days+5: return False
    last = feat_df.tail(days)
    hi, lo, mid = float(last["high"].max()), float(last["low"].min()), float(last["close"].mean())
    return (hi-lo)/mid <= CONSOL_RANGE_MAX

def trend_score(feat_df: pd.DataFrame) -> int:
    last = feat_df.iloc[-1]; score=0
    close=float(last["close"]); ema20=float(last["ema20"]); ema50=float(last["ema50"])
    r5=float(last["ret5"]); mh=float(last["macd_hist"])
    score += 1 if close >= ema20 else 0
    score += 1 if ema20 >= ema50 else 0
    score += 1 if r5>-0.03 else 0
    score += 1 if mh>=-0.001 else 0
    return score

def broke_recent_low(df_ohlc: pd.DataFrame, lookback:int=LOOKBACK_LOW_DAYS) -> bool:
    if len(df_ohlc)<lookback+5: return False
    recent_low = float(df_ohlc.tail(lookback)["low"].min())
    return float(df_ohlc["close"].iloc[-1]) < recent_low*0.995

def spike_without_base(feat_df: pd.DataFrame) -> bool:
    last = feat_df.iloc[-1]; return float(last["ret1"]) >= SPIKE_RET1 and (not is_consolidating(feat_df,CONSOL_DAYS))

def risk_check_stop(df_ohlc: pd.DataFrame, stop: float) -> bool:
    if len(df_ohlc)<10: return False
    recent_low=float(df_ohlc.tail(5)["low"].min())
    return not (recent_low<stop)

def passes_rules(df_ohlc: pd.DataFrame, feat_df: pd.DataFrame):
    reasons=[]; last=feat_df.iloc[-1]
    close=float(last["close"]); ema20=float(last["ema20"]); ema50=float(last["ema50"]); rsi14=float(last["rsi14"])
    if broke_recent_low(df_ohlc): reasons.append("Rejected: recent low broken")
    if trend_score(feat_df)<MIN_TREND_SCORE: reasons.append("Rejected: weak trend")
    if spike_without_base(feat_df): reasons.append("Rejected: 1-day spike without base")
    if rsi14>RSI_OVERBOUGHT: reasons.append("Rejected: RSI overbought")
    if not (RSI_MIN<=rsi14<=RSI_MAX) and not (rsi14>=35 and is_consolidating(feat_df,CONSOL_DAYS)):
        reasons.append("Rejected: RSI not in safe band/no consolidation exception")
    if ema20>0 and (close-ema20)/ema20>MAX_ABOVE_EMA20: reasons.append("Rejected: price > EMA20 limit")
    if ema50>0 and (close-ema50)/ema50>MAX_ABOVE_EMA50: reasons.append("Rejected: price > EMA50 limit")
    h=feat_df["macd_hist"].tail(4).values
    if len(h)==4 and not (h[-1]>=h[-2] or h[-1]>=0): reasons.append("Rejected: MACD not improving")
    if not is_consolidating(feat_df,CONSOL_DAYS) and trend_score(feat_df)<3: reasons.append("Rejected: no base before entry")
    return (len(reasons)==0), reasons

def rules_score(feat_df: pd.DataFrame) -> int:
    if len(feat_df)<5: return 30
    last=feat_df.iloc[-1]; score=0
    score += 15 if float(last["close"])>=float(last["ema20"]) else 0
    score += 15 if float(last["ema20"])>=float(last["ema50"]) else 0
    r=float(last["rsi14"])
    score += 20 if RSI_MIN<=r<=RSI_MAX else (10 if 35<=r<RSI_MIN else 0)
    score += 15 if float(last["macd_hist"])>=0 else 5
    score += 20 if is_consolidating(feat_df,CONSOL_DAYS) else 5
    close=float(last["close"]); ema20=float(last["ema20"])
    if ema20>0 and (close-ema20)/ema20>MAX_ABOVE_EMA20: score-=10
    return max(0,min(100,score))

def analyze_one(ticker: str):
    t=(ticker or "").strip().upper(); t+= ".SR" if not t.endswith(".SR") else ""
    df=fetch_yahoo_prices(t); 
    if df is None or len(df)<80:
        return {"ticker": t, "recommendation":"NO_TRADE", "status":"NO_DATA", "confidence_pct":0,
                "ml_confidence":None,"rules_score":0,"entry":None,"take_profit":None,"stop_loss":None,
                "reason":"No price data or insufficient rows","last_close":None}
    feat_df=build_features(df)
    if len(feat_df)<10:
        return {"ticker": t, "recommendation":"NO_TRADE", "status":"NO_DATA", "confidence_pct":0,
                "ml_confidence":None,"rules_score":0,"entry":None,"take_profit":None,"stop_loss":None,
                "reason":"Not enough data after feature engineering","last_close":None}

    last_close=float(feat_df["close"].iloc[-1]); entry=last_close
    take_profit=entry*(1+TP_PCT); stop_loss=entry*(1-SL_PCT)
    ok,reasons=passes_rules(df,feat_df)
    if ok and not risk_check_stop(df,stop_loss): ok=False; reasons.append("Rejected: -2% stop too tight")
    r_score=rules_score(feat_df)

    ml_conf=None
    if model is not None:
        try: X,_=latest_feature_vector(feat_df); ml_conf=float(model.predict_proba(X)[0,1])
        except: ml_conf=None

    # تقييم فئات
    if not ok:
        recommendation="NO_TRADE"; status="REJECTED"; conf_pct=int(round(r_score))
        reason=" | ".join(reasons[:3]) if reasons else "Rejected by rules."
    else:
        if ml_conf is not None:
            recommendation="BUY" if ml_conf>=ML_THRESHOLD else "FOLLOW"
            status="APPROVED" if recommendation=="BUY" else "APPROVED_BUT_LOW_ML"
            conf_pct=int(round(ml_conf*100))
            reason="Rules+ML confirmed" if recommendation=="BUY" else "Rules OK but ML below threshold"
        else:
            recommendation="BUY" if r_score>=70 else "FOLLOW"
            status="APPROVED_RULES_ONLY" if recommendation=="BUY" else "APPROVED_BUT_LOW_SCORE"
            conf_pct=int(round(r_score))
            reason="Rules-based decision (ML not available)"

    return {"ticker":t,"recommendation":recommendation,"status":status,"confidence_pct":conf_pct,
            "ml_confidence":ml_conf,"rules_score":int(r_score),"entry":round(entry,4),
            "take_profit":round(take_profit,4),"stop_loss":round(stop_loss,4),
            "reason":reason,"last_close":round(last_close,4)}

# =========================
# 5) تحميل قائمة الأسهم
# =========================
def load_tickers_from_file(path: str):
    if not os.path.exists(path): return []
    items=[line.strip().upper() for line in open(path,"r",encoding="utf-8") if line.strip()]
    out=[]; seen=set()
    for x in items:
        x+= ".SR" if not x.endswith(".SR") else ""
        if x not in seen: seen.add(x); out.append(x)
    return out

# =========================
# 6) Endpoints
# =========================
@app.get("/health")
def health():
    return {"status":"OK","model_loaded":bool(model),"rules_version":"v1.2","tp_pct":TP_PCT,
            "sl_pct":SL_PCT,"tickers_file":TICKERS_PATH,"ml_threshold":ML_THRESHOLD,
            "top10_workers":TOP10_WORKERS,"prices_cache_ttl_sec":CACHE_TTL_SEC,
            "top10_cache_ttl_sec":TOP10_CACHE_TTL_SEC}

@app.get("/predict")
def predict(ticker:str): return analyze_one(ticker)

@app.get("/debug_prices")
def debug_prices(ticker:str):
    t=ticker.strip().upper(); t+= ".SR" if not t.endswith(".SR") else ""
    df=fetch_yahoo_prices(t)
    if df is None: return {"ticker":t,"ok":False,"rows":0}
    return {"ticker":t,"ok":True,"rows":int(len(df)),"last_close":float(df["close"].iloc[-1])}

@app.get("/check_tickers")
def check_tickers():
    path=os.path.abspath(TICKERS_PATH)
    tickers=load_tickers_from_file(TICKERS_PATH)
    return {"exists":os.path.exists(TICKERS_PATH),"path":path,"count":len(tickers),"sample":tickers[:10]}

@app.get("/top10")
def top10(max_workers: int = TOP10_WORKERS):
    now=time.time()
    if _top10_cache["data"] and (now-_top10_cache["ts"]<TOP10_CACHE_TTL_SEC):
        cached_payload=dict(_top10_cache["data"]); cached_payload["cached"]=True
        cached_payload["cached_age_sec"]=int(now-_top10_cache["ts"])
        return cached_payload

    tickers=load_tickers_from_file(TICKERS_PATH)
    if not tickers: return {"items":[],"error":f"Tickers file not found or empty: {TICKERS_PATH}","cached":False}

    results=[]; errors=0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures={ex.submit(analyze_one,t):t for t in tickers}
        for fut in as_completed(futures):
            try: results.append(fut.result())
            except: errors+=1

    def score(item):
        rec=item.get("recommendation","NO_TRADE")
        confp=int(item.get("confidence_pct",0) or 0)
        mlc=float(item.get("ml_confidence") or 0)
        rank = 3 if rec=="BUY" else 2 if rec=="FOLLOW" else 1
        return (-rank,-confp,-mlc)
    
    results_sorted=sorted(results,key=score)
    top10_results=results_sorted[:10]

    payload={"items":top10_results,"cached":False,"cached_age_sec":0,"errors":errors,"total_checked":len(results)}
    _top10_cache["data"]=payload; _top10_cache["ts"]=time.time()
    return payload
