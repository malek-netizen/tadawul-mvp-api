import os
import time
import joblib
import requests
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

# --------------------------
MODEL_PATH = "model.joblib"
TICKERS_PATH = "tickers_sa.txt"

TP_PCT = 0.05
SL_PCT = 0.02

MIN_ROWS = 120
MIN_TRAIN_SAMPLES = 500
MAX_TICKERS = None
FEATURE_COLS = [
    "sma10", "sma20", "sma50",
    "ema10", "ema20", "ema50",
    "rsi14",
    "macd", "macd_signal", "macd_hist",
    "bb_width",
    "atr14",
    "ret1", "ret3", "ret5", "vol20",
    "vol_ratio",
]

# --------------------------
# بيانات Yahoo
def fetch_yahoo_prices(ticker: str, range_="2y", interval="1d"):
    bases = ["https://query1.finance.yahoo.com/v8/finance/chart/", "https://query2.finance.yahoo.com/v8/finance/chart/"]
    headers = {"User-Agent": "Mozilla/5.0"}
    for _ in range(3):
        for base in bases:
            try:
                url = f"{base}{ticker}"
                r = requests.get(url, params={"range": range_, "interval": interval}, headers=headers, timeout=25)
                if r.status_code != 200:
                    continue
                js = r.json()
                result = (js.get("chart", {}).get("result") or [None])[0]
                if not result:
                    continue
                quote = (result.get("indicators", {}).get("quote") or [None])[0]
                if not quote:
                    continue
                df = pd.DataFrame({
                    "open": quote.get("open", []),
                    "high": quote.get("high", []),
                    "low": quote.get("low", []),
                    "close": quote.get("close", []),
                    "volume": quote.get("volume", []),
                }).dropna(subset=["close"]).reset_index(drop=True)
                return df if len(df) > 0 else None
            except:
                continue
        time.sleep(1)
    return None

# --------------------------
# مؤشرات
def sma(series, window): return series.rolling(window).mean()
def ema(series, span): return series.ewm(span=span, adjust=False).mean()
def rsi(close, period=14):
    delta = close.diff()
    gain = delta.where(delta>0,0.0)
    loss = (-delta).where(delta<0,0.0)
    rs = gain.rolling(period).mean() / (loss.rolling(period).mean()+1e-9)
    return 100 - (100/(1+rs))
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
    d["sma10"]=sma(d["close"],10)
    d["sma20"]=sma(d["close"],20)
    d["sma50"]=sma(d["close"],50)
    d["ema10"]=ema(d["close"],10)
    d["ema20"]=ema(d["close"],20)
    d["ema50"]=ema(d["close"],50)
    d["rsi14"]=rsi(d["close"],14)
    macd_line, sig_line, hist = macd(d["close"])
    d["macd"]=macd_line; d["macd_signal"]=sig_line; d["macd_hist"]=hist
    bb_u, bb_m, bb_l = bollinger(d["close"])
    d["bb_width"]=(bb_u-bb_l)/(bb_m+1e-9)
    d["atr14"]=atr(d["high"], d["low"], d["close"])
    d["ret1"]=d["close"].pct_change(1)
    d["ret3"]=d["close"].pct_change(3)
    d["ret5"]=d["close"].pct_change(5)
    d["vol20"]=d["ret1"].rolling(20).std()
    d["vol_ma20"]=d["volume"].rolling(20).mean()
    d["vol_ratio"]=d["volume"]/(d["vol_ma20"]+1e-9)
    return d.dropna().reset_index(drop=True)

def make_labels(df, feat, tp_pct=TP_PCT, sl_pct=SL_PCT):
    highs = feat["high"].values
    lows = feat["low"].values
    closes = feat["close"].values
    labels = np.full(len(feat), np.nan)
    for i in range(len(feat)-2):
        entry=closes[i]; tp=entry*(1+tp_pct); sl=entry*(1-sl_pct)
        hit=None
        for j in range(i+1,len(feat)):
            if lows[j]<=sl: hit=0; break
            if highs[j]>=tp: hit=1; break
        labels[i]=hit if hit is not None else np.nan
    return labels

def load_tickers(path=TICKERS_PATH):
    if not os.path.exists(path): return []
    out=[]
    with open(path,"r",encoding="utf-8") as f:
        for l in f: t=l.strip().upper(); out.append(t if t.endswith(".SR") else t+".SR")
    seen=set(); uniq=[]
    for x in out:
        if x not in seen: seen.add(x); uniq.append(x)
    return uniq

# --------------------------
# MAIN
def main():
    tickers = load_tickers()
    if MAX_TICKERS: tickers=tickers[:MAX_TICKERS]
    rows=[]; used=0
    for t in tickers:
        df=fetch_yahoo_prices(t)
        if df is None or len(df)<MIN_ROWS: continue
        feat=build_features(df)
        if len(feat)<100: continue
        y=make_labels(df, feat)
        X=feat[FEATURE_COLS].copy(); X["label"]=y; X=X.dropna(subset=["label"])
        if len(X)<30: continue
        X["ticker"]=t; rows.append(X); used+=1
        print(f"[OK] {t} samples={len(X)}"); time.sleep(0.2)

    if not rows: raise RuntimeError("No training data collected.")
    data=pd.concat(rows, ignore_index=True)
    print("Tickers used:", used, "Total samples:", len(data))
    if len(data)<MIN_TRAIN_SAMPLES: raise RuntimeError(f"Need >= {MIN_TRAIN_SAMPLES}, got {len(data)}")

    y=data["label"].astype(int).values
    X=data[FEATURE_COLS].astype(float).values

    # TRAIN
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42,stratify=y)
    clf=RandomForestClassifier(n_estimators=400,max_depth=8,min_samples_leaf=20,class_weight="balanced",n_jobs=-1,random_state=42)
    clf.fit(X_train, y_train)
    proba=clf.predict_proba(X_test)[:,1]
    print("AUC:", roc_auc_score(y_test, proba))
    joblib.dump(clf, MODEL_PATH)
    print("Saved:", MODEL_PATH)

    # تجربة على آخر بيانات كل سهم
    print("\n=== Sample ML Confidence ===")
    for t in tickers[:5]:  # نجرب أول 5 أسهم
        df=fetch_yahoo_prices(t)
        feat=build_features(df)
        last_feat=feat[FEATURE_COLS].iloc[-1].values.reshape(1,-1)
        ml_conf=clf.predict_proba(last_feat)[0,1]
        print(t, "ML confidence:", round(ml_conf,3))

if __name__=="__main__":
    main()
