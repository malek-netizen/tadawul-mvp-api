import os, time, joblib, requests, pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# =========================
# إعدادات عامة لتوافق مع app.py
# =========================
MODEL_PATH = "model.joblib"
TICKERS_PATH = "tickers_sa.txt"
FEATURE_COLS = [
    "sma10","sma20","sma50","ema10","ema20","ema50",
    "rsi14","macd","macd_signal","macd_hist","bb_width",
    "atr14","ret1","ret3","ret5","vol20","vol_ratio"
]
TP_PCT = 0.05
SL_PCT = 0.02
MIN_ROWS = 80        # نفس app.py
MIN_TRAIN_SAMPLES = 500

# =========================
# الدوال الفنية (مطابقة app.py)
# =========================
def sma(series, window): return series.rolling(window).mean()
def ema(series, span): return series.ewm(span=span, adjust=False).mean()
def rsi(close, period=14):
    delta = close.diff()
    gain = delta.where(delta>0,0.0)
    loss = (-delta).where(delta<0,0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100/(1+rs))
def macd(close, fast=12, slow=26, signal=9):
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    sig_line = ema(macd_line, signal)
    hist = macd_line - sig_line
    return macd_line, sig_line, hist
def bollinger(close, window=20, num_std=2):
    mid = sma(close, window)
    std = close.rolling(window).std()
    upper = mid + num_std*std
    lower = mid - num_std*std
    return upper, mid, lower
def atr(high, low, close, period=14):
    prev = close.shift(1)
    tr = pd.concat([high-low, (high-prev).abs(), (low-prev).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()
def build_features(df):
    d = df.copy()
    close=df["close"]; high=df["high"]; low=df["low"]; vol=df["volume"].fillna(0)

    d["sma10"]=sma(close,10); d["sma20"]=sma(close,20); d["sma50"]=sma(close,50)
    d["ema10"]=ema(close,10); d["ema20"]=ema(close,20); d["ema50"]=ema(close,50)
    d["rsi14"]=rsi(close,14)
    macd_line, sig_line, hist = macd(close)
    d["macd"]=macd_line; d["macd_signal"]=sig_line; d["macd_hist"]=hist
    bb_u, bb_m, bb_l = bollinger(close)
    d["bb_width"] = (bb_u - bb_l)/(bb_m+1e-9)
    d["atr14"] = atr(high, low, close)
    d["ret1"] = close.pct_change(1); d["ret3"]=close.pct_change(3); d["ret5"]=close.pct_change(5)
    d["vol20"] = d["ret1"].rolling(20).std()
    d["vol_ma20"] = vol.rolling(20).mean()
    d["vol_ratio"] = vol/(d["vol_ma20"]+1e-9)
    return d.dropna().reset_index(drop=True)

# =========================
# جلب الأسعار (مطابق app.py)
# =========================
def fetch_yahoo_prices(ticker: str, range_="1y", interval="1d", min_rows=MIN_ROWS):
    bases = ["https://query1.finance.yahoo.com/v8/finance/chart/",
             "https://query2.finance.yahoo.com/v8/finance/chart/"]
    headers = {"User-Agent":"Mozilla/5.0"}
    for _ in range(3):
        for base in bases:
            try:
                url = f"{base}{ticker}"
                r = requests.get(url, params={"range":range_, "interval":interval}, headers=headers, timeout=25)
                if r.status_code != 200: continue
                js = (r.json().get("chart",{}).get("result") or [None])[0]
                if not js: continue
                quote = (js.get("indicators",{}).get("quote") or [None])[0]
                if not quote: continue
                df = pd.DataFrame({
                    "open": quote.get("open",[]),
                    "high": quote.get("high",[]),
                    "low": quote.get("low",[]),
                    "close": quote.get("close",[]),
                    "volume": quote.get("volume",[]),
                }).dropna(subset=["close"])
                if len(df) >= min_rows:
                    return df
            except: continue
        time.sleep(1)
    return None

# =========================
# Labels للتدريب (Buy=1, NoTrade=0)
# =========================
def make_labels(df, feat, tp_pct=TP_PCT, sl_pct=SL_PCT):
    highs, lows, closes = feat["high"].values, feat["low"].values, feat["close"].values
    labels = np.full(len(feat), np.nan)
    for i in range(len(feat)-2):
        entry = closes[i]
        tp = entry*(1+tp_pct)
        sl = entry*(1-sl_pct)
        hit=None
        for j in range(i+1, len(feat)):
            if lows[j] <= sl: hit=0; break
            if highs[j] >= tp: hit=1; break
        labels[i] = hit if hit is not None else np.nan
    return labels

# =========================
# تحميل قائمة الأسهم
# =========================
def load_tickers(path=TICKERS_PATH):
    if not os.path.exists(path): return []
    tickers = [l.strip().upper() for l in open(path,"r",encoding="utf-8") if l.strip()]
    tickers = [t if t.endswith(".SR") else t+".SR" for t in tickers]
    return list(dict.fromkeys(tickers))

# =========================
# التدريب
# =========================
def main():
    tickers = load_tickers()
    rows = []

    for t in tickers:
        df = fetch_yahoo_prices(t)
        if df is None or len(df)<MIN_ROWS: continue
        feat = build_features(df)
        if len(feat) < 100: continue
        y = make_labels(df, feat)
        X = feat[FEATURE_COLS].copy()
        X["label"] = y
        X = X.dropna(subset=["label"])
        if len(X)<30: continue
        X["ticker"] = t
        rows.append(X)
        print(f"[OK] {t} samples={len(X)}")
        time.sleep(0.1)

    if not rows:
        raise RuntimeError("No training data collected.")

    data = pd.concat(rows, ignore_index=True)
    if len(data) < MIN_TRAIN_SAMPLES:
        raise RuntimeError(f"Need >= {MIN_TRAIN_SAMPLES} samples, got {len(data)}")

    y = data["label"].astype(int).values
    X = data[FEATURE_COLS].astype(float).values

    # TRAIN
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    clf = RandomForestClassifier(
        n_estimators=400, max_depth=8, min_samples_leaf=20,
        class_weight="balanced", n_jobs=-1, random_state=42
    )
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)[:,1]
    print("AUC:", roc_auc_score(y_test, proba))
    joblib.dump(clf, MODEL_PATH)
    print("Saved:", MODEL_PATH)

if __name__=="__main__":
    main()
