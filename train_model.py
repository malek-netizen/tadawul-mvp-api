import os
import time
import joblib
import requests
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# إعدادات
# -----------------------------
MODEL_PATH = "model.joblib"
TICKERS_PATH = "tickers_sa.txt"

TP_PCT = 0.05
SL_PCT = 0.02

MIN_ROWS = 120
MIN_TRAIN_SAMPLES = 500
MAX_TICKERS = None  # None = كل الأسهم أو ضع رقم للتجربة

# -----------------------------
# تحميل أسعار Yahoo
# -----------------------------
def fetch_yahoo_prices(ticker: str, range_: str = "2y", interval: str = "1d"):
    bases = [
        "https://query1.finance.yahoo.com/v8/finance/chart/",
        "https://query2.finance.yahoo.com/v8/finance/chart/",
    ]
    headers = {"User-Agent": "Mozilla/5.0"}

    for _ in range(3):
        for base in bases:
            try:
                url = f"{base}{ticker}"
                r = requests.get(url, params={"range": range_, "interval": interval}, headers=headers, timeout=25)
                if r.status_code != 200:
                    continue
                js = r.json()
                chart = js.get("chart", {})
                if chart.get("error"):
                    continue
                result = (chart.get("result") or [None])[0]
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
            except Exception:
                continue
        time.sleep(1)
    return None

# -----------------------------
# مؤشرات فنية
# -----------------------------
def sma(series, window):
    return series.rolling(window).mean()

def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def rsi(close, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    rs = gain.rolling(period).mean() / (loss.rolling(period).mean() + 1e-9)
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
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower

def atr(high, low, close, period=14):
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    close = d["close"]
    high = d["high"]
    low = d["low"]
    vol = d["volume"].fillna(0)

    # مؤشرات
    d["sma10"], d["sma20"], d["sma50"] = sma(close, 10), sma(close, 20), sma(close, 50)
    d["ema10"], d["ema20"], d["ema50"] = ema(close, 10), ema(close, 20), ema(close, 50)
    d["rsi14"] = rsi(close, 14)
    macd_line, signal_line, hist = macd(close)
    d["macd"], d["macd_signal"], d["macd_hist"] = macd_line, signal_line, hist
    bb_u, bb_m, bb_l = bollinger(close)
    d["bb_width"] = (bb_u - bb_l) / (bb_m + 1e-9)
    d["atr14"] = atr(high, low, close)

    # العوائد والحجم
    d["ret1"], d["ret3"], d["ret5"] = close.pct_change(1), close.pct_change(3), close.pct_change(5)
    d["vol20"] = d["ret1"].rolling(20).std()
    d["vol_ma20"] = vol.rolling(20).mean()
    d["vol_ratio"] = vol / (d["vol_ma20"] + 1e-9)

    return d.dropna().reset_index(drop=True)

FEATURE_COLS = [
    "sma10","sma20","sma50",
    "ema10","ema20","ema50",
    "rsi14",
    "macd","macd_signal","macd_hist",
    "bb_width",
    "atr14",
    "ret1","ret3","ret5","vol20","vol_ratio"
]

# -----------------------------
# عمل الـLabels
# -----------------------------
def make_labels(df_ohlc: pd.DataFrame, feat_df: pd.DataFrame, tp_pct=TP_PCT, sl_pct=SL_PCT):
    highs, lows, closes = feat_df["high"].values, feat_df["low"].values, feat_df["close"].values
    labels = np.full(len(feat_df), np.nan, dtype=float)
    for i in range(len(feat_df) - 2):
        entry = float(closes[i])
        tp = entry*(1+tp_pct)
        sl = entry*(1-sl_pct)
        hit = None
        for j in range(i+1, len(feat_df)):
            if lows[j] <= sl:
                hit = 0
                break
            if highs[j] >= tp:
                hit = 1
                break
        labels[i] = hit if hit is not None else np.nan
    return labels

# -----------------------------
# تحميل قائمة الأسهم
# -----------------------------
def load_tickers(path: str):
    if not os.path.exists(path): return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip().upper()
            if not t: continue
            if not t.endswith(".SR"): t += ".SR"
            out.append(t)
    # unique
    seen = set(); uniq = []
    for x in out:
        if x not in seen: seen.add(x); uniq.append(x)
    return uniq

# -----------------------------
# التدريب
# -----------------------------
def main():
    tickers = load_tickers(TICKERS_PATH)
    if MAX_TICKERS is not None:
        tickers = tickers[:MAX_TICKERS]

    if not tickers:
        raise RuntimeError(f"No tickers found in {TICKERS_PATH}")

    rows, used = [], 0
    for t in tickers:
        df = fetch_yahoo_prices(t)
        if df is None or len(df) < MIN_ROWS: continue
        feat = build_features(df)
        if len(feat) < 100: continue
        y = make_labels(df, feat)
        X = feat[FEATURE_COLS].copy(); X["label"]=y
        X = X.dropna(subset=["label"]).reset_index(drop=True)
        if len(X)<30: continue
        X["ticker"]=t
        rows.append(X)
        used += 1
        print(f"[OK] {t} samples={len(X)}")
        time.sleep(0.2)

    if not rows:
        raise RuntimeError("No training data collected")

    data = pd.concat(rows, ignore_index=True)
    print("Tickers used:", used, "Total samples:", len(data))
    if len(data) < MIN_TRAIN_SAMPLES:
        raise RuntimeError(f"Not enough samples: {len(data)} < {MIN_TRAIN_SAMPLES}")

    y = data["label"].astype(int).values
    X = data[FEATURE_COLS].astype(float).values

    # تقسيم Train/Test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=400, max_depth=8, min_samples_leaf=20,
        random_state=42, n_jobs=-1, class_weight="balanced"
    )
    clf.fit(X_train, y_train)

    proba = clf.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, proba)
    print("AUC:", auc)

    joblib.dump(clf, MODEL_PATH)
    print("Saved:", MODEL_PATH)

if __name__=="__main__":
    main()
