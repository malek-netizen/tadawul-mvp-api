import os
import time
import joblib
import requests
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

MODEL_PATH = "model.joblib"
TICKERS_PATH = "tickers_sa.txt"

# ----------------------------
# Yahoo fetch (Chart API)
# ----------------------------
def fetch_yahoo_prices(ticker: str, range_: str = "5y", interval: str = "1d", min_rows: int = 220):
    bases = [
        "https://query1.finance.yahoo.com/v8/finance/chart/",
        "https://query2.finance.yahoo.com/v8/finance/chart/",
    ]
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
        "Accept-Language": "en-US,en;q=0.9",
    }

    for attempt in range(3):
        for base in bases:
            try:
                url = f"{base}{ticker}"
                r = requests.get(
                    url,
                    params={"range": range_, "interval": interval},
                    headers=headers,
                    timeout=25,
                )
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

                df = pd.DataFrame(
                    {
                        "open": quote.get("open", []),
                        "high": quote.get("high", []),
                        "low": quote.get("low", []),
                        "close": quote.get("close", []),
                        "volume": quote.get("volume", []),
                    }
                ).dropna(subset=["close"]).reset_index(drop=True)

                if len(df) >= min_rows:
                    return df

            except Exception:
                continue

        time.sleep(1)

    return None

# ----------------------------
# Indicators
# ----------------------------
def sma(s, w): return s.rolling(w).mean()
def ema(s, span): return s.ewm(span=span, adjust=False).mean()

def rsi(close, period=14):
    d = close.diff()
    gain = d.where(d > 0, 0.0)
    loss = (-d).where(d < 0, 0.0)
    avg_g = gain.rolling(period).mean()
    avg_l = loss.rolling(period).mean()
    rs = avg_g / (avg_l + 1e-9)
    return 100 - (100 / (1 + rs))

def macd(close, fast=12, slow=26, signal=9):
    f = ema(close, fast)
    s = ema(close, slow)
    m = f - s
    sig = ema(m, signal)
    hist = m - sig
    return m, sig, hist

def bollinger(close, w=20, n=2):
    mid = sma(close, w)
    std = close.rolling(w).std()
    up = mid + n * std
    lo = mid - n * std
    return up, mid, lo

def atr(high, low, close, period=14):
    prev = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev).abs(), (low - prev).abs()],
        axis=1
    ).max(axis=1)
    return tr.rolling(period).mean()

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    c = d["close"]
    h = d["high"]
    l = d["low"]
    v = d["volume"].fillna(0)

    d["sma10"] = sma(c, 10)
    d["sma20"] = sma(c, 20)
    d["sma50"] = sma(c, 50)

    d["ema10"] = ema(c, 10)
    d["ema20"] = ema(c, 20)
    d["ema50"] = ema(c, 50)

    d["rsi14"] = rsi(c, 14)

    m, sig, hist = macd(c)
    d["macd"] = m
    d["macd_signal"] = sig
    d["macd_hist"] = hist

    up, mid, lo = bollinger(c, 20, 2)
    d["bb_upper"] = up
    d["bb_mid"] = mid
    d["bb_lower"] = lo
    d["bb_width"] = (up - lo) / (mid + 1e-9)

    d["atr14"] = atr(h, l, c, 14)

    d["ret1"] = c.pct_change(1)
    d["ret5"] = c.pct_change(5)
    d["vol20"] = d["ret1"].rolling(20).std()

    d["vol_ma20"] = v.rolling(20).mean()
    d["vol_ratio"] = v / (d["vol_ma20"] + 1e-9)

    return d.dropna().reset_index(drop=True)

FEATURE_COLS = [
    "sma10","sma20","sma50",
    "ema10","ema20","ema50",
    "rsi14",
    "macd","macd_signal","macd_hist",
    "bb_width",
    "atr14",
    "ret1","ret5","vol20",
    "vol_ratio",
]

# ----------------------------
# Label: TP before SL within horizon
# TP = +5%, SL = -2%, horizon=30
# using daily HIGH/LOW (realistic)
# ----------------------------
def make_label(df_feat: pd.DataFrame, horizon=30, tp=0.05, sl=0.02):
    closes = df_feat["close"].values
    highs = df_feat["high"].values
    lows  = df_feat["low"].values

    y = np.zeros(len(df_feat), dtype=int)

    for i in range(len(df_feat) - horizon):
        entry = closes[i]
        tp_level = entry * (1 + tp)
        sl_level = entry * (1 - sl)

        hit_tp = None
        hit_sl = None

        for j in range(1, horizon+1):
            if highs[i+j] >= tp_level:
                hit_tp = j
                break

        for j in range(1, horizon+1):
            if lows[i+j] <= sl_level:
                hit_sl = j
                break

        # success if TP happens and either SL never happens OR TP earlier than SL
        if hit_tp is not None and (hit_sl is None or hit_tp < hit_sl):
            y[i] = 1
        else:
            y[i] = 0

    # last horizon rows have no future window -> drop later
    return y

def load_tickers(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}")
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip().upper()
            if not t:
                continue
            if not t.endswith(".SR"):
                t += ".SR"
            out.append(t)
    # unique keep order
    seen=set()
    uniq=[]
    for t in out:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq

def main():
    tickers = load_tickers(TICKERS_PATH)
    print("Tickers:", len(tickers))

    X_all = []
    y_all = []

    for k, t in enumerate(tickers, 1):
        df = fetch_yahoo_prices(t, range_="5y", interval="1d", min_rows=260)
        if df is None:
            print("skip (no data):", t)
            continue

        feat = build_features(df)
        if len(feat) < 260:
            print("skip (short feat):", t, len(feat))
            continue

        y = make_label(feat, horizon=30, tp=0.05, sl=0.02)

        # remove last 30 rows (no label)
        feat2 = feat.iloc[:-30].copy()
        y2 = y[:-30]

        X_all.append(feat2[FEATURE_COLS].values)
        y_all.append(y2)

        if k % 20 == 0:
            print(f"processed {k}/{len(tickers)}")

    if not X_all:
        raise RuntimeError("No training data built. Check tickers and Yahoo access.")

    X = np.vstack(X_all)
    y = np.concatenate(y_all)

    print("Dataset shape:", X.shape, "Pos rate:", y.mean().round(4))

    # time-safe split (simple): last 20% as test
    n = len(X)
    split = int(n * 0.8)

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            max_iter=4000,
            class_weight="balanced",
            solver="lbfgs"
        ))
    ])

    clf.fit(X_train, y_train)

    p = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, p) if len(np.unique(y_test)) > 1 else float("nan")
    print("AUC:", auc)

    payload = {
        "pipeline": clf,
        "feature_cols": FEATURE_COLS,
        "meta": {
            "tp": 0.05,
            "sl": 0.02,
            "horizon": 30,
            "threshold": 0.55,   # نخففها شوي ليظهر BUY منطقي
            "auc": float(auc) if auc == auc else None,
            "built_from": len(tickers)
        }
    }

    joblib.dump(payload, MODEL_PATH)
    print("Saved", MODEL_PATH)

if __name__ == "__main__":
    main()
