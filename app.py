import os, time, requests, pandas as pd, numpy as np

# --- إعدادات
TICKERS_PATH = "tickers_sa.txt"
MIN_ROWS = 120

# --- دوال فنية
def sma(series, window): return series.rolling(window).mean()
def ema(series, span): return series.ewm(span=span, adjust=False).mean()
def rsi(close, period=14):
    delta = close.diff()
    gain = delta.where(delta>0,0.0)
    loss = (-delta).where(delta<0,0.0)
    rs = gain.rolling(period).mean() / (loss.rolling(period).mean() + 1e-9)
    return 100 - 100 / (1 + rs)
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

# --- بناء المؤشرات
def build_features(df):
    d = df.copy()
    close, high, low, vol = d["close"], d["high"], d["low"], d["volume"]
    d["sma10"], d["sma20"], d["sma50"] = sma(close,10), sma(close,20), sma(close,50)
    d["ema10"], d["ema20"], d["ema50"] = ema(close,10), ema(close,20), ema(close,50)
    d["rsi14"] = rsi(close)
    macd_line, sig_line, hist = macd(close)
    d["macd"], d["macd_signal"], d["macd_hist"] = macd_line, sig_line, hist
    bb_u, bb_m, bb_l = bollinger(close)
    d["bb_width"] = (bb_u - bb_l) / (bb_m + 1e-9)
    d["atr14"] = atr(high, low, close)
    d["ret1"], d["ret3"], d["ret5"] = close.pct_change(1), close.pct_change(3), close.pct_change(5)
    d["vol20"] = d["ret1"].rolling(20).std()
    d["vol_ma20"] = vol.rolling(20).mean()
    d["vol_ratio"] = vol / (d["vol_ma20"] + 1e-9)
    return d.dropna().reset_index(drop=True)

# --- جلب بيانات Yahoo Finance
def fetch_yahoo_prices(ticker: str, range_="2y", interval="1d"):
    bases = ["https://query1.finance.yahoo.com/v8/finance/chart/", "https://query2.finance.yahoo.com/v8/finance/chart/"]
    headers = {"User-Agent":"Mozilla/5.0"}
    for _ in range(3):
        for base in bases:
            try:
                url=f"{base}{ticker}"
                r = requests.get(url, params={"range":range_,"interval":interval}, headers=headers, timeout=25)
                if r.status_code != 200: continue
                js = (r.json().get("chart",{}).get("result") or [None])[0]
                if not js: continue
                quote = (js.get("indicators",{}).get("quote") or [None])[0]
                if not quote: continue
                df = pd.DataFrame({
                    "open": quote.get("open", []),
                    "high": quote.get("high", []),
                    "low": quote.get("low", []),
                    "close": quote.get("close", []),
                    "volume": quote.get("volume", [])
                }).dropna(subset=["close"])
                return df if len(df) > 0 else None
            except: continue
        time.sleep(1)
    return None

# --- تحميل الأسهم
def load_tickers(path=TICKERS_PATH):
    if not os.path.exists(path): return []
    tickers = [l.strip().upper() for l in open(path,"r",encoding="utf-8") if l.strip()]
    tickers = [t if t.endswith(".SR") else t+".SR" for t in tickers]
    return list(dict.fromkeys(tickers))

# --- توليد إشارات شراء/بيع قوية
def generate_signals(df):
    signals = []
    for i in range(1, len(df)):
        # الشروط:
        price_up = df["close"].iloc[i] > df["close"].iloc[i-1]
        ema_up = df["ema10"].iloc[i] > df["ema20"].iloc[i] and df["ema10"].iloc[i] > df["ema10"].iloc[i-1]
        macd_up = df["macd"].iloc[i] > df["macd"].iloc[i-1]
        rsi_up = df["rsi14"].iloc[i] > df["rsi14"].iloc[i-1] and df["rsi14"].iloc[i] < 70
        trend_up = df["ema10"].iloc[i] > df["ema50"].iloc[i]  # تأكيد صعودي طويل المدى

        if price_up and ema_up and macd_up and rsi_up and trend_up:
            signals.append({"date": df.index[i], "signal":"BUY", "price": df["close"].iloc[i]})
        elif df["close"].iloc[i] < df["ema50"].iloc[i]:  # كسر الاتجاه الصعودي
            signals.append({"date": df.index[i], "signal":"SELL", "price": df["close"].iloc[i]})
    return signals

# --- البرنامج الرئيسي
def main():
    tickers = load_tickers()
    all_signals = {}
    for t in tickers:
        print(f"[INFO] Fetching {t} ...")
        df = fetch_yahoo_prices(t)
        if df is None or len(df) < MIN_ROWS: 
            print(f"[WARN] {t} insufficient data")
            continue
        df_features = build_features(df)
        df_features.index = pd.to_datetime(df.index)
        signals = generate_signals(df_features)
        all_signals[t] = signals
        print(f"[OK] {t} signals={len(signals)}")
        time.sleep(0.2)

    # طباعة ملخص الإشارات
    for t, sigs in all_signals.items():
        print(f"\nTicker: {t}")
        for s in sigs[-5:]:  # آخر 5 إشارات
            print(s)

if __name__=="__main__":
    main()
