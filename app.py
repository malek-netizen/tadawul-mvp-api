# app.py
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import yfinance as yf

app = FastAPI(title="Tadawul MVP Technical Signal API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # مناسب لـ Thunkable
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# إعدادات الاستراتيجية (مثل البحث)
# -----------------------------
TP = 0.05         # +5%
SL = 0.02         # -2%
HOLDING_DAYS = 30 # للعرض فقط (MVP)

# مؤشرات
RSI_PERIOD = 14
BB_WINDOW = 20
BB_STD = 2.0

EMA_FAST = 12
EMA_SLOW = 26
MACD_SIGNAL = 9

SMA_LONG = 50


# -----------------------------
# مساعدات المؤشرات
# -----------------------------
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    return 100 - (100 / (1 + rs))

def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def bollinger(close: pd.Series, window: int = 20, num_std: float = 2.0):
    mid = close.rolling(window, min_periods=window).mean()
    std = close.rolling(window, min_periods=window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower

def download_prices(ticker: str, period: str = "1y") -> pd.DataFrame:
    df = yf.download(ticker, period=period, auto_adjust=False, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.dropna(inplace=True)
    return df


# -----------------------------
# منطق التوصية (MVP Rules)
# -----------------------------
def compute_signal_and_confidence(df: pd.DataFrame):
    """
    قواعد بسيطة وواقعية:
    - BUY إذا:
      1) الاتجاه صاعد: Close > SMA50
      2) MACD histogram إيجابي (زخم صاعد)
      3) RSI ليس متشبع شراء قوي (RSI < 70)
      4) Close قريب/أعلى من BB_mid (استمرار اتجاه)
    - NO_TRADE خلاف ذلك
    Confidence = متوسط نقاط الشروط (0..1)
    """
    close = df["Close"]

    df["SMA50"] = close.rolling(SMA_LONG, min_periods=SMA_LONG).mean()
    df["RSI"] = rsi(close, RSI_PERIOD)
    df["MACD"], df["MACD_SIGNAL"], df["MACD_HIST"] = macd(close, EMA_FAST, EMA_SLOW, MACD_SIGNAL)
    df["BB_UPPER"], df["BB_MID"], df["BB_LOWER"] = bollinger(close, BB_WINDOW, BB_STD)

    last = df.iloc[-1]

    conditions = []

    # شرط 1: Trend filter
    c1 = bool(last["Close"] > last["SMA50"]) if pd.notna(last["SMA50"]) else False
    conditions.append(c1)

    # شرط 2: MACD momentum
    c2 = bool(last["MACD_HIST"] > 0) if pd.notna(last["MACD_HIST"]) else False
    conditions.append(c2)

    # شرط 3: RSI not overbought hard
    c3 = bool(last["RSI"] < 70) if pd.notna(last["RSI"]) else False
    conditions.append(c3)

    # شرط 4: price above BB_mid (trend confirmation)
    c4 = bool(last["Close"] >= last["BB_MID"]) if pd.notna(last["BB_MID"]) else False
    conditions.append(c4)

    confidence = float(np.mean(conditions))  # 0..1

    recommendation = "BUY" if confidence >= 0.75 else "NO_TRADE"

    details = {
        "close": float(round(last["Close"], 4)),
        "rsi": float(round(last["RSI"], 4)) if pd.notna(last["RSI"]) else None,
        "macd_hist": float(round(last["MACD_HIST"], 6)) if pd.notna(last["MACD_HIST"]) else None,
        "sma50": float(round(last["SMA50"], 4)) if pd.notna(last["SMA50"]) else None,
        "bb_mid": float(round(last["BB_MID"], 4)) if pd.notna(last["BB_MID"]) else None,
    }

    return recommendation, confidence, details


# -----------------------------
# Endpoints
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/predict")
def predict(
    ticker: str = Query(..., description="Saudi ticker in Yahoo format, e.g., 2222.SR")
):
    try:
        df = download_prices(ticker, period="1y")
        if df.empty:
            return {"error": "No price data returned. Check ticker format like 2222.SR."}

        # تحتاج بيانات كافية لحساب SMA50 و Bollinger
        if len(df) < 80:
            return {"ticker": ticker, "recommendation": "NO_TRADE", "reason": "Not enough price history (need ~80 days)."}

        reco, conf, details = compute_signal_and_confidence(df)

        entry = float(round(df["Close"].iloc[-1], 2))
        take_profit = float(round(entry * (1 + TP), 2))
        stop_loss = float(round(entry * (1 - SL), 2))

        return {
            "ticker": ticker,
            "recommendation": reco,
            "entry": entry,
            "take_profit": take_profit,
            "stop_loss": stop_loss,
            "holding_days": HOLDING_DAYS,
            "confidence": round(conf, 4),
            "indicators": details,
            "as_of": str(df.index[-1].date()),
            "disclaimer": "For educational/research purposes only. Not financial advice."
        }

    except Exception as e:
        return {"error": str(e)}
