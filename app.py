import requests
import pandas as pd

def fetch_yahoo_prices(ticker: str, range_: str = "1y", interval: str = "1d"):
    """
    Fetch OHLCV data using Yahoo chart endpoint.
    Returns: DataFrame with columns: ['open','high','low','close','volume']
    """
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {"range": range_, "interval": interval}
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
    }

    r = requests.get(url, params=params, headers=headers, timeout=20)
    r.raise_for_status()
    js = r.json()

    chart = js.get("chart", {})
    if chart.get("error"):
        return None

    result = (chart.get("result") or [None])[0]
    if not result:
        return None

    timestamps = result.get("timestamp") or []
    quote = (result.get("indicators", {}).get("quote") or [None])[0]
    if not timestamps or not quote:
        return None

    df = pd.DataFrame({
        "open": quote.get("open", []),
        "high": quote.get("high", []),
        "low": quote.get("low", []),
        "close": quote.get("close", []),
        "volume": quote.get("volume", []),
    })

    # تنظيف القيم الفارغة
    df = df.dropna(subset=["close"]).reset_index(drop=True)
    if len(df) < 60:
        return None

    return df
