import os
import pandas as pd
import numpy as np

# مسار البيانات الخام
base_path = "/content/saudi_stocks_raw_data"

# جلب قائمة الأسهم
tickers = [d for d in os.listdir(base_path)
           if os.path.isdir(os.path.join(base_path, d))
           and not d.startswith(".")]

# دالة لتقييم السهم بناء على المعايير الأساسية والثانوية
def evaluate_stock(df_price, df_financial):
    """
    df_price: بيانات السعر اليومية
    df_financial: القوائم المالية السنوية
    """
    # 1️⃣ المعايير الأساسية
    # شرط 1: حجم التداول اليومي متوسط أكبر من حد معين
    avg_volume = df_price['volume'].mean()
    if avg_volume < 500000:  # مثال للحد الأدنى
        return 'No Trade'
    
    # شرط 2: السعر الحالي ضمن 52-week range
    current_price = df_price['close'].iloc[-1]
    low_52 = df_price['low'].min()
    high_52 = df_price['high'].max()
    if current_price < low_52 or current_price > high_52:
        return 'No Trade'
    
    # 2️⃣ المعايير الثانوية (تعطي نقاط)
    score = 0
    
    # اتجاه السعر: سعر آخر يوم أعلى من متوسط 20 يوم
    ma20 = df_price['close'].rolling(window=20).mean().iloc[-1]
    if current_price > ma20:
        score += 1
    
    # RSI افتراضي: إذا RSI بين 30 و70 نزيد نقطة
    # هنا مجرد مثال: نحسب RSI تقريبي
    delta = df_price['close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    if 30 < rsi.iloc[-1] < 70:
        score += 1
    
    # معدل نمو الأرباح السنوي أكبر من 0
    if df_financial['net_income'].pct_change().iloc[-1] > 0:
        score += 1
    
    # 3️⃣ التقييم النهائي حسب النقاط الثانوية
    if score >= 2:
        return 'Buy'
    elif score == 1:
        return 'Follow'
    else:
        return 'No Trade'

# 4️⃣ جمع نتائج كل الأسهم
results = []

for ticker in tickers:
    try:
        price_file = os.path.join(base_path, ticker, "price.csv")
        financial_file = os.path.join(base_path, ticker, "financial.csv")
        
        df_price = pd.read_csv(price_file)
        df_financial = pd.read_csv(financial_file)
        
        rating = evaluate_stock(df_price, df_financial)
        results.append({
            'Ticker': ticker,
            'Rating': rating,
            'Current Price': df_price['close'].iloc[-1]
        })
    except Exception as e:
        print(f"خطأ في السهم {ticker}: {e}")

# تحويل النتائج إلى DataFrame وترتيب Top 10 حسب الأولوية
df_results = pd.DataFrame(results)

# ترتيب: Buy > Follow > No Trade ثم حسب السعر الحالي
priority_map = {'Buy': 3, 'Follow': 2, 'No Trade': 1}
df_results['Priority'] = df_results['Rating'].map(priority_map)
df_results.sort_values(by=['Priority', 'Current Price'], ascending=[False, False], inplace=True)

# اختيار Top 10
top10 = df_results.head(10)

print("=== Top 10 Stocks ===")
print(top10[['Ticker', 'Rating', 'Current Price']])
