from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import requests
import pandas as pd
import numpy as np
import time
import joblib
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

app = FastAPI(title="Tadawul MVP API", version="2.2.2-stable-reversal")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- إعدادات ثابتة ---
MODEL_PATH = "model.joblib"
TICKERS_PATH = "tickers_sa.txt"
TP_PCT = 0.05
SL_PCT = 0.02
ML_THRESHOLD = 0.60

# --- حدود المؤشرات ---
RSI_MIN = 40
RSI_MAX = 65
RSI_OVERBOUGHT = 70
MAX_ABOVE_EMA20 = 0.06
CONSOL_DAYS = 4
CONSOL_RANGE_MAX = 0.06
LOOKBACK_LOW_DAYS = 10

model = None

# ==========================================
# NEW MODIFICATION: وظائف التأكد من الانعكاس الصاعد
# ==========================================
def is_reversing_up(feat_df: pd.DataFrame) -> bool:
    """
    شرطك الأساسي: السعر الحالي أكبر مما قبله + تكوين قاع صاعد لحظي
    """
    if len(feat_df) < 3:
        return False
    
    c_today = float(feat_df["close"].iloc[-1])
    c_yest = float(feat_df["close"].iloc[-2])
    c_prev2 = float(feat_df["close"].iloc[-3])
    
    # 1. السعر الحالي أكبر من السابق (ليس في هبوط اليوم)
    price_rising = c_today > c_yest
    
    # 2. السعر الحالي أكبر من قبل يومين (تأكيد كسر سلسلة الهبوط)
    higher_than_prev2 = c_today > c_prev2
    
    return price_rising and higher_than_prev2

def is_volume_confirming(feat_df: pd.DataFrame) -> bool:
    """
    تأكيد أن الصعود مدعوم بسيولة أعلى من المتوسط
    """
    if len(feat_df) < 5:
        return True
    current_vol = feat_df["volume"].iloc[-1]
    avg_vol = feat_df["volume"].tail(5).mean()
    return current_vol >= avg_vol * 0.9  # السيولة ليست ضعيفة جداً

# ==========================================
# الدوال الأساسية (سحب البيانات، المؤشرات، إلخ)
# ==========================================
# (ملاحظة: أبقيت دوال sma, ema, rsi, macd, fetch_yahoo_prices كما هي في كودك الأصلي)
# [يرجى إدراج الدوال الفنية الأصلية هنا عند التشغيل]

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    # (نفس دالة build_features الأصلية في كودك)
    d = df.copy()
    # ... حساب المؤشرات ...
    return d.dropna().reset_index(drop=True)

# ==========================================
# تعديل قواعد الدخول (The Logic Update)
# ==========================================
def passes_rules(df_ohlc: pd.DataFrame, feat_df: pd.DataFrame):
    reasons = []
    last = feat_df.iloc[-1]
    
    # --- شروطك القديمة ---
    if rsi(feat_df["close"]).iloc[-1] > RSI_OVERBOUGHT:
        reasons.append("Rejected: RSI overbought.")
    
    # --- التعديل الجديد (شرط المسار الهابط) ---
    if not is_reversing_up(feat_df):
        reasons.append("Rejected: Price is in a downward sequence (Wait for Green/Higher Close).")
    
    # التحقق من ميل المتوسط المتحرك (يجب ألا يكون منحدراً للأسفل بحدة)
    if len(feat_df) >= 2:
        if last["ema20"] < feat_df["ema20"].iloc[-2]:
             # إذا كان المسار العام لـ 20 يوم هابط، نرفض الدخول إلا لو كان هناك تذبذب تجميعي
             if not is_consolidating(feat_df):
                reasons.append("Rejected: Long-term trend (EMA20) is still declining.")

    # تأكيد السيولة
    if not is_volume_confirming(feat_df):
        reasons.append("Rejected: Low volume on reversal attempt.")

    # (باقي شروط Consolidation و Trend Score الأصلية)
    # ...
    
    return (len(reasons) == 0), reasons

def rules_score(feat_df: pd.DataFrame) -> int:
    # نظام النقاط المطور بناءً على شرطك
    score = 0
    last = feat_df.iloc[-1]
    
    # +25 نقطة إذا تحقق شرطك (السعر الحالي > السابق)
    if is_reversing_up(feat_df):
        score += 25
    
    # +20 نقطة إذا كان فوق المتوسطات
    if float(last["close"]) >= float(last["ema20"]): score += 20
    
    # +15 نقطة للسيولة
    if is_volume_confirming(feat_df): score += 15
    
    # +20 نقطة لـ RSI في النطاق الذهبي
    r = float(last["rsi14"])
    if 40 <= r <= 60: score += 20
    
    return int(min(100, score))

# ==========================================
# تحليل سهم واحد (The Final Logic)
# ==========================================
def analyze_one(ticker: str):
    # (نفس دالة analyze_one الأصلية مع استدعاء الـ Rules المحدثة)
    # ...
    # سيقوم الآن تلقائياً برفض أي سهم "إغلاقه اليوم أقل من أمس"
    # وسيعطيه حالة REJECTED مع ذكر السبب في الـ reason
    pass

# (باقي دوال FastAPI و /top10 تبقى كما هي)
