import pandas as pd
import numpy as np

# ==========================================
# 1. منطق "قناص القيعان" المعدل (The Core)
# ==========================================
def passes_rules(feat_df: pd.DataFrame):
    reasons = []
    curr = feat_df.iloc[-1]
    prev = feat_df.iloc[-2]
    
    # [تعديل 1]: شرط السيولة الانفجارية (محرك أسهم النسب)
    # يجب أن يكون حجم تداول الساعة الحالية أعلى من المتوسط بـ 20% على الأقل
    cond_vol_surge = curr['volume'] > (curr['vol_ma20'] * 1.2)
    if not cond_vol_surge:
        reasons.append("Rejected: Weak Volume (No real surge)")

    # [تعديل 2]: اختراق منتصف البولينجر (نقطة التحول من خمول لارتفاع)
    cond_bb_break = (curr['close'] > curr['bb_mid']) and (prev['close'] <= prev['bb_mid'])
    # التأكد أنه كان تحت المتوسط في آخر 5 ساعات (يعني جاي من قاع)
    was_below = (feat_df['close'].shift(1).tail(5) < feat_df['bb_mid'].shift(1).tail(5)).any()
    if not (cond_bb_break or (curr['close'] > curr['bb_mid'] and was_below)):
        reasons.append("Rejected: No Mid-BB Breakout")

    # [تعديل 3]: استدارة الماكد من تحت الصفر (بصمة ساسكو)
    cond_macd_turn = (curr['macd'] > curr['macd_signal']) and (curr['macd'] < 0.2)
    if not cond_macd_turn:
        reasons.append("Rejected: MACD not turning from base")

    # [تعديل 4]: سقف الأمان (تجنب القمم المتضخمة)
    if curr['rsi14'] > 65:
        reasons.append(f"Rejected: RSI Overextended ({round(curr['rsi14'],1)})")
    
    # [تعديل 5]: منع الانفجار السعري المبالغ فيه (المطاردة)
    dist_from_ema = (curr['close'] - curr['ema20']) / curr['ema20']
    if dist_from_ema > 0.03:
        reasons.append("Rejected: Price too far from EMA20")

    return (len(reasons) == 0), reasons

# ==========================================
# 2. تحليل السهم وحساب الوقف الفني
# ==========================================
def analyze_one(ticker: str):
    # (كود جلب البيانات وبناء الميزات feat_df يبقى كما هو)
    
    df = fetch_yahoo_prices(ticker)
    if df is None or len(df) < 30: return None
    feat_df = build_features(df)
    curr = feat_df.iloc[-1]
    last_close = float(curr["close"])
    
    # [تعديل 6]: إيقاف الخسارة الفني (تحت أدنى سعر لآخر 3 شموع)
    recent_low = float(feat_df['low'].tail(3).min())
    technical_stop = round(recent_low * 0.997, 2) # هامش أمان بسيط (0.3%)
    
    # حماية: الوقف لا يقل عن 1.5% ولا يزيد عن 4% من سعر الدخول
    stop_dist_pct = (last_close - technical_stop) / last_close
    if stop_dist_pct > 0.04: technical_stop = round(last_close * 0.96, 2)
    if stop_dist_pct < 0.015: technical_stop = round(last_close * 0.985, 2)

    ok, reasons = passes_rules(feat_df)
    
    # [تعديل 7]: حساب الثقة بوزن ثقيل للسيولة
    score = 0
    if curr['volume'] > curr['vol_ma20'] * 1.5: score += 40 # سيولة ضخمة
    elif curr['volume'] > curr['vol_ma20'] * 1.2: score += 30 # سيولة جيدة
    
    if (curr['macd'] > curr['macd_signal']): score += 30
    if (curr['close'] > curr['bb_mid']): score += 30
    
    recommendation = "BUY" if ok else "NO_TRADE"
    
    return {
        "ticker": ticker,
        "recommendation": recommendation,
        "confidence_pct": min(100, score),
        "entry": round(last_close, 2),
        "take_profit": round(last_close * 1.05, 2), # الهدف الأول 5%
        "stop_loss": technical_stop,
        "reason": " | ".join(reasons) if reasons else "سيولة انفجارية + اختراق قاع (نموذج النسب)",
        "status": "APPROVED" if ok else "REJECTED"
    }
