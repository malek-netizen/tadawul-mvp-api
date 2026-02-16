def analyze_one(ticker: str):
    try:
        t = ticker.strip().upper()
        if not t.endswith(".SR"): t += ".SR"
        
        df_raw = fetch_data(t, interval="1h", period="60d")
        if df_raw is None or len(df_raw) < 30: return None

        # الفلتر الذكي: تجاهل شمعة المزاد والبيانات الخامدة
        last_vol = df_raw.iloc[-1]['volume']
        avg_vol = df_raw['volume'].tail(20).mean()
        df_active = df_raw.iloc[:-1] if last_vol < (avg_vol * 0.1) else df_raw
            
        df_active = apply_indicators(df_active)
        curr = df_active.iloc[-1]
        prev = df_active.iloc[-2]

        # --- 1. شرط الزخم المتصاعد (Momentum Check) ---
        # الماكد إيجابي + الفجوة بين الخطين تتوسع (يعني القوة تزيد)
        macd_gap = curr['macd'] - curr['signal']
        prev_macd_gap = prev['macd'] - prev['signal']
        cond_macd = (curr['macd'] > curr['signal']) and (macd_gap > prev_macd_gap)

        # --- 2. شرط عدم التضخم (Anti-Top Filter) ---
        # RSI صاعد (أعلى من السابق) لكنه "مرتاح" تحت الـ 60 (يمنع دخول مثل الأهلي)
        cond_rsi = (curr['rsi'] > prev['rsi']) and (curr['rsi'] < 60)

        # --- 3. شرط القرب من القاعدة (Base Check) ---
        # يمنع مطاردة الأسهم التي طارت بعيداً عن متوسطها (أقل من 3% من SMA20)
        is_not_extended = (curr['close'] < curr['sma20'] * 1.03)

        # --- 4. شروط إضافية (السيولة والمساحة) ---
        cond_vol = curr['volume'] > (curr['vol_avg'] * 0.9)
        cond_bb_space = curr['close'] < (curr['bb_mid'] + (curr['bb_upper'] - curr['bb_mid']) * 0.5)

        # ميزان النقاط الجديد
        score = 0
        if cond_macd:       score += 30 # قوة الزخم
        if cond_rsi:        score += 30 # تأكيد الصعود الآمن
        if is_not_extended: score += 20 # الأمان من القمم
        if cond_vol:        score += 10 # الوقود
        if cond_bb_space:   score += 10 # مساحة الهدف

        recommendation = "BUY" if score >= 80 else "NO_TRADE"

        return {
            "ticker": t,
            "recommendation": recommendation,
            "confidence_pct": score,
            "
