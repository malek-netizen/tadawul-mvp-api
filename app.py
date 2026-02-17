# ======================== نقاط نهاية التشخيص (للفحص فقط) ========================

@app.get("/debug/ticker/{ticker}")
async def debug_ticker(ticker: str):
    """إرجاع معلومات تشخيصية مفصلة عن سهم واحد."""
    t = ticker.strip().upper()
    if not t.endswith(".SR"):
        t += ".SR"
    
    df = fetch_yahoo_prices(t)
    if df is None:
        return {"error": "لا توجد بيانات لهذا السهم"}
    
    try:
        feat_df = build_features(df)
    except Exception as e:
        return {"error": f"فشل بناء المؤشرات: {str(e)}"}
    
    if feat_df is None or len(feat_df) == 0:
        return {"error": "البيانات غير كافية لحساب المؤشرات"}
    
    # آخر 5 أيام من البيانات الخام (قبل حساب المؤشرات)
    raw_last5 = df.tail(5).to_dict(orient="records")
    
    # آخر يوم من المؤشرات المحسوبة (تحويل القيم numpy إلى float/json)
    curr = feat_df.iloc[-1].to_dict()
    # تحويل أي قيم numpy إلى أنواع python قياسية
    for k, v in curr.items():
        if isinstance(v, (np.integer, np.floating)):
            curr[k] = float(v)
        elif isinstance(v, np.bool_):
            curr[k] = bool(v)
    
    # فحص الشروط الأساسية
    passed, reasons = passes_core_rules(feat_df)
    
    # فحص الاستبعاد
    excluded, exclude_reason = should_exclude(feat_df)
    
    # نقاط المجموعات
    scores = calculate_group_scores(feat_df)
    
    return {
        "ticker": t,
        "raw_last_5": raw_last5,
        "indicators_last": curr,
        "core_rules_passed": passed,
        "core_rules_reasons": reasons,
        "excluded": excluded,
        "exclude_reason": exclude_reason,
        "group_scores": scores
    }


# دالة مساعدة لجمع إحصائيات debug_summary (دون تخزين النتائج كاملة)
def analyze_one_debug(ticker: str):
    """نسخة مبسطة من analyze_one لإرجاع الحالة والسبب فقط."""
    t = ticker.strip().upper()
    if not t.endswith(".SR"):
        t += ".SR"
    
    df = fetch_yahoo_prices(t)
    if df is None:
        return None
    
    try:
        feat_df = build_features(df)
    except Exception:
        return None
    
    # الاستبعاد أولاً
    excluded, exclude_reason = should_exclude(feat_df)
    if excluded:
        return {"status": "EXCLUDED", "reason": exclude_reason}
    
    # الشروط الأساسية
    passed, reasons = passes_core_rules(feat_df)
    if not passed:
        return {"status": "REJECTED", "reason": " | ".join(reasons)}
    
    return {"status": "APPROVED", "reason": ""}


@app.get("/debug/summary")
async def debug_summary():
    """تحليل جميع الأسهم وإرجاع إحصائيات حول أسباب الرفض والاستبعاد."""
    if not os.path.exists(TICKERS_PATH):
        raise HTTPException(status_code=500, detail=f"ملف {TICKERS_PATH} غير موجود")
    
    with open(TICKERS_PATH, "r", encoding="utf-8") as f:
        tickers = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    
    stats = {
        "total": 0,
        "excluded_count": 0,
        "excluded_reasons": {},
        "core_failed_count": 0,
        "core_failed_reasons": {},
        "approved_count": 0
    }
    
    with ThreadPoolExecutor(max_workers=TOP10_WORKERS) as executor:
        futures = {executor.submit(analyze_one_debug, t): t for t in tickers}
        for future in as_completed(futures):
            res = future.result()
            if res is None:
                continue  # تجاهل الأسهم التي فشل جلب بياناتها
            stats["total"] += 1
            if res["status"] == "EXCLUDED":
                stats["excluded_count"] += 1
                reason = res["reason"]
                stats["excluded_reasons"][reason] = stats["excluded_reasons"].get(reason, 0) + 1
            elif res["status"] == "REJECTED":
                stats["core_failed_count"] += 1
                # قد يكون هناك عدة أسباب مفصولة بـ " | "
                reasons_list = res["reason"].split(" | ")
                for r in reasons_list:
                    stats["core_failed_reasons"][r] = stats["core_failed_reasons"].get(r, 0) + 1
            elif res["status"] == "APPROVED":
                stats["approved_count"] += 1
    
    return stats
