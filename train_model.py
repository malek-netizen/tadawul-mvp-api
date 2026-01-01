import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from app import fetch_yahoo_prices, build_features, FEATURE_COLS

# اختر مجموعة أسهم للتدريب (غير 2222 فقط)
TICKERS = [
    "1120.SR", "2010.SR", "2082.SR", "1180.SR", "2280.SR",
    "2200.SR", "1211.SR", "1050.SR"
]

HORIZON_DAYS = 10       # أفق التقييم
TP = 0.05               # +5%
SL = -0.02              # -2%

def label_tp_sl(close_series, horizon=10, tp=0.05, sl=-0.02):
    """
    Label = 1 إذا خلال الأفق تحقق TP قبل SL
    Label = 0 خلاف ذلك
    """
    closes = close_series.values
    labels = []

    for i in range(len(closes)):
        entry = closes[i]
        tp_price = entry * (1 + tp)
        sl_price = entry * (1 + sl)

        # لا نقدر نحكم إذا ما عندنا مستقبل كافي
        if i + horizon >= len(closes):
            labels.append(np.nan)
            continue

        future = closes[i+1:i+horizon+1]
        hit_tp = np.where(future >= tp_price)[0]
        hit_sl = np.where(future <= sl_price)[0]

        if len(hit_tp) == 0 and len(hit_sl) == 0:
            labels.append(0)
        elif len(hit_tp) > 0 and len(hit_sl) == 0:
            labels.append(1)
        elif len(hit_tp) == 0 and len(hit_sl) > 0:
            labels.append(0)
        else:
            # كلاهما حدث: نأخذ الأقرب زمنًا (TP قبل SL؟)
            labels.append(1 if hit_tp[0] < hit_sl[0] else 0)

    return pd.Series(labels)

def build_dataset():
    all_rows = []

    for t in TICKERS:
        df = fetch_yahoo_prices(t, range_="3y", interval="1d")
        if df is None:
            print("No data for", t)
            continue

        feat = build_features(df)
        feat["label"] = label_tp_sl(feat["close"], horizon=HORIZON_DAYS, tp=TP, sl=SL)

        feat = feat.dropna(subset=["label"]).reset_index(drop=True)
        if len(feat) < 200:
            print("Too few rows after labeling for", t)
            continue

        all_rows.append(feat)

    if not all_rows:
        raise RuntimeError("No training data collected. Try different tickers.")

    data = pd.concat(all_rows, ignore_index=True)
    return data

if __name__ == "__main__":
    data = build_dataset()

    X = data[FEATURE_COLS].astype(float)
    y = data["label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)

    prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, prob)
    print("AUC:", auc)

    joblib.dump(model, "model.joblib")
    print("Saved model.joblib")
