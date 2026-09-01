"""
AsthmAI - Fix #3: Redesigned Hybrid System (Safety-Net Architecture)
Key change: heuristics ONLY upgrade risk (never downgrade).
Uses the exact same feature pipeline as ensemble_model.py.
"""
import pickle
import numpy as np
import pandas as pd
import warnings
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')


def advanced_feature_engineering(df):
    df = df.copy()
    df['AQI_PM_ratio']       = df['AQI'] / (df['PM2.5'] + 1)
    df['pollution_index']    = (df['AQI'] * 0.4 + df['PM2.5'] * 0.3 +
                                df['NO2 level'] * 0.15 + df['SO2 level'] * 0.15)
    df['gas_pollution']      = df['CO2 level'] * df['NO2 level'] * df['SO2 level'] / 10000
    df['humidity_pollution'] = df['Humidity'] * df['pollution_index'] / 100
    df['temp_pollution']     = df['Temperature'] * df['pollution_index'] / 100
    df['AQI_critical']       = (df['AQI'] > 200).astype(int)
    df['AQI_unhealthy']      = ((df['AQI'] > 100) & (df['AQI'] <= 200)).astype(int)
    df['PM25_high']          = (df['PM2.5'] > 75).astype(int)

    symptom_map  = {'Daily': 4, 'Frequently (Weekly)': 3, '1-2 times a month': 2, 'Less than once a month': 1}
    exposure_map = {'Yes, often': 3, 'Occasionally': 2, 'No': 1}
    night_map    = {'Frequently': 3, 'Occasionally': 2, 'Rarely': 1, 'Never': 0}

    df['symptom_severity'] = df['Asthma Symptoms Frequency'].map(symptom_map).fillna(0)
    df['exposure_score']   = df['Poor Air Quality Exposure'].map(exposure_map).fillna(0)
    df['night_score']      = df['Night Breathing Difficulty'].map(night_map).fillna(0)
    df['trigger_count']    = df['Triggers'].apply(lambda x: str(x).count(',') + 1)

    df['clinical_risk_score'] = (df['symptom_severity'] * 0.4 +
                                  df['exposure_score'] * 0.3 +
                                  df['night_score'] * 0.3)
    max_pi = max(df['pollution_index'].max(), 1)
    df['env_risk_score'] = (df['AQI_critical'] * 0.3 + df['AQI_unhealthy'] * 0.2 +
                             df['PM25_high'] * 0.25 +
                             (df['pollution_index'] / max_pi) * 0.25)
    df['total_risk_interaction'] = df['clinical_risk_score'] * df['env_risk_score']
    return df


def prepare_X(df, scaler):
    numerical_cols  = ['AQI', 'PM2.5', 'SO2 level', 'NO2 level', 'CO2 level', 'Humidity', 'Temperature']
    engineered_cols = ['AQI_PM_ratio', 'pollution_index', 'gas_pollution', 'humidity_pollution',
                       'temp_pollution', 'AQI_critical', 'AQI_unhealthy', 'PM25_high',
                       'symptom_severity', 'exposure_score', 'night_score', 'trigger_count',
                       'clinical_risk_score', 'env_risk_score', 'total_risk_interaction']
    categorical_cols = ['Asthma Symptoms Frequency', 'Triggers', 'Weather Sensitivity',
                        'Poor Air Quality Exposure', 'Night Breathing Difficulty']

    df = advanced_feature_engineering(df)
    X_num = df[numerical_cols + engineered_cols].values
    X_cat_list = []
    for col in categorical_cols:
        le = LabelEncoder()
        le.fit(df[col].astype(str))
        X_cat_list.append(le.transform(df[col].astype(str)))
    X_cat = np.column_stack(X_cat_list)
    X = np.hstack([X_num, X_cat])
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    return scaler.transform(X)


def shannon_entropy(proba):
    eps = 1e-10
    proba = np.clip(proba, eps, 1.0)
    return -np.sum(proba * np.log2(proba), axis=1)


def run_hybrid_v2():
    print("=" * 60)
    print("AsthmAI - Safety-Net Hybrid System v2")
    print("=" * 60)

    with open('results/best_ensemble_model.pkl', 'rb') as f:
        saved = pickle.load(f)
    model  = saved['model']
    scaler = saved['scaler']
    le     = saved['label_encoder']

    df_test = pd.read_csv('data/test.csv')
    y_true  = le.transform(df_test['Risk Class'])
    X_sc    = prepare_X(df_test, scaler)

    classes   = list(le.classes_)  # e.g. ['High', 'Low', 'Medium']
    high_idx  = classes.index('High')
    med_idx   = classes.index('Medium')
    low_idx   = classes.index('Low')

    # ── Pure ML baseline ──────────────────────────────────────────
    ml_proba  = model.predict_proba(X_sc)
    ml_pred   = model.predict(X_sc)
    ml_acc    = accuracy_score(y_true, ml_pred)
    ml_f1     = f1_score(y_true, ml_pred, average='weighted')

    # ── Safety-Net Hybrid ─────────────────────────────────────────
    # Rule: ONLY upgrade risk, NEVER downgrade
    hybrid_pred = ml_pred.copy()
    triggered   = np.zeros(len(df_test), dtype=bool)

    for i, (_, row) in enumerate(df_test.iterrows()):
        cur   = hybrid_pred[i]
        freq  = str(row.get('Asthma Symptoms Frequency', ''))
        night = str(row.get('Night Breathing Difficulty', ''))
        aqi   = float(row.get('AQI', 0))

        # Rule A: Daily symptoms → upgrade to High
        if freq == 'Daily' and cur != high_idx:
            hybrid_pred[i] = high_idx
            triggered[i]   = True

        # Rule B: Weekly symptoms → upgrade to at least Medium
        elif freq == 'Frequently (Weekly)' and cur == low_idx:
            hybrid_pred[i] = med_idx
            triggered[i]   = True

        # Rule C: Frequent night difficulty AND high AQI → upgrade to at least Medium
        elif night == 'Frequently' and aqi > 150 and cur == low_idx:
            hybrid_pred[i] = med_idx
            triggered[i]   = True

        # Rule D: Extreme pollution → upgrade to at least Medium
        elif aqi > 200 and cur == low_idx:
            hybrid_pred[i] = med_idx
            triggered[i]   = True

    hybrid_acc = accuracy_score(y_true, hybrid_pred)
    hybrid_f1  = f1_score(y_true, hybrid_pred, average='weighted')

    # ── Entropy confidence tiers ──────────────────────────────────
    H         = shannon_entropy(ml_proba)
    high_conf = H < 0.5
    med_conf  = (H >= 0.5) & (H < 1.0)
    low_conf  = H >= 1.0

    def safe_acc(mask):
        if mask.sum() == 0: return 0.0
        return accuracy_score(y_true[mask], hybrid_pred[mask])

    hc_acc = safe_acc(high_conf)
    mc_acc = safe_acc(med_conf)
    lc_acc = safe_acc(low_conf)

    print(f"\nTotal Samples:        {len(y_true)}")
    print(f"Heuristic Triggers:   {triggered.sum()} ({triggered.mean()*100:.1f}%)")
    print(f"\n--- ACCURACY ---")
    print(f"Pure ML Ensemble:     {ml_acc*100:.2f}%   F1={ml_f1:.4f}")
    print(f"Safety-Net Hybrid:    {hybrid_acc*100:.2f}%   F1={hybrid_f1:.4f}")
    print(f"Improvement:          {(hybrid_acc - ml_acc)*100:+.2f}%")
    print(f"\n--- ENTROPY CONFIDENCE ---")
    print(f"High Conf (H<0.5) :  {high_conf.sum():3d} ({high_conf.mean()*100:.0f}%)   Acc={hc_acc*100:.1f}%")
    print(f"Med  Conf (0.5-1) :  {med_conf.sum():3d} ({med_conf.mean()*100:.0f}%)   Acc={mc_acc*100:.1f}%")
    print(f"Low  Conf (H>=1.0):  {low_conf.sum():3d} ({low_conf.mean()*100:.0f}%)   Acc={lc_acc*100:.1f}%")
    print(f"\n--- CLASSIFICATION REPORT (HYBRID) ---")
    print(classification_report(y_true, hybrid_pred, target_names=le.classes_))

    with open('results/hybrid_v2_results.txt', 'w') as f:
        f.write("AsthmAI Safety-Net Hybrid System v2\n")
        f.write("=" * 40 + "\n")
        f.write(f"Total Samples:      {len(y_true)}\n")
        f.write(f"Heuristic Triggers: {triggered.sum()} ({triggered.mean()*100:.1f}%)\n\n")
        f.write(f"Pure ML Accuracy:   {ml_acc:.4f}\n")
        f.write(f"Hybrid Accuracy:    {hybrid_acc:.4f}\n")
        f.write(f"Improvement:        {(hybrid_acc - ml_acc)*100:+.2f}%\n\n")
        f.write(f"High Conf ({high_conf.mean()*100:.0f}% coverage): Acc={hc_acc*100:.1f}%\n")
        f.write(f"Med  Conf ({med_conf.mean()*100:.0f}% coverage): Acc={mc_acc*100:.1f}%\n")
        f.write(f"Low  Conf ({low_conf.mean()*100:.0f}% coverage): Acc={lc_acc*100:.1f}%\n\n")
        f.write(classification_report(y_true, hybrid_pred, target_names=le.classes_))

    print("\nSaved: results/hybrid_v2_results.txt")
    return hybrid_acc, ml_acc


if __name__ == "__main__":
    run_hybrid_v2()
