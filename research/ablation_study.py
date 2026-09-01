"""
AsthmAI - Fix #5: Ablation Study
Tests contribution of each component (feature groups, number of base learners)
to demonstrate novelty and justify architecture choices.
"""
import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import (RandomForestClassifier, StackingClassifier,
                               GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings('ignore')
RANDOM_STATE = 42


def load_data():
    train = pd.read_csv('data/train.csv')
    val   = pd.read_csv('data/validation.csv')
    test  = pd.read_csv('data/test.csv')
    df    = pd.concat([train, val, test], ignore_index=True)

    le = LabelEncoder()
    y  = le.fit_transform(df['Risk Class'])

    # Numerical features
    numerical = ['AQI', 'PM2.5', 'SO2 level', 'NO2 level', 'CO2 level', 'Humidity', 'Temperature']

    # Engineered features
    df['pollution_index']   = df['AQI']*0.4 + df['PM2.5']*0.3 + df['NO2 level']*0.15 + df['SO2 level']*0.15
    df['AQI_critical']      = (df['AQI'] > 200).astype(int)
    df['AQI_unhealthy']     = ((df['AQI'] > 100) & (df['AQI'] <= 200)).astype(int)
    df['PM25_high']         = (df['PM2.5'] > 75).astype(int)
    df['AQI_PM_ratio']      = df['AQI'] / (df['PM2.5'] + 1)
    df['gas_pollution']     = df['CO2 level'] * df['NO2 level'] * df['SO2 level'] / 10000
    df['humidity_pollution']= df['Humidity'] * df['pollution_index'] / 100
    df['temp_pollution']    = df['Temperature'] * df['pollution_index'] / 100

    symp_map = {'Daily': 4, 'Frequently (Weekly)': 3, '1-2 times a month': 2, 'Less than once a month': 1}
    exp_map  = {'Yes, often': 3, 'Occasionally': 2, 'No': 1}
    night_map= {'Frequently': 3, 'Occasionally': 2, 'Rarely': 1, 'Never': 0}

    df['symptom_severity']  = df['Asthma Symptoms Frequency'].map(symp_map).fillna(0)
    df['exposure_score']    = df['Poor Air Quality Exposure'].map(exp_map).fillna(0)
    df['night_score']       = df['Night Breathing Difficulty'].map(night_map).fillna(0)
    df['clinical_risk']     = df['symptom_severity']*0.4 + df['exposure_score']*0.3 + df['night_score']*0.3
    df['env_risk']          = df['AQI_critical']*0.3 + df['AQI_unhealthy']*0.2 + df['PM25_high']*0.25
    df['risk_interaction']  = df['clinical_risk'] * df['env_risk']

    engineered = ['pollution_index', 'AQI_critical', 'AQI_unhealthy', 'PM25_high',
                  'AQI_PM_ratio', 'gas_pollution', 'humidity_pollution', 'temp_pollution',
                  'symptom_severity', 'exposure_score', 'night_score',
                  'clinical_risk', 'env_risk', 'risk_interaction']

    cat_encoded = ['symptom_severity', 'exposure_score', 'night_score']

    return df, y, le, numerical, engineered


def evaluate_cv(X, y, model_fn, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    accs, f1s = [], []
    for train_idx, val_idx in skf.split(X, y):
        Xtr, Xva = X[train_idx], X[val_idx]
        ytr, yva = y[train_idx], y[val_idx]
        sc  = StandardScaler()
        Xtr = sc.fit_transform(Xtr)
        Xva = sc.transform(Xva)
        m   = model_fn()
        m.fit(Xtr, ytr)
        yp  = m.predict(Xva)
        accs.append(accuracy_score(yva, yp))
        f1s.append(f1_score(yva, yp, average='weighted'))
    return np.mean(accs), np.std(accs), np.mean(f1s)


def stacking_n(n_learners):
    """Return a stacking model with n base learners."""
    all_base = [
        ('xgb',  XGBClassifier(n_estimators=100, random_state=RANDOM_STATE,
                               eval_metric='mlogloss', verbosity=0)),
        ('lgbm', LGBMClassifier(n_estimators=100, random_state=RANDOM_STATE, verbose=-1)),
        ('rf',   RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)),
        ('gb',   GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE)),
        ('et',   ExtraTreesClassifier(n_estimators=100, random_state=RANDOM_STATE)),
        ('mlp',  MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300,
                               random_state=RANDOM_STATE)),
    ]
    base = all_base[:n_learners]
    meta = LogisticRegression(max_iter=300, random_state=RANDOM_STATE)
    return lambda: StackingClassifier(estimators=base, final_estimator=meta, cv=3)


def run_ablation():
    print("=" * 60)
    print("AsthmAI - Ablation Study")
    print("=" * 60)

    df, y, le, numerical, engineered = load_data()
    results = []

    # ── Feature Ablation ─────────────────────────────────────────
    print("\n--- FEATURE GROUP ABLATION ---")

    configs = {
        'Environmental only':  numerical,
        'Clinical only':       ['symptom_severity', 'exposure_score', 'night_score'],
        'No engineered feats': numerical,
        'Engineered only':     engineered,
        'All features (full)': numerical + engineered,
    }

    def best_model():
        base = [
            ('xgb',  XGBClassifier(n_estimators=100, random_state=RANDOM_STATE,
                                   eval_metric='mlogloss', verbosity=0)),
            ('lgbm', LGBMClassifier(n_estimators=100, random_state=RANDOM_STATE, verbose=-1)),
            ('rf',   RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)),
        ]
        return StackingClassifier(estimators=base,
                                   final_estimator=LogisticRegression(max_iter=300),
                                   cv=3)

    for name, cols in configs.items():
        cols_actual = [c for c in cols if c in df.columns]
        X = df[cols_actual].fillna(0).values
        acc, std, f1 = evaluate_cv(X, y, best_model)
        results.append({'config': name, 'accuracy': acc, 'std': std, 'f1': f1,
                        'n_features': len(cols_actual), 'type': 'feature'})
        print(f"  {name:25s}  {len(cols_actual):2d} feats  Acc={acc*100:.1f}% +/-{std*100:.1f}%  F1={f1:.3f}")

    # ── Base Learner Ablation ─────────────────────────────────────
    print("\n--- BASE LEARNER COUNT ABLATION ---")
    X_full = df[numerical + engineered].fillna(0).values

    for n in [1, 2, 3, 6]:
        if n == 1:
            # Single best model (XGBoost)
            m_fn = lambda: XGBClassifier(n_estimators=100, random_state=RANDOM_STATE,
                                          eval_metric='mlogloss', verbosity=0)
        else:
            m_fn = stacking_n(min(n, 6))()
            # wrap in lambda
            m_fn_copy = stacking_n(min(n, 6))
            acc, std, f1 = evaluate_cv(X_full, y, m_fn_copy)
            results.append({'config': f'Stacking ({n} learners)', 'accuracy': acc,
                            'std': std, 'f1': f1, 'n_features': len(numerical+engineered),
                            'type': 'learner'})
            print(f"  Stacking ({n} learners):           Acc={acc*100:.1f}% +/-{std*100:.1f}%  F1={f1:.3f}")
            continue
        acc, std, f1 = evaluate_cv(X_full, y, m_fn)
        results.append({'config': f'Single (XGBoost)', 'accuracy': acc,
                        'std': std, 'f1': f1,
                        'n_features': len(numerical+engineered), 'type': 'learner'})
        print(f"  Single XGBoost:                Acc={acc*100:.1f}% +/-{std*100:.1f}%  F1={f1:.3f}")

    # Save
    lines = ["AsthmAI Ablation Study Results\n", "=" * 40 + "\n\n",
             "Feature Group Ablation:\n"]
    for r in results:
        if r['type'] == 'feature':
            lines.append(f"  {r['config']:25s}  Acc={r['accuracy']*100:.1f}%+/-{r['std']*100:.1f}%  F1={r['f1']:.3f}\n")
    lines.append("\nBase Learner Count Ablation:\n")
    for r in results:
        if r['type'] == 'learner':
            lines.append(f"  {r['config']:25s}  Acc={r['accuracy']*100:.1f}%+/-{r['std']*100:.1f}%  F1={r['f1']:.3f}\n")

    with open('results/ablation_results.txt', 'w') as f:
        f.writelines(lines)

    print("\nSaved: results/ablation_results.txt")
    return results


if __name__ == "__main__":
    run_ablation()
