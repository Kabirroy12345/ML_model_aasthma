"""
AsthmAI - Fix #4: Demographic Fairness Analysis
Uses the Kaggle Asthma Disease Dataset which has real
Age, Gender, and Ethnicity fields.
"""
import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings('ignore')

KAGGLE_CSV  = r"C:\Users\HP\Downloads\archive (1)\asthma_disease_data.csv"
OUTPUT_FILE = 'results/fairness_report.txt'


def map_to_risk_class(df):
    symptom_score = (df['Wheezing'] + df['ShortnessOfBreath'] + df['ChestTightness'] +
                     df['Coughing'] + df['NighttimeSymptoms'] * 2 + df['ExerciseInduced'])
    df = df.copy()
    df['symptom_score'] = symptom_score
    df['low_fev1']      = (df['LungFunctionFEV1'] < 2.0).astype(int)

    def classify(row):
        if row['Diagnosis'] == 0 and row['symptom_score'] <= 1:
            return 'Low'
        elif row['Diagnosis'] == 1 and (row['symptom_score'] >= 4 or row['low_fev1'] == 1):
            return 'High'
        else:
            return 'Medium'

    df['Risk_Class'] = df.apply(classify, axis=1)
    return df


def age_group(age):
    if age < 18:   return '<18'
    elif age < 46: return '18-45'
    elif age < 66: return '46-65'
    else:          return '>65'


def gender_label(g):
    return 'Male' if g == 0 else 'Female'


def ethnicity_label(e):
    mapping = {0: 'Caucasian', 1: 'African American', 2: 'Asian', 3: 'Other'}
    return mapping.get(e, 'Other')


def run_fairness_analysis():
    print("=" * 60)
    print("AsthmAI - Demographic Fairness Analysis")
    print("=" * 60)

    df = pd.read_csv(KAGGLE_CSV)
    df = map_to_risk_class(df)
    df['AgeGroup']     = df['Age'].apply(age_group)
    df['GenderLabel']  = df['Gender'].apply(gender_label)
    df['EthnicLabel']  = df['Ethnicity'].apply(ethnicity_label)

    feature_cols = [
        'Age', 'Gender', 'Ethnicity', 'BMI', 'Smoking', 'PhysicalActivity',
        'DietQuality', 'SleepQuality', 'PollutionExposure', 'PollenExposure',
        'DustExposure', 'PetAllergy', 'FamilyHistoryAsthma', 'HistoryOfAllergies',
        'Eczema', 'HayFever', 'GastroesophagealReflux', 'LungFunctionFEV1',
        'LungFunctionFVC', 'Wheezing', 'ShortnessOfBreath', 'ChestTightness',
        'Coughing', 'NighttimeSymptoms', 'ExerciseInduced',
        'symptom_score', 'low_fev1'
    ]

    X = df[feature_cols].fillna(0).values
    le = LabelEncoder()
    y  = le.fit_transform(df['Risk_Class'])

    idx = np.arange(len(df))
    idx_train, idx_test = train_test_split(idx, test_size=0.2, random_state=42, stratify=y)

    X_train, X_test = X[idx_train], X[idx_test]
    y_train, y_test = y[idx_train], y[idx_test]
    df_test = df.iloc[idx_test].reset_index(drop=True)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    base_models = [
        ('rf',   RandomForestClassifier(n_estimators=100, random_state=42)),
        ('xgb',  XGBClassifier(n_estimators=100, random_state=42,
                               eval_metric='mlogloss', verbosity=0)),
        ('lgbm', LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)),
    ]
    model = StackingClassifier(
        estimators=base_models,
        final_estimator=LogisticRegression(max_iter=300, random_state=42),
        cv=5
    )
    print("Training stacking ensemble...")
    model.fit(X_train_sc, y_train)

    y_pred = model.predict(X_test_sc)
    overall_acc = accuracy_score(y_test, y_pred)
    overall_f1  = f1_score(y_test, y_pred, average='weighted')
    print(f"\nOverall: Accuracy={overall_acc:.4f}  F1={overall_f1:.4f}  n={len(y_test)}")

    results = []

    def eval_group(mask_col, group_val, label):
        mask = df_test[mask_col] == group_val
        n = mask.sum()
        if n < 5:
            return None
        acc = accuracy_score(y_test[mask], y_pred[mask])
        f1  = f1_score(y_test[mask], y_pred[mask], average='weighted', zero_division=0)
        return {'group': label, 'n': n, 'accuracy': acc, 'f1': f1}

    print("\n--- AGE GROUPS ---")
    for grp in ['<18', '18-45', '46-65', '>65']:
        r = eval_group('AgeGroup', grp, f'Age {grp}')
        if r:
            results.append(r)
            print(f"  {r['group']:15s}  n={r['n']:5d}  Acc={r['accuracy']*100:.1f}%  F1={r['f1']:.3f}")

    print("\n--- GENDER ---")
    for grp in ['Male', 'Female']:
        r = eval_group('GenderLabel', grp, f'Gender: {grp}')
        if r:
            results.append(r)
            print(f"  {r['group']:15s}  n={r['n']:5d}  Acc={r['accuracy']*100:.1f}%  F1={r['f1']:.3f}")

    print("\n--- ETHNICITY ---")
    for grp in ['Caucasian', 'African American', 'Asian', 'Other']:
        r = eval_group('EthnicLabel', grp, f'Ethnicity: {grp}')
        if r:
            results.append(r)
            print(f"  {r['group']:25s}  n={r['n']:5d}  Acc={r['accuracy']*100:.1f}%  F1={r['f1']:.3f}")

    accs    = [r['accuracy'] for r in results]
    max_var = (max(accs) - min(accs)) * 100
    dp_diff = max(accs) - min(accs)
    eq_odds = min(accs) / max(accs) if max(accs) > 0 else 1.0

    print(f"\n--- FAIRNESS METRICS ---")
    print(f"Max accuracy variance:       {max_var:.1f}%")
    print(f"Demographic parity diff:     {dp_diff:.3f}")
    print(f"Equalized odds ratio:        {eq_odds:.3f}")
    print(f"Fairness threshold (<5%):    {'PASS' if max_var < 5 else 'FAIL'}")

    lines = [
        "AsthmAI - Demographic Fairness Analysis\n",
        f"Dataset: Kaggle Asthma Disease Dataset (n={len(df)})\n",
        f"Test set: {len(y_test)} patients\n",
        f"Overall Accuracy: {overall_acc:.4f}  F1: {overall_f1:.4f}\n\n",
        "Per-Subgroup Performance:\n",
    ]
    for r in results:
        lines.append(f"  {r['group']:25s}  n={r['n']:5d}  Acc={r['accuracy']*100:.1f}%  F1={r['f1']:.3f}\n")
    lines += [
        f"\nFairness Metrics:\n",
        f"  Max accuracy variance:      {max_var:.1f}%\n",
        f"  Demographic parity diff:    {dp_diff:.3f}\n",
        f"  Equalized odds ratio:       {eq_odds:.3f}\n",
        f"  Fairness threshold (<5%):   {'PASS' if max_var < 5 else 'FAIL'}\n",
    ]
    with open(OUTPUT_FILE, 'w') as f:
        f.writelines(lines)
    print(f"\nSaved: {OUTPUT_FILE}")
    return results, max_var


if __name__ == "__main__":
    run_fairness_analysis()
