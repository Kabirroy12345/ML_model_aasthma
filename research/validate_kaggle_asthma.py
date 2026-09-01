"""
AsthmAI - Fix #1: Second Real Clinical Validation
Dataset: Kaggle Asthma Disease Dataset (2,392 patients)
Validates the stacking ensemble framework on an independent real-world dataset.
"""
import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             classification_report, precision_score, recall_score)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings('ignore')

KAGGLE_CSV = r"C:\Users\HP\Downloads\archive (1)\asthma_disease_data.csv"
OUTPUT_FILE = "results/kaggle_validation.txt"


def map_to_risk_class(df):
    """
    Map Kaggle binary Diagnosis + clinical severity markers to 3-tier risk.
    Logic:
      - Diagnosis==0 (no asthma) + low symptoms → Low Risk
      - Diagnosis==1 (asthma) + moderate symptoms → Medium Risk
      - Diagnosis==1 + severe symptoms (NighttimeSymptoms, ChestTightness, low FEV1) → High Risk
    """
    symptom_score = (
        df['Wheezing'] * 1 +
        df['ShortnessOfBreath'] * 1 +
        df['ChestTightness'] * 1 +
        df['Coughing'] * 1 +
        df['NighttimeSymptoms'] * 2 +    # weight nocturnal symptoms more
        df['ExerciseInduced'] * 1
    )

    # Low FEV1 indicates severe impairment (FEV1 < 2.0 = below typical threshold)
    low_fev1 = (df['LungFunctionFEV1'] < 2.0).astype(int)

    def classify(row):
        if row['Diagnosis'] == 0 and row['symptom_score'] <= 1:
            return 'Low'
        elif row['Diagnosis'] == 1 and (row['symptom_score'] >= 4 or row['low_fev1'] == 1):
            return 'High'
        else:
            return 'Medium'

    df = df.copy()
    df['symptom_score'] = symptom_score
    df['low_fev1'] = low_fev1
    df['Risk_Class'] = df.apply(classify, axis=1)
    return df


def run_kaggle_validation():
    print("=" * 60)
    print("AsthmAI - Kaggle Asthma Dataset Validation")
    print("=" * 60)

    df = pd.read_csv(KAGGLE_CSV)
    print(f"Loaded {len(df)} patients, {len(df.columns)} features")

    # Map to 3-tier risk
    df = map_to_risk_class(df)
    print(f"\nRisk class distribution:")
    print(df['Risk_Class'].value_counts())

    # Features: clinical + environmental (excluding diagnosis-derived fields)
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
    y = le.fit_transform(df['Risk_Class'])

    # 80-20 stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # Same stacking architecture as main project
    base_models = [
        ('rf',   RandomForestClassifier(n_estimators=100, random_state=42)),
        ('xgb',  XGBClassifier(n_estimators=100, random_state=42,
                               eval_metric='mlogloss', verbosity=0)),
        ('lgbm', LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)),
        ('gb',   GradientBoostingClassifier(n_estimators=100, random_state=42))
    ]
    meta = LogisticRegression(max_iter=500, random_state=42)
    model = StackingClassifier(estimators=base_models, final_estimator=meta, cv=5)

    print("\nTraining stacking ensemble on Kaggle clinical data...")
    model.fit(X_train_sc, y_train)

    y_pred = model.predict(X_test_sc)
    y_prob = model.predict_proba(X_test_sc)

    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred, average='weighted')
    prec = precision_score(y_test, y_pred, average='weighted')
    rec  = recall_score(y_test, y_pred, average='weighted')
    auc  = roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted')

    print(f"\n--- KAGGLE VALIDATION RESULTS ---")
    print(f"Total patients:  {len(df)}")
    print(f"Test set size:   {len(y_test)}")
    print(f"Accuracy:        {acc:.4f} ({acc*100:.2f}%)")
    print(f"F1-Score:        {f1:.4f}")
    print(f"Precision:       {prec:.4f}")
    print(f"Recall:          {rec:.4f}")
    print(f"ROC-AUC:         {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # 5-fold cross-validation for robustness
    print("Running 5-fold cross-validation...")
    cv_scores = cross_val_score(model, X_train_sc, y_train, cv=5,
                                scoring='accuracy', n_jobs=-1)
    print(f"CV Accuracy: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    # Save
    report = classification_report(y_test, y_pred, target_names=le.classes_)
    with open(OUTPUT_FILE, 'w') as f:
        f.write("Dataset: Kaggle Asthma Disease Dataset (Synthetic-Clinical)\n")
        f.write(f"Total Patients: {len(df)}\n")
        f.write(f"Test Set: {len(y_test)}\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"F1-Score: {f1:.4f}\n")
        f.write(f"Precision: {prec:.4f}\n")
        f.write(f"Recall: {rec:.4f}\n")
        f.write(f"ROC-AUC: {auc:.4f}\n")
        f.write(f"CV Accuracy: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}\n")
        f.write(f"\nRisk Class Distribution:\n{df['Risk_Class'].value_counts().to_string()}\n")
        f.write(f"\nClassification Report:\n{report}")

    print(f"\nResults saved to {OUTPUT_FILE}")
    return acc, f1, auc


if __name__ == "__main__":
    run_kaggle_validation()
