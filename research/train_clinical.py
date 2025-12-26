import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score

# 1. Load Data (Zenodo Clinical Dataset)
df = pd.read_csv('real_clinical_data.csv')

# Clean columns - strip potential quotes/spaces
df.columns = df.columns.str.strip().str.replace("'", "").str.replace('"', '')

# 2. Map ACTScore to Risk Classes
# ACT 25: Completely controlled (Low Risk)
# ACT 20-24: Well controlled (Medium Risk)
# ACT <20: Uncontrolled (High Risk)
def map_act_to_risk(score):
    try:
        score = float(score)
    except:
        return 'Medium'
    if score >= 25: return 'Low'
    elif score >= 20: return 'Medium'
    else: return 'High'

df['Risk_Class'] = df['ACTScore'].apply(map_act_to_risk)

# 3. Preprocess
df = df.reset_index(drop=True)
features = ['Age', 'Gender', 'Status', 'Humidity', 'Temperature', 'Pressure', 'UVIndex', 'WindSpeed']

# Clean Age (handle ranges like '19-30')
def clean_age(age):
    if isinstance(age, str):
        if '-' in age:
            parts = age.split('-')
            try:
                return (float(parts[0]) + float(parts[1])) / 2
            except:
                return float(parts[0])
        try:
            return float(age)
        except:
            return 30.0 # Default
    return float(age)

df['Age'] = df['Age'].apply(clean_age)

# Use a safer selection
X = df.reindex(columns=features).copy()

# Simple Encoding
X['Gender'] = X['Gender'].map({'Male': 0, 'Female': 1}).fillna(0)
le_status = LabelEncoder()
X['Status'] = le_status.fit_transform(X['Status'].astype(str))

# Convert everything to numeric, forcing errors to NaN then filling
for col in ['Humidity', 'Temperature', 'Pressure', 'UVIndex', 'WindSpeed']:
    X[col] = pd.to_numeric(X[col], errors='coerce')

# Handle any missing values
X = X.fillna(X.median(numeric_only=True))

# Target
le_risk = LabelEncoder()
y = le_risk.fit_transform(df['Risk_Class'])

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Define Models (Same as our project architecture)
base_models = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('xgb', XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='mlogloss')),
    ('lgbm', LGBMClassifier(n_estimators=100, random_state=42))
]

meta_model = LogisticRegression()

stacking_model = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5
)

# 5. Train & Evaluate
print("Training Stacking Ensemble on REAL Zenith Clinical Data...")
stacking_model.fit(X_train_scaled, y_train)
y_pred = stacking_model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='macro')

print(f"\n--- FINAL SCORES (REAL CLINICAL DATA) ---")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1-Score: {f1:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le_risk.classes_))

# Save results for paper
with open('clinical_benchmark.txt', 'w') as f:
    f.write(f"Dataset: Zenodo Clinical Asthma (Radiah Haque et al.)\n")
    f.write(f"Samples: {len(df)}\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"F1-Score: {f1:.4f}\n")
    f.write("\nClassification Report:\n")
    f.write(classification_report(y_test, y_pred, target_names=le_risk.classes_))
