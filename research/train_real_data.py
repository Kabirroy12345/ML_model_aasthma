import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score

# 1. Load Data
df = pd.read_csv('real_asthma_data.csv')

# 2. Preprocess
# Features: Environmental factors
features = ['PM25_lag0', 'PM10_lag0', 'SO2_lag0', 'NO2_lag0', 'O3_lag0', 'CO_lag0', 'temp_lag0', 'humid_lag0']
X = df[features].copy()

# Target: Asthma (Total cases) - Bin into 3 Risk Classes (Low, Med, High)
# Using quantiles for balanced classes
df['Risk_Class'] = pd.qcut(df['Asthma'], q=3, labels=['Low', 'Medium', 'High'])
le = LabelEncoder()
y = le.fit_transform(df['Risk_Class'])

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Define Models (Same as our project architecture)
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

# 4. Train & Evaluate
print("Training Stacking Ensemble on Real Beijing Data...")
stacking_model.fit(X_train_scaled, y_train)
y_pred = stacking_model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='macro')

print(f"\n--- FINAL SCORES (REAL DATA) ---")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1-Score: {f1:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Save results for paper
with open('real_data_benchmark.txt', 'w') as f:
    f.write(f"Dataset: Beijing Weather & Health (Real World)\n")
    f.write(f"Samples: {len(df)}\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"F1-Score: {f1:.4f}\n")
    f.write("\nClassification Report:\n")
    f.write(classification_report(y_test, y_pred, target_names=le.classes_))
