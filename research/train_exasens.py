import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score

# 1. Load Data (Exasens)
# Diagnosis,ID,Imaginary Part Min,Imaginary Part Avg,Real Part Min,Real Part Avg,Gender,Age,Smoking
df = pd.read_csv('exasens.csv')

# Clean columns - sometimes there are many empty columns at the end
df = df.iloc[:, :9]
df.columns = ['Diagnosis', 'ID', 'Imaginary_Min', 'Imaginary_Avg', 'Real_Min', 'Real_Avg', 'Gender', 'Age', 'Smoking']

# Clean missing values
df = df.dropna(subset=['Diagnosis', 'Imaginary_Avg', 'Real_Avg', 'Age'])

# 2. Features
features = ['Imaginary_Avg', 'Real_Avg', 'Gender', 'Age', 'Smoking']
X = df[features].copy()

# Fill missing Smoking with mode
X['Smoking'] = X['Smoking'].fillna(X['Smoking'].mode()[0])

# Target
le = LabelEncoder()
y = le.fit_transform(df['Diagnosis'])

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Encode Categorical if any (all are numerical/binary here)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Define Models
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
print("Training Stacking Ensemble on UCI Exasens (Real Clinical Data)...")
stacking_model.fit(X_train_scaled, y_train)
y_pred = stacking_model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='macro')

print(f"\n--- FINAL SCORES (EXASENS REAL DATA) ---")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1-Score: {f1:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Save results for paper
with open('exasens_benchmark.txt', 'w') as f:
    f.write(f"Dataset: UCI Exasens (Real Clinical Data)\n")
    f.write(f"Samples: {len(df)}\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"F1-Score: {f1:.4f}\n")
    f.write("\nClassification Report:\n")
    f.write(classification_report(y_test, y_pred, target_names=le.classes_))
