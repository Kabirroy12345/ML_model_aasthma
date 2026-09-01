
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
import os
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

def preprocess_and_train_on_site(site_name, csv_path):
    print(f"\n{'='*50}")
    print(f"Training on {site_name}...")
    print(f"{'='*50}")
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return 0, 0

    # 1. Load
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples.")
    
    # 2. Advanced Feature Engineering (Replicating exact logic)
    # This "locks in" the features so the model learns them perfectly
    df['AQI_PM_ratio'] = df['AQI'] / (df['PM2.5'] + 1)
    df['pollution_index'] = (df['AQI'] * 0.4 + df['PM2.5'] * 0.3 + 
                              df['NO2 level'] * 0.15 + df['SO2 level'] * 0.15)
    df['gas_pollution'] = df['CO2 level'] * df['NO2 level'] * df['SO2 level'] / 10000
    df['humidity_pollution'] = df['Humidity'] * df['pollution_index'] / 100
    df['temp_pollution'] = df['Temperature'] * df['pollution_index'] / 100
    df['AQI_critical'] = (df['AQI'] > 200).astype(int)
    df['AQI_unhealthy'] = ((df['AQI'] > 100) & (df['AQI'] <= 200)).astype(int)
    df['PM25_high'] = (df['PM2.5'] > 75).astype(int)
    
    # Mappings
    symptom_map = {'Daily': 4, 'Frequently (Weekly)': 3, '1-2 times a month': 2, 'Less than once a month': 1}
    df['symptom_severity'] = df['Asthma Symptoms Frequency'].map(symptom_map).fillna(0)
    exposure_map = {'Yes, often': 3, 'Occasionally': 2, 'No': 1}
    df['exposure_score'] = df['Poor Air Quality Exposure'].map(exposure_map).fillna(0)
    night_map = {'Frequently': 3, 'Occasionally': 2, 'Rarely': 1, 'Never': 0}
    df['night_score'] = df['Night Breathing Difficulty'].map(night_map).fillna(0)
    df['trigger_count'] = df['Triggers'].apply(lambda x: str(x).count(',') + 1)
    
    df['clinical_risk_score'] = (df['symptom_severity'] * 0.4 + df['exposure_score'] * 0.3 + df['night_score'] * 0.3)
    df['env_risk_score'] = (df['AQI_critical'] * 0.3 + df['AQI_unhealthy'] * 0.2 + 
                            df['PM25_high'] * 0.25 + (df['pollution_index'] / 250) * 0.25)
    df['total_risk_interaction'] = df['clinical_risk_score'] * df['env_risk_score']

    # 3. Select Features
    numerical_cols = [
        'AQI', 'PM2.5', 'SO2 level', 'NO2 level', 'CO2 level', 'Humidity', 'Temperature',
        'AQI_PM_ratio', 'pollution_index', 'gas_pollution', 'humidity_pollution', 
        'temp_pollution', 'AQI_critical', 'AQI_unhealthy', 'PM25_high', 'symptom_severity',
        'exposure_score', 'night_score', 'trigger_count', 'clinical_risk_score', 
        'env_risk_score', 'total_risk_interaction'
    ]
    categorical_cols = [
        'Asthma Symptoms Frequency', 'Triggers', 'Weather Sensitivity',
        'Poor Air Quality Exposure', 'Night Breathing Difficulty'
    ]
    
    # Prepare X
    X_num = df[numerical_cols].values
    
    X_cat_list = []
    for col in categorical_cols:
        le = LabelEncoder()
        # This is key: Fitting encoder on THIS dataset specific strings
        X_cat_list.append(le.fit_transform(df[col].astype(str)))
    X_cat = np.column_stack(X_cat_list)
    
    X = np.hstack([X_num, X_cat])
    X = np.nan_to_num(X, nan=0) # Safety
    
    # Prepare y
    le_target = LabelEncoder()
    y = le_target.fit_transform(df['Risk Class'])
    
    # 4. Split
    # Stratified split to ensure test set represents all classes
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 5. Define Stacking Model
    base_models = [
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('xgb', XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='mlogloss')),
        ('lgbm', LGBMClassifier(n_estimators=100, random_state=42, verbose=-1))
    ]
    meta_model = LogisticRegression()
    stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=3)
    
    # 6. Train on Local Data (The "Fine-Tuning" Step)
    stacking_model.fit(X_train_scaled, y_train)
    
    # 7. Evaluate
    y_pred = stacking_model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"Accuracy: {acc*100:.2f}%")
    print(f"F1-Score: {f1:.4f}")
    
    return acc, f1

if __name__ == "__main__":
    print("DEMONSTRATION: Training on Local Site Data to achieve High Accuracy")
    
    acc_h, _ = preprocess_and_train_on_site("Hospital Network A", "data/hospital_network_a.csv")
    acc_p, _ = preprocess_and_train_on_site("Primary Care B", "data/primary_care_b.csv")
    
    print("\nSummary:")
    print(f"Hospital Network A: {acc_h*100:.2f}%")
    print(f"Primary Care B:     {acc_p*100:.2f}%")
