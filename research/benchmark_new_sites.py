
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report

def advanced_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Create advanced features to match training pipeline (Copied from ensemble_model.py)."""
    df = df.copy()
    
    # Pollution interaction features
    df['AQI_PM_ratio'] = df['AQI'] / (df['PM2.5'] + 1)
    df['pollution_index'] = (df['AQI'] * 0.4 + df['PM2.5'] * 0.3 + 
                              df['NO2 level'] * 0.15 + df['SO2 level'] * 0.15)
    df['gas_pollution'] = df['CO2 level'] * df['NO2 level'] * df['SO2 level'] / 10000
    
    # Weather-pollution interactions
    df['humidity_pollution'] = df['Humidity'] * df['pollution_index'] / 100
    df['temp_pollution'] = df['Temperature'] * df['pollution_index'] / 100
    
    # AQI categories
    df['AQI_critical'] = (df['AQI'] > 200).astype(int)
    df['AQI_unhealthy'] = ((df['AQI'] > 100) & (df['AQI'] <= 200)).astype(int)
    df['PM25_high'] = (df['PM2.5'] > 75).astype(int)
    
    # Symptom severity score
    symptom_map = {
        'Daily': 4, 'Frequently (Weekly)': 3, 
        '1-2 times a month': 2, 'Less than once a month': 1
    }
    df['symptom_severity'] = df['Asthma Symptoms Frequency'].map(symptom_map).fillna(0)
    
    # Exposure risk score
    exposure_map = {'Yes, often': 3, 'Occasionally': 2, 'No': 1}
    df['exposure_score'] = df['Poor Air Quality Exposure'].map(exposure_map).fillna(0)
    
    # Night symptoms score
    night_map = {'Frequently': 3, 'Occasionally': 2, 'Rarely': 1, 'Never': 0}
    df['night_score'] = df['Night Breathing Difficulty'].map(night_map).fillna(0)
    
    # Trigger count
    df['trigger_count'] = df['Triggers'].apply(lambda x: str(x).count(',') + 1)
    
    # Combined clinical scores
    df['clinical_risk_score'] = (
        df['symptom_severity'] * 0.4 + 
        df['exposure_score'] * 0.3 + 
        df['night_score'] * 0.3
    )
    
    df['env_risk_score'] = (
        df['AQI_critical'] * 0.3 +
        df['AQI_unhealthy'] * 0.2 +
        df['PM25_high'] * 0.25 +
        (df['pollution_index'] / df['pollution_index'].max()) * 0.25
    )
    
    df['total_risk_interaction'] = df['clinical_risk_score'] * df['env_risk_score']
    
    return df

def prepare_features(df, numerical_cols, categorical_cols, scaler, fit_encoders=False):
    # Apply engineering
    df = advanced_feature_engineering(df)
    
    engineered_cols = [
        'AQI_PM_ratio', 'pollution_index', 'gas_pollution',
        'humidity_pollution', 'temp_pollution', 'AQI_critical',
        'AQI_unhealthy', 'PM25_high', 'symptom_severity',
        'exposure_score', 'night_score', 'trigger_count',
        'clinical_risk_score', 'env_risk_score', 'total_risk_interaction'
    ]
    
    all_numerical = numerical_cols + engineered_cols
    X_num = df[all_numerical].values
    
    # Encode categorical (Mimicking the "fit per dataset" logic of the original code)
    X_cat_list = []
    for col in categorical_cols:
        encoder = LabelEncoder()
        # In original code, for test set, it fit on the test set itself!
        encoder.fit(df[col].astype(str))
        X_cat_list.append(encoder.transform(df[col].astype(str)))
    
    X_cat = np.column_stack(X_cat_list)
    X = np.hstack([X_num, X_cat])
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    
    if scaler:
        X = scaler.transform(X)
        
    return X

def load_and_evaluate(site_name, csv_path):
    print(f"\nEvaluating {site_name}...")
    
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    
    # Load model
    with open('results/best_ensemble_model.pkl', 'rb') as f:
        saved = pickle.load(f)
        model = saved['model']
        scaler = saved['scaler']
        label_encoder_target = saved['label_encoder']

    # Config (from class)
    numerical_cols = [
        'AQI', 'PM2.5', 'SO2 level', 'NO2 level', 
        'CO2 level', 'Humidity', 'Temperature'
    ]
    categorical_cols = [
        'Asthma Symptoms Frequency', 'Triggers', 'Weather Sensitivity',
        'Poor Air Quality Exposure', 'Night Breathing Difficulty'
    ]
    
    # Prepare
    X = prepare_features(df, numerical_cols, categorical_cols, scaler)
    y_true = label_encoder_target.transform(df['Risk Class'])
    
    # Predict
    y_pred = model.predict(X)
    
    # Metrics
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    print(f"Samples: {len(df)}")
    print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"F1-Score: {f1:.4f}")
    print(classification_report(y_true, y_pred, target_names=label_encoder_target.classes_))
    
    return acc, f1

if __name__ == "__main__":
    print(" Benchmarking New Sites...")
    load_and_evaluate("Hospital Network A", "data/hospital_network_a.csv")
    load_and_evaluate("Primary Care B", "data/primary_care_b.csv")
