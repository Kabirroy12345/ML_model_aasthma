
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load test data
data_dir = 'data'
test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))

# Load model and encoders
with open('results/best_ensemble_model.pkl', 'rb') as f:
    saved = pickle.load(f)
    model = saved['model']
    scaler = saved['scaler']
    label_encoder_target = saved['label_encoder']

# Define Clinical Heuristic Logic (Safety Layer)
def clinical_heuristics(row):
    symptom_freq = row['Asthma Symptoms Frequency']
    if symptom_freq == "Daily":
        return "High", True
    elif symptom_freq == "Frequently (Weekly)":
        return "Medium", True
    return None, False

# Feature Engineering (Matching ensemble_model.py)
def engineer_features(df):
    df = df.copy()
    df['AQI_PM_ratio'] = df['AQI'] / (df['PM2.5'] + 1)
    df['pollution_index'] = (df['AQI'] * 0.4 + df['PM2.5'] * 0.3 + 
                              df['NO2 level'] * 0.15 + df['SO2 level'] * 0.15)
    df['gas_pollution'] = df['CO2 level'] * df['NO2 level'] * df['SO2 level'] / 10000
    df['humidity_pollution'] = df['Humidity'] * df['pollution_index'] / 100
    df['temp_pollution'] = df['Temperature'] * df['pollution_index'] / 100
    df['AQI_critical'] = (df['AQI'] > 200).astype(int)
    df['AQI_unhealthy'] = ((df['AQI'] > 100) & (df['AQI'] <= 200)).astype(int)
    df['PM25_high'] = (df['PM2.5'] > 75).astype(int)
    
    symptom_map = {'Daily': 4, 'Frequently (Weekly)': 3, '1-2 times a month': 2, 'Less than once a month': 1}
    df['symptom_severity'] = df['Asthma Symptoms Frequency'].map(symptom_map).fillna(0)
    exposure_map = {'Yes, often': 3, 'Occasionally': 2, 'No': 1}
    df['exposure_score'] = df['Poor Air Quality Exposure'].map(exposure_map).fillna(0)
    night_map = {'Frequently': 3, 'Occasionally': 2, 'Rarely': 1, 'Never': 0}
    df['night_score'] = df['Night Breathing Difficulty'].map(night_map).fillna(0)
    df['trigger_count'] = df['Triggers'].apply(lambda x: str(x).count(',') + 1)
    
    df['clinical_risk_score'] = (df['symptom_severity'] * 0.4 + df['exposure_score'] * 0.3 + df['night_score'] * 0.3)
    df['env_risk_score'] = (df['AQI_critical'] * 0.3 + df['AQI_unhealthy'] * 0.2 + 
                            df['PM25_high'] * 0.25 + (df['pollution_index'] / df['pollution_index'].max()) * 0.25)
    df['total_risk_interaction'] = df['clinical_risk_score'] * df['env_risk_score']
    
    return df

# 1. Prepare ML Input (Batch Processing to get Encodings correct)
df_eng = engineer_features(test_df)

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

X_num = df_eng[numerical_cols].values

# Correct Encoding Strategy: Fit LabelEncoder on the column values
# (Replicates the behavior of ensemble_model.py's prepare_features for test set)
X_cat_list = []
for col in categorical_cols:
    le = LabelEncoder()
    # Fit on the data itself to generate consistent integers for this set
    # (Assuming the test set contains similar lexical range as training)
    X_cat_list.append(le.fit_transform(df_eng[col].astype(str)))

X_cat = np.column_stack(X_cat_list)
X = np.hstack([X_num, X_cat])
X = np.nan_to_num(X, nan=0)

# Scale
X_scaled = scaler.transform(X)

# 2. Get Pure ML Predictions
ml_pred_indices = model.predict(X_scaled)
ml_preds = label_encoder_target.inverse_transform(ml_pred_indices)
ml_probs = model.predict_proba(X_scaled) if hasattr(model, "predict_proba") else None

# 3. Apply/Verify Hybrid Logic
correct_hybrid = 0
heuristic_triggers = 0
total = len(test_df)

for i in range(total):
    actual = test_df.iloc[i]['Risk Class']
    
    # Check Heuristic
    heuristic_pred, triggered = clinical_heuristics(test_df.iloc[i])
    
    final_pred = None
    if triggered:
        heuristic_triggers += 1
        final_pred = heuristic_pred
    else:
        final_pred = ml_preds[i]
        
    if final_pred == actual:
        correct_hybrid += 1

hybrid_accuracy = (correct_hybrid / total) * 100

print(f"Total Samples: {total}")
print(f"Heuristic Triggers: {heuristic_triggers} ({(heuristic_triggers/total)*100:.1f}%)")
print(f"Hybrid Correct: {correct_hybrid}")
print(f"Hybrid Accuracy: {hybrid_accuracy:.2f}%")

# Save detailed results
with open('results/hybrid_verification.txt', 'w') as f:
    f.write(f"Hybrid System Verification (RECTIFIED)\n")
    f.write(f"======================================\n")
    f.write(f"Total Samples: {total}\n")
    f.write(f"Heuristic Triggers (Safety Layer): {heuristic_triggers}\n")
    f.write(f"Hybrid Accuracy: {hybrid_accuracy:.2f}%\n")
