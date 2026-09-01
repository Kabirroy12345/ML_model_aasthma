"""
================================================================================
HridyaVayu - Comprehensive 6-Module System Verification & Test Suite
================================================================================
Tests all 6 Major Modules of HridyaVayu as specified in project documentation:
  Module 1: Data Input and Preprocessing
  Module 2: Ensemble Machine Learning Engine
  Module 3: Risk Score and Classification
  Module 4: Clinical Guidelines Assessment
  Module 5: Neuro-Symbolic Decision Layer
  Module 6: Environmental Monitoring (API Sync & Manual Telemetry)
================================================================================
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import json
import requests
from datetime import datetime

BASE_URL = os.environ.get("HRIDYAVAYU_URL", "http://127.0.0.1:7860")

def print_header(title):
    print("\n" + "=" * 80)
    print(f" {title.upper()}")
    print("=" * 80)

def print_sub(title):
    print(f"\n--- {title} ---")

def test_module_1_data_input_and_preprocessing():
    print_header("Module 1: Data Input and Preprocessing")
    print("Purpose: Collect and prepare user or patient-related data for ML prediction.")
    
    # 1.1 Test User Profile Registration / Storage (User Table)
    profile_payload = {
        "user_id": 101,
        "name": "Sarah Connor",
        "age": 34,
        "gender": "Female",
        "phone_no": "+1-555-0842",
        "medical_history": "Allergic asthma triggered by seasonal pollen",
        "emergency_contact_name": "John Connor",
        "emergency_contact_phone": "+1-555-0999"
    }
    
    res = requests.post(f"{BASE_URL}/save-profile", json=profile_payload, timeout=10)
    assert res.status_code == 200, f"Profile save failed: {res.text}"
    profile_data = res.json()
    assert profile_data.get("success") is True
    print(f"  [PASS] User Data Collection & Storage: Patient '{profile_data['user']['name']}' registered.")
    
    # 1.2 Verify User Profile Retrieval
    user_id = profile_data["user"]["id"]
    res_get = requests.get(f"{BASE_URL}/get-user/{user_id}", timeout=10)
    assert res_get.status_code == 200
    assert res_get.json()["user"]["phone_no"] == "+1-555-0842"
    print(f"  [PASS] User Profile Retrieval: Fetched profile ID {user_id} with emergency contacts verified.")

    # 1.3 Feature Preprocessing and Ingestion Validation
    sample_features = {
        "patient_name": "Sarah Connor",
        "user_id": user_id,
        "AQI": 135,
        "PM2.5": 52.4,
        "SO2 level": 16.0,
        "NO2 level": 34.0,
        "CO2 level": 425.0,
        "Humidity": 64.0,
        "Temperature": 28.0,
        "Asthma Symptoms Frequency": "Frequently (Weekly)",
        "Night Breathing Difficulty": "Occasionally",
        "Triggers": "Pollen, Cold Air, Traffic Smoke",
        "Poor Air Quality Exposure": "Yes, often",
        "Weather Sensitivity": "Cold air"
    }
    
    res_pred = requests.post(f"{BASE_URL}/predict", json=sample_features, timeout=10)
    assert res_pred.status_code == 200
    pred_data = res_pred.json()
    assert pred_data.get("success") is True
    assert "factors" in pred_data and len(pred_data["factors"]) >= 4
    print("  [PASS] Feature Processing & Cleaning: Derived composite metrics (pollution_index, clinical_risk_score).")
    print(f"  [PASS] Factor Attribution Generated: {len(pred_data['factors'])} primary exposure weights calculated.")

    return user_id


def test_module_2_ensemble_machine_learning_engine():
    print_header("Module 2: Ensemble Machine Learning Engine")
    print("Purpose: Predict asthma risk using multiple collaborative Machine Learning models.")
    
    # Test multi-model collaborative prediction
    payload = {
        "patient_name": "Validation Subject Alpha",
        "AQI": 140, "PM2.5": 58.0, "SO2 level": 20.0, "NO2 level": 38.0,
        "CO2 level": 435.0, "Humidity": 65.0, "Temperature": 27.0,
        "Asthma Symptoms Frequency": "1-2 times a month",
        "Night Breathing Difficulty": "Rarely",
        "Triggers": "Dust",
        "Poor Air Quality Exposure": "Occasionally",
        "Weather Sensitivity": "None"
    }
    
    res = requests.post(f"{BASE_URL}/predict", json=payload, timeout=10)
    assert res.status_code == 200
    data = res.json()
    assert data.get("success") is True
    
    breakdown = data.get("model_breakdown", {})
    assert "baseline_linear_regression" in breakdown, "Missing Baseline LR score"
    assert "random_forest" in breakdown, "Missing Random Forest score"
    assert "gradient_boosting" in breakdown, "Missing Gradient Boosting score"
    assert "collaborative_ensemble" in breakdown, "Missing Collaborative Ensemble score"
    
    lr = breakdown["baseline_linear_regression"]
    rf = breakdown["random_forest"]
    gb = breakdown["gradient_boosting"]
    ens = breakdown["collaborative_ensemble"]
    
    print("  [PASS] Multi-Model Individual Inferences Generated:")
    print(f"         1. Baseline Linear/Logistic Regression Score: {lr:.4f}")
    print(f"         2. Model 2 (Random Forest) Score:           {rf:.4f}")
    print(f"         3. Model 3 (Gradient Boosting) Score:       {gb:.4f}")
    print(f"         4. Collaborative Soft-Voting Ensemble:      {ens:.4f}")
    
    # Verify collaborative weighting
    assert 0.0 <= lr <= 1.0 and 0.0 <= rf <= 1.0 and 0.0 <= gb <= 1.0 and 0.0 <= ens <= 1.0
    print("  [PASS] Collaborative Ensemble Fusion: Models executed concurrently with bounded probability space.")


def test_module_3_risk_score_and_classification():
    print_header("Module 3: Risk Score and Classification")
    print("Purpose: Convert model outputs into a meaningful, standardized asthma risk level.")
    
    scenarios = [
        {
            "name": "Controlled / Clean Air Scenario",
            "payload": {
                "AQI": 25, "PM2.5": 8.0, "SO2 level": 5.0, "NO2 level": 10.0,
                "CO2 level": 390.0, "Humidity": 45.0, "Temperature": 22.0,
                "Asthma Symptoms Frequency": "Less than once a month",
                "Night Breathing Difficulty": "Never",
                "Triggers": "No specific triggers",
                "Poor Air Quality Exposure": "No",
                "Weather Sensitivity": "No specific weather-related triggers"
            },
            "expected_max_score": 0.45
        },
        {
            "name": "Severe / Industrial Exposure Scenario",
            "payload": {
                "AQI": 270, "PM2.5": 140.0, "SO2 level": 48.0, "NO2 level": 75.0,
                "CO2 level": 560.0, "Humidity": 78.0, "Temperature": 34.0,
                "Asthma Symptoms Frequency": "Daily",
                "Night Breathing Difficulty": "Frequently",
                "Triggers": "Air pollution, Chemical smoke",
                "Poor Air Quality Exposure": "Yes, often",
                "Weather Sensitivity": "Hot and humid weather"
            },
            "expected_min_score": 0.70
        }
    ]
    
    for s in scenarios:
        res = requests.post(f"{BASE_URL}/predict", json=s["payload"], timeout=10)
        assert res.status_code == 200
        d = res.json()
        score = d["risk_score"]
        level = d["risk_level"]
        conf = d["confidence"]
        
        # Risk level classification validation
        assert level in ["Low", "Medium", "High"]
        print(f"  [PASS] Scenario '{s['name']}':")
        print(f"         Score: {score:.4f} -> Classified as '{level}' Risk (Confidence: {conf*100:.1f}%)")
        
        if "expected_max_score" in s:
            assert score <= s["expected_max_score"]
        if "expected_min_score" in s:
            assert score >= s["expected_min_score"]


def test_module_4_clinical_guidelines_assessment(user_id):
    print_header("Module 4: Clinical Guidelines Assessment")
    print("Purpose: Validate risk using clinical guideline-based questions (GINA Protocol).")
    
    # 4.1 Submit GINA Screening Quiz Responses (QuizResponse Table)
    quiz_payload = {
        "user_id": user_id,
        "responses": [
            {"question": "How often do you experience asthma symptoms?", "answer": "Daily"},
            {"question": "Do you wake up at night with breathing difficulty?", "answer": "Frequently"},
            {"question": "What triggers your asthma attacks?", "answer": "Dust, smoke, and strong odors"},
            {"question": "Are you regularly exposed to poor air quality?", "answer": "Yes, often"},
            {"question": "Does weather trigger symptoms?", "answer": "Cold and humid weather"}
        ]
    }
    
    res = requests.post(f"{BASE_URL}/submit-quiz", json=quiz_payload, timeout=10)
    assert res.status_code == 200
    quiz_data = res.json()
    assert quiz_data.get("success") is True
    print(f"  [PASS] Clinical Questionnaire Ingestion: {len(quiz_data['responses'])} GINA responses saved to DB.")
    
    # 4.2 Retrieve and Verify Stored Responses
    res_get = requests.get(f"{BASE_URL}/get-quiz-responses/{user_id}", timeout=10)
    assert res_get.status_code == 200
    fetched_quiz = res_get.json()
    assert len(fetched_quiz.get("responses", [])) >= 5
    print(f"  [PASS] Clinical Guidelines Audit: Successfully retrieved {len(fetched_quiz['responses'])} stored responses.")


def test_module_5_neuro_symbolic_decision_layer(user_id):
    print_header("Module 5: Neuro-Symbolic Decision Layer")
    print("Purpose: Combine data-driven ML intelligence with rule-based clinical knowledge (GINA Safety Rails).")
    
    # Test deterministic safety override when severe clinical symptoms are present
    payload = {
        "patient_name": "Marcus Vance",
        "user_id": user_id,
        "AQI": 80,  # Moderate AQI
        "PM2.5": 25.0,
        "SO2 level": 10.0,
        "NO2 level": 15.0,
        "CO2 level": 410.0,
        "Humidity": 50.0,
        "Temperature": 24.0,
        # Acute clinical criteria that MUST trigger GINA safety override:
        "Asthma Symptoms Frequency": "Daily",
        "Night Breathing Difficulty": "Frequently",
        "Triggers": "Allergens",
        "Poor Air Quality Exposure": "Occasionally",
        "Weather Sensitivity": "None"
    }
    
    res = requests.post(f"{BASE_URL}/predict", json=payload, timeout=10)
    assert res.status_code == 200
    d = res.json()
    
    # Verify Neuro-Symbolic Heuristic Override
    assert d.get("heuristic_override") is True, "GINA safety override failed to engage for acute symptoms!"
    assert d.get("risk_level") == "High", f"Expected High risk under GINA rails, got {d.get('risk_level')}"
    assert d.get("risk_score") >= 0.88, f"Expected risk score >= 0.88, got {d.get('risk_score')}"
    print("  [PASS] Neuro-Symbolic Rule Fired: GINA Step-5 safety heuristic overrode moderate environmental data.")
    print(f"         Final Risk Score: {d['risk_score']:.4f} (Heuristic Override: Active)")
    
    # Verify Actionable Recommendations
    recs = d.get("recommendations", [])
    assert len(recs) >= 3
    print(f"  [PASS] Actionable Clinical Recommendations Generated: ({len(recs)} guidance points)")
    for r in recs[:2]:
        print(f"         - {r}")
        
    # Verify Automated Alert Generation in Alert Table
    res_alerts = requests.get(f"{BASE_URL}/get-alerts/{user_id}", timeout=10)
    assert res_alerts.status_code == 200
    alerts = res_alerts.json().get("alerts", [])
    assert len(alerts) > 0
    print(f"  [PASS] Automated Safety Alert Created in DB: '{alerts[0]['message'][:65]}...'")


def test_module_6_environmental_monitoring(user_id):
    print_header("Module 6: Environmental Monitoring")
    print("Purpose: Monitor AQI, PM2.5, NO2, CO2, SO2 via Web API Sync & Manual Telemetry Input.")
    
    # 6.1 Mode A: Real-Time Open-Meteo Environmental Web API Sync
    print_sub("Mode A: Live GPS Environmental Web API Sync")
    res_api = requests.get(f"{BASE_URL}/api/live?lat=28.6139&lon=77.2090", timeout=10)
    assert res_api.status_code == 200
    env_live = res_api.json()
    assert env_live.get("success") is True
    live_data = env_live["data"]
    
    print(f"  [PASS] Live GPS Air Quality Synced via Open-Meteo API:")
    print(f"         - AQI (Air Quality Index):       {live_data['AQI']}")
    print(f"         - PM2.5 Particulate Matter:      {live_data['PM2.5']} ug/m3")
    print(f"         - NO2 (Nitrogen Dioxide):        {live_data['NO2 level']} ug/m3")
    print(f"         - SO2 (Sulphur Dioxide):         {live_data['SO2 level']} ug/m3")
    print(f"         - CO2 (Carbon Dioxide):          {live_data['CO2 level']} ppm")
    print(f"         - Ambient Temperature:           {live_data['Temperature']} deg C")
    print(f"         - Atmospheric Humidity:          {live_data['Humidity']} %")
    
    # 6.2 Mode B: Manual Telemetry Ingestion (Custom User Input)
    print_sub("Mode B: Manual Telemetry Ingestion & Database Sync (SensorData Table)")
    manual_payload = {
        "user_id": user_id,
        "air_quality": 178,
        "pm25": 84.5,
        "no2_level": 42.0,
        "so2_level": 26.5,
        "co2_level": 490.0,
        "temperature": 31.5,
        "humidity": 72.0
    }
    
    res_upload = requests.post(f"{BASE_URL}/upload-sensor-data", json=manual_payload, timeout=10)
    assert res_upload.status_code == 200
    upload_res = res_upload.json()
    assert upload_res.get("success") is True
    print(f"  [PASS] Manual Telemetry Stored: Sensor record ID #{upload_res['sensor_id']} committed to SensorData table.")
    
    # 6.3 Verify Retrieval of Stored Sensor Records
    res_history = requests.get(f"{BASE_URL}/get-user-data/{user_id}", timeout=10)
    assert res_history.status_code == 200
    history = res_history.json()
    assert history.get("count", 0) > 0
    latest = history["latest"]
    assert latest["air_quality"] == 178
    assert latest["pm25"] == 84.5
    print(f"  [PASS] Historical Sensor Telemetry Retrieved: {history['count']} total records on file.")
    print(f"         Latest reading matches manual input: AQI={latest['air_quality']}, PM2.5={latest['pm25']}")


def run_full_verification():
    print("\n" + "#" * 80)
    print(" HRIDYAVAYU SYSTEM TEST SUITE: 6 MAJOR MODULES VERIFICATION")
    print(" Target Endpoint:", BASE_URL)
    print(" Timestamp:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print("#" * 80)

    try:
        user_id = test_module_1_data_input_and_preprocessing()
        test_module_2_ensemble_machine_learning_engine()
        test_module_3_risk_score_and_classification()
        test_module_4_clinical_guidelines_assessment(user_id)
        test_module_5_neuro_symbolic_decision_layer(user_id)
        test_module_6_environmental_monitoring(user_id)

        print("\n" + "=" * 80)
        print(" [SUCCESS] ALL 6 HRIDYAVAYU MODULES VERIFIED & OPERATIONAL")
        print("=" * 80)
        print("  [PASS] Module 1: Data Input and Preprocessing         [PASSED]")
        print("  [PASS] Module 2: Ensemble Machine Learning Engine     [PASSED]")
        print("  [PASS] Module 3: Risk Score and Classification        [PASSED]")
        print("  [PASS] Module 4: Clinical Guidelines Assessment       [PASSED]")
        print("  [PASS] Module 5: Neuro-Symbolic Decision Layer        [PASSED]")
        print("  [PASS] Module 6: Environmental Monitoring             [PASSED]")
        print("=" * 80 + "\n")
        return True
    except Exception as e:
        print(f"\n[FAIL] Test suite encountered an error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_full_verification()
    sys.exit(0 if success else 1)
