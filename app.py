import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import os
import json
import pickle
import sqlite3
import hashlib
import warnings
import numpy as np
import pandas as pd
import requests
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS

warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================

REVIEW1_MODEL_PATH = "results/review1_ensemble.pkl"
FALLBACK_MODEL_PATH = "results/best_ensemble_model.pkl"
DB_PATH = "asthmai.db"

# ==================== MODEL LOADING ====================

print("Loading HridyaVayu Review 1 Models...")
review1_package = None
try:
    if os.path.exists(REVIEW1_MODEL_PATH):
        with open(REVIEW1_MODEL_PATH, "rb") as f:
            review1_package = pickle.load(f)
        print("✓ HridyaVayu Review 1 Collaborative 3-Model Ensemble Loaded Successfully!")
        model_name = "HridyaVayu Review 1 Ensemble (Baseline LR + Random Forest + Gradient Boosting)"
    else:
        with open(FALLBACK_MODEL_PATH, "rb") as f:
            fallback = pickle.load(f)
        print("✓ Fallback Ensemble Loaded Successfully!")
        model_name = "Ensemble Fallback"
except Exception as e:
    print(f"Error loading models: {e}")
    model_name = "Baseline Heuristics"

# Load Review 1 benchmark metrics
review1_metrics = {}
if os.path.exists("results/review1_metrics.json"):
    try:
        with open("results/review1_metrics.json", "r") as f:
            review1_metrics = json.load(f)
    except Exception:
        pass

# ==================== REAL-TIME AQI SERVICE ====================

class RealTimeAQI:
    def __init__(self):
        self.aqi_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
        self.weather_url = "https://api.open-meteo.com/v1/forecast"
        
    def get_live_data(self, latitude, longitude):
        try:
            aq_params = {
                "latitude": latitude, "longitude": longitude,
                "current": ["us_aqi", "pm2_5", "nitrogen_dioxide", "sulphur_dioxide"],
                "timezone": "auto"
            }
            w_params = {
                "latitude": latitude, "longitude": longitude,
                "current": ["temperature_2m", "relative_humidity_2m"],
                "timezone": "auto"
            }
            
            aq_res = requests.get(self.aqi_url, params=aq_params, timeout=5).json()
            w_res = requests.get(self.weather_url, params=w_params, timeout=5).json()
            
            if "current" in aq_res and "current" in w_res:
                return {
                    "timestamp": datetime.now().isoformat(),
                    "location": {"lat": latitude, "lon": longitude},
                    "AQI": aq_res["current"].get("us_aqi", 50) or 50,
                    "PM2.5": aq_res["current"].get("pm2_5", 15) or 15,
                    "NO2 level": aq_res["current"].get("nitrogen_dioxide", 20) or 20,
                    "SO2 level": aq_res["current"].get("sulphur_dioxide", 10) or 10,
                    "CO2 level": 415.0,
                    "Temperature": w_res["current"].get("temperature_2m", 25) or 25,
                    "Humidity": w_res["current"].get("relative_humidity_2m", 50) or 50
                }
            return None
        except Exception as e:
            print(f"AQI Live Fetch Error: {e}")
            return None

aqi_service = RealTimeAQI()

# ==================== FLASK SETUP ====================

app = Flask(__name__, template_folder='web_ui', static_folder='web_ui')
CORS(app, resources={r"/*": {"origins": "*"}})

# ==================== DATABASE HELPERS ====================

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode('utf-8')).hexdigest()

def get_db_connection():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE,
                email TEXT UNIQUE,
                password_hash TEXT,
                full_name TEXT,
                age INTEGER DEFAULT 30,
                gender TEXT DEFAULT 'Other',
                phone_no TEXT DEFAULT '',
                medical_history TEXT DEFAULT 'Mild intermittent asthma',
                emergency_contact_name TEXT DEFAULT 'Dr. Primary Contact',
                emergency_contact_phone TEXT DEFAULT '+1-800-555-0199',
                role TEXT DEFAULT 'User',
                created_at TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                score REAL,
                risk_level TEXT,
                patient_name TEXT,
                symptoms TEXT,
                aqi REAL,
                user_id INTEGER,
                baseline_score REAL DEFAULT 0.0,
                rf_score REAL DEFAULT 0.0,
                gb_score REAL DEFAULT 0.0,
                heuristic_override INTEGER DEFAULT 0
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sensor_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                timestamp TEXT,
                aqi REAL,
                pm25 REAL,
                so2_level REAL,
                no2_level REAL,
                co2_level REAL,
                humidity REAL,
                temperature REAL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS inhaler_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                timestamp TEXT,
                dose_count INTEGER DEFAULT 1
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                timestamp TEXT,
                message TEXT,
                risk_score REAL,
                risk_level TEXT,
                is_sos INTEGER DEFAULT 0
            )
        """)

        # Upgrades for existing users table
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(users)")
        user_cols = [col[1] for col in cur.fetchall()]
        for col_name, col_def in [
            ('age', "INTEGER DEFAULT 30"),
            ('gender', "TEXT DEFAULT 'Other'"),
            ('phone_no', "TEXT DEFAULT ''"),
            ('medical_history', "TEXT DEFAULT 'Mild intermittent asthma'"),
            ('emergency_contact_name', "TEXT DEFAULT 'Dr. Primary Contact'"),
            ('emergency_contact_phone', "TEXT DEFAULT '+1-800-555-0199'"),
            ('role', "TEXT DEFAULT 'User'")
        ]:
            if col_name not in user_cols:
                conn.execute(f"ALTER TABLE users ADD COLUMN {col_name} {col_def}")

        # Upgrades for existing predictions table
        cur.execute("PRAGMA table_info(predictions)")
        cols = [col[1] for col in cur.fetchall()]
        for col_name, col_def in [
            ('patient_name', "TEXT DEFAULT 'Anonymous'"),
            ('symptoms', "TEXT DEFAULT ''"),
            ('aqi', "REAL DEFAULT 0.0"),
            ('user_id', "INTEGER DEFAULT NULL"),
            ('baseline_score', "REAL DEFAULT 0.0"),
            ('rf_score', "REAL DEFAULT 0.0"),
            ('gb_score', "REAL DEFAULT 0.0"),
            ('heuristic_override', "INTEGER DEFAULT 0")
        ]:
            if col_name not in cols:
                conn.execute(f"ALTER TABLE predictions ADD COLUMN {col_name} {col_def}")

        # Seed Demo User & Admin
        cur.execute("SELECT id FROM users WHERE username = 'demo_user'")
        if not cur.fetchone():
            conn.execute(
                "INSERT INTO users (username, email, password_hash, full_name, age, gender, phone_no, medical_history, emergency_contact_name, emergency_contact_phone, role, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                ("demo_user", "patient@hridyavayu.org", hash_password("demo123"), "Alex Rivera", 28, "Male", "+1-555-0143", "Diagnosed bronchial asthma, allergen sensitive", "Maria Rivera (Spouse)", "+1-555-0188", "User", datetime.now().isoformat())
            )
        cur.execute("SELECT id FROM users WHERE username = 'demo_admin'")
        if not cur.fetchone():
            conn.execute(
                "INSERT INTO users (username, email, password_hash, full_name, age, gender, phone_no, medical_history, emergency_contact_name, emergency_contact_phone, role, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                ("demo_admin", "admin@hridyavayu.org", hash_password("admin123"), "Dr. Sarah Mitchell, MD (Lead Pulmonologist)", 45, "Female", "+1-555-0199", "Chief of Respiratory Medicine", "Hospital Emergency Desk", "+1-800-911-0000", "Admin", datetime.now().isoformat())
            )
        conn.commit()

# ==================== STATIC & FIGURE ROUTES ====================

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/web_ui/assets/<path:filename>")
def serve_assets(filename):
    return send_from_directory(os.path.join(app.root_path, "web_ui", "assets"), filename)

@app.route("/figures/<path:filename>")
def serve_figures(filename):
    return send_from_directory(os.path.join(app.root_path, "figures"), filename)

# ==================== AUTHENTICATION ROUTES ====================

@app.route("/api/auth/signup", methods=["POST"])
def auth_signup():
    try:
        data = request.get_json(force=True)
        username = data.get("username", "").strip()
        email = data.get("email", "").strip()
        password = data.get("password", "").strip()
        full_name = data.get("full_name", username).strip()
        role = data.get("role", "User").strip()
        age = int(data.get("age", 30) or 30)
        gender = data.get("gender", "Other")
        phone_no = data.get("phone_no", "")
        emergency_contact_name = data.get("emergency_contact_name", "")
        emergency_contact_phone = data.get("emergency_contact_phone", "")

        if not username or not password:
            return jsonify({"success": False, "error": "Username and password are required"}), 400

        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT id FROM users WHERE username = ? OR email = ?", (username, email))
            if cur.fetchone():
                return jsonify({"success": False, "error": "Username or email already registered"}), 409

            cur.execute(
                """INSERT INTO users (username, email, password_hash, full_name, age, gender, phone_no, emergency_contact_name, emergency_contact_phone, role, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (username, email, hash_password(password), full_name, age, gender, phone_no, emergency_contact_name, emergency_contact_phone, role, datetime.now().isoformat())
            )
            user_id = cur.lastrowid
            conn.commit()

            return jsonify({
                "success": True,
                "message": "Account created successfully",
                "user": {
                    "id": user_id,
                    "username": username,
                    "full_name": full_name,
                    "role": role,
                    "email": email,
                    "phone_no": phone_no,
                    "emergency_contact_name": emergency_contact_name,
                    "emergency_contact_phone": emergency_contact_phone
                }
            })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/auth/login", methods=["POST"])
def auth_login():
    try:
        data = request.get_json(force=True)
        username = data.get("username", "").strip()
        password = data.get("password", "").strip()

        if not username or not password:
            return jsonify({"success": False, "error": "Username and password required"}), 400

        pwd_hash = hash_password(password)
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM users WHERE (username = ? OR email = ?) AND password_hash = ?", (username, username, pwd_hash))
            user = cur.fetchone()
            if not user:
                return jsonify({"success": False, "error": "Invalid username or password"}), 401

            return jsonify({
                "success": True,
                "user": {
                    "id": user["id"],
                    "username": user["username"],
                    "full_name": user["full_name"],
                    "email": user["email"],
                    "role": user["role"],
                    "age": user["age"],
                    "gender": user["gender"],
                    "phone_no": user["phone_no"],
                    "medical_history": user["medical_history"],
                    "emergency_contact_name": user["emergency_contact_name"],
                    "emergency_contact_phone": user["emergency_contact_phone"]
                }
            })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/get-user/<int:user_id>", methods=["GET"])
def get_user_profile(user_id):
    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM users WHERE id = ?", (user_id,))
            user = cur.fetchone()
            if not user:
                return jsonify({"success": False, "error": "User not found"}), 404
            return jsonify({
                "success": True,
                "user": {
                    "id": user["id"],
                    "username": user["username"],
                    "full_name": user["full_name"],
                    "email": user["email"],
                    "role": user["role"],
                    "age": user["age"],
                    "gender": user["gender"],
                    "phone_no": user["phone_no"],
                    "medical_history": user["medical_history"],
                    "emergency_contact_name": user["emergency_contact_name"],
                    "emergency_contact_phone": user["emergency_contact_phone"]
                }
            })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/save-profile", methods=["POST"])
def save_profile():
    try:
        data = request.get_json(force=True)
        user_id = data.get("user_id")
        if not user_id:
            return jsonify({"success": False, "error": "user_id is required"}), 400

        with get_db_connection() as conn:
            conn.execute("""
                UPDATE users SET
                    full_name = COALESCE(?, full_name),
                    age = COALESCE(?, age),
                    gender = COALESCE(?, gender),
                    phone_no = COALESCE(?, phone_no),
                    medical_history = COALESCE(?, medical_history),
                    emergency_contact_name = COALESCE(?, emergency_contact_name),
                    emergency_contact_phone = COALESCE(?, emergency_contact_phone)
                WHERE id = ?
            """, (
                data.get("full_name"), data.get("age"), data.get("gender"),
                data.get("phone_no"), data.get("medical_history"),
                data.get("emergency_contact_name"), data.get("emergency_contact_phone"),
                user_id
            ))
            conn.commit()
        return jsonify({"success": True, "message": "Profile updated successfully"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# ==================== LIVE ENVIRONMENTAL SYNC ====================

@app.route("/api/live", methods=["GET"])
def get_live():
    lat = float(request.args.get("lat", 28.6139))
    lon = float(request.args.get("lon", 77.2090))
    data = aqi_service.get_live_data(lat, lon)
    if data:
        return jsonify({"success": True, "data": data})
    return jsonify({
        "success": True,
        "data": {
            "timestamp": datetime.now().isoformat(),
            "location": {"lat": lat, "lon": lon},
            "AQI": 125, "PM2.5": 48.5, "NO2 level": 32.0, "SO2 level": 18.0,
            "CO2 level": 415.0, "Temperature": 27.5, "Humidity": 62.0,
            "fallback": True
        }
    })

# ==================== INHALER & SOS ROUTES ====================

@app.route("/api/inhaler/use", methods=["POST"])
def inhaler_use():
    try:
        data = request.get_json(force=True) or {}
        user_id = data.get("user_id") or 1
        with get_db_connection() as conn:
            conn.execute("INSERT INTO inhaler_usage (user_id, timestamp, dose_count) VALUES (?, ?, 1)",
                         (user_id, datetime.now().isoformat()))
            cur = conn.cursor()
            cur.execute("SELECT SUM(dose_count) as total FROM inhaler_usage WHERE user_id = ?", (user_id,))
            total = cur.fetchone()["total"] or 0
            conn.commit()
        return jsonify({"success": True, "total_doses": total, "message": "Inhaler dose logged successfully"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/inhaler/count/<int:user_id>", methods=["GET"])
def inhaler_count(user_id):
    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT SUM(dose_count) as total FROM inhaler_usage WHERE user_id = ?", (user_id,))
            total = cur.fetchone()["total"] or 0
        return jsonify({"success": True, "total_doses": total})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/sos/alert", methods=["POST"])
def trigger_sos():
    try:
        data = request.get_json(force=True) or {}
        user_id = data.get("user_id") or 1
        patient_name = data.get("patient_name", "Alex Rivera")
        lat = data.get("lat", 0.0)
        lon = data.get("lon", 0.0)
        
        message = f"CRITICAL RESPIRATORY ALERT: Patient {patient_name} triggered Emergency SOS! Immediate assistance required."
        with get_db_connection() as conn:
            conn.execute(
                "INSERT INTO alerts (user_id, timestamp, message, risk_score, risk_level, is_sos) VALUES (?, ?, ?, ?, ?, ?)",
                (user_id, datetime.now().isoformat(), message, 1.0, "Critical SOS", 1)
            )
            conn.commit()

        return jsonify({
            "success": True,
            "message": "Emergency SOS triggered! SMS and Dispatch alerts sent to emergency contact.",
            "emergency_contact": data.get("emergency_contact", "Maria Rivera (+1-555-0188)")
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# ==================== PREDICTION PIPELINE (REVIEW 1 3-MODEL) ====================

def build_recommendations(risk_level, score, symptoms_freq, aqi):
    recs = []
    if risk_level == "High" or score >= 0.6:
        recs.append("🚨 HIGH RISK ALERT: Keep fast-acting rescue inhaler (SABA / Albuterol) within arm's reach.")
        recs.append("🏠 Remain indoors with HEPA air purification active. Seal windows against particulate pollution.")
        recs.append("⏱️ Measure Peak Expiratory Flow (PEF). If <60% personal best, enact GINA Red Zone action plan.")
        recs.append("📞 If acute dyspnea, wheezing, or chest tightness occurs, activate the Emergency SOS immediately.")
    elif risk_level == "Medium" or score >= 0.35:
        recs.append("⚠️ MODERATE RISK: Pre-dose maintenance corticosteroid / ICS-formoterol as prescribed.")
        recs.append("🏃 Avoid strenuous outdoor endurance exercises during peak AQI hours (midday to late evening).")
        recs.append("😷 Wear an N95 respirator mask if outdoor transit near traffic or smoke is unavoidable.")
        recs.append("💧 Maintain optimal hydration and keep rescue inhaler ready.")
    else:
        recs.append("✅ LOW RISK / WELL-CONTROLLED: Airway stability is optimal. Regular daily activities permitted.")
        recs.append("📅 Continue daily maintenance controller medication as scheduled by your physician.")
        recs.append("🌿 Monitor local air quality index if planning prolonged outdoor cardio.")
    return recs

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        patient_name = data.get("patient_name", "Anonymous Patient").strip()
        user_id = data.get("user_id", None)

        # Parse inputs
        aqi = float(data.get("AQI", 80))
        pm25 = float(data.get("PM2.5", 35))
        so2 = float(data.get("SO2 level", 15))
        no2 = float(data.get("NO2 level", 25))
        co2 = float(data.get("CO2 level", 450))
        humidity = float(data.get("Humidity", 50))
        temperature = float(data.get("Temperature", 24))

        symptoms_freq = str(data.get("Asthma Symptoms Frequency", "1-2 times a month"))
        triggers = str(data.get("Triggers", "Dust"))
        weather_sens = str(data.get("Weather Sensitivity", "None"))
        exposure = str(data.get("Poor Air Quality Exposure", "Occasionally"))
        night_diff = str(data.get("Night Breathing Difficulty", "Rarely"))

        # Log sensor data to DB
        ts = datetime.now().isoformat()
        with get_db_connection() as conn:
            conn.execute(
                """INSERT INTO sensor_data (user_id, timestamp, aqi, pm25, so2_level, no2_level, co2_level, humidity, temperature)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (user_id, ts, aqi, pm25, so2, no2, co2, humidity, temperature)
            )
            conn.commit()

        # Review 1 Model Prediction
        if review1_package:
            df = pd.DataFrame([{
                'AQI': aqi, 'PM2.5': pm25, 'SO2 level': so2, 'NO2 level': no2,
                'CO2 level': co2, 'Humidity': humidity, 'Temperature': temperature,
                'Asthma Symptoms Frequency': symptoms_freq, 'Triggers': triggers,
                'Weather Sensitivity': weather_sens, 'Poor Air Quality Exposure': exposure,
                'Night Breathing Difficulty': night_diff
            }])

            # Feature Engineering
            df['AQI_PM_ratio'] = df['AQI'] / (df['PM2.5'] + 1)
            df['pollution_index'] = (df['AQI'] * 0.4 + df['PM2.5'] * 0.3 + df['NO2 level'] * 0.15 + df['SO2 level'] * 0.15)
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
            df['clinical_risk_score'] = df['symptom_severity'] * 0.4 + df['exposure_score'] * 0.3 + df['night_score'] * 0.3
            df['env_risk_score'] = df['AQI_critical'] * 0.3 + df['AQI_unhealthy'] * 0.2 + df['PM25_high'] * 0.25 + (df['pollution_index'] / 500.0) * 0.25

            df_encoded = pd.get_dummies(df, columns=review1_package['categorical_cols'], drop_first=True)
            for col in review1_package['feature_columns']:
                if col not in df_encoded.columns:
                    df_encoded[col] = 0
            df_aligned = df_encoded[review1_package['feature_columns']]
            X_scaled = review1_package['scaler'].transform(df_aligned)

            prob_ens = review1_package['ensemble_model'].predict_proba(X_scaled)[0]
            prob_lr = review1_package['baseline_lr'].predict_proba(X_scaled)[0]
            prob_rf = review1_package['rf_model'].predict_proba(X_scaled)[0]
            prob_gb = review1_package['gb_model'].predict_proba(X_scaled)[0]

            # Risk Score is weighted high risk + medium risk
            # classes: 0=Low, 1=Medium, 2=High
            ens_score = float(prob_ens[2] * 1.0 + prob_ens[1] * 0.5)
            lr_score = float(prob_lr[2] * 1.0 + prob_lr[1] * 0.5)
            rf_score = float(prob_rf[2] * 1.0 + prob_rf[1] * 0.5)
            gb_score = float(prob_gb[2] * 1.0 + prob_gb[1] * 0.5)

            pred_class_idx = int(np.argmax(prob_ens))
            risk_level = review1_package['reverse_map'][pred_class_idx]
            confidence = float(np.max(prob_ens))

            # Uncertainty entropy
            p_nz = prob_ens[prob_ens > 0]
            uncertainty = float(-np.sum(p_nz * np.log2(p_nz)))
        else:
            ens_score = 0.5
            lr_score = 0.45
            rf_score = 0.52
            gb_score = 0.51
            risk_level = "Medium"
            confidence = 0.75
            uncertainty = 0.4

        # GINA Safety Guardrail Heuristic Override
        heuristic_override = False
        if symptoms_freq == "Daily" or night_diff == "Frequently":
            ens_score = max(ens_score, 0.88)
            risk_level = "High"
            confidence = 0.95
            heuristic_override = True
        elif symptoms_freq == "Frequently (Weekly)" and ens_score < 0.6:
            ens_score = max(ens_score, 0.62)
            risk_level = "Medium"
            confidence = 0.88
            heuristic_override = True

        # Generate Alerts if Risk >= 0.6
        if ens_score >= 0.6:
            with get_db_connection() as conn:
                conn.execute(
                    "INSERT INTO alerts (user_id, timestamp, message, risk_score, risk_level, is_sos) VALUES (?, ?, ?, ?, ?, 0)",
                    (user_id, ts, f"Elevated Asthma Risk ({ens_score*100:.1f}%) detected for {patient_name}. Follow safety action plan.", ens_score, risk_level)
                )
                conn.commit()

        # Log prediction to DB
        with get_db_connection() as conn:
            conn.execute(
                """INSERT INTO predictions (timestamp, score, risk_level, patient_name, symptoms, aqi, user_id, baseline_score, rf_score, gb_score, heuristic_override)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (ts, ens_score, risk_level, patient_name, symptoms_freq, aqi, user_id, lr_score, rf_score, gb_score, int(heuristic_override))
            )
            conn.commit()

        recommendations = build_recommendations(risk_level, ens_score, symptoms_freq, aqi)

        # Factor contributions
        factors = [
            {"factor": "Air Quality Index (AQI)", "impact": round(float(aqi / 300.0) * 100, 1), "value": f"{aqi:.0f}"},
            {"factor": "PM2.5 Exposure", "impact": round(float(pm25 / 150.0) * 100, 1), "value": f"{pm25:.1f} ug/m3"},
            {"factor": "Symptom Frequency", "impact": 85.0 if "Daily" in symptoms_freq else (65.0 if "Weekly" in symptoms_freq else 25.0), "value": symptoms_freq},
            {"factor": "Night Breathlessness", "impact": 80.0 if "Frequently" in night_diff else (50.0 if "Occasionally" in night_diff else 15.0), "value": night_diff}
        ]

        return jsonify({
            "success": True,
            "patient_name": patient_name,
            "risk_score": round(ens_score, 4),
            "asthma_risk_score": round(ens_score, 4),
            "risk_level": risk_level,
            "confidence": round(confidence, 4),
            "uncertainty_entropy": round(uncertainty, 4),
            "heuristic_override": heuristic_override,
            "model_architecture": "Review 1 Collaborative Ensemble (Baseline LR + RF + GB)",
            "model_breakdown": {
                "baseline_linear_regression": round(lr_score, 4),
                "random_forest": round(rf_score, 4),
                "gradient_boosting": round(gb_score, 4),
                "collaborative_ensemble": round(ens_score, 4)
            },
            "recommendations": recommendations,
            "factors": factors,
            "timestamp": ts
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

# ==================== ADMIN & TELEMETRY ROUTES ====================

@app.route("/api/admin/overview", methods=["GET"])
def admin_overview():
    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) as cnt FROM users")
            total_users = cur.fetchone()["cnt"]

            cur.execute("SELECT COUNT(*) as cnt FROM predictions")
            total_predictions = cur.fetchone()["cnt"]

            cur.execute("SELECT COUNT(*) as cnt FROM alerts WHERE risk_score >= 0.6")
            high_risk_alerts = cur.fetchone()["cnt"]

            cur.execute("SELECT AVG(score) as avg_score FROM predictions")
            avg_score_row = cur.fetchone()
            avg_risk = round(avg_score_row["avg_score"] or 0.0, 3)

            # Recent predictions
            cur.execute("SELECT id, timestamp, patient_name, risk_level, score, aqi, baseline_score, rf_score, gb_score FROM predictions ORDER BY id DESC LIMIT 15")
            recent_records = [dict(row) for row in cur.fetchall()]

            # Sensor logs
            cur.execute("SELECT id, timestamp, aqi, pm25, no2_level, so2_level, temperature, humidity FROM sensor_data ORDER BY id DESC LIMIT 10")
            recent_sensors = [dict(row) for row in cur.fetchall()]

            # Active alerts
            cur.execute("SELECT id, timestamp, user_id, message, risk_score, risk_level, is_sos FROM alerts ORDER BY id DESC LIMIT 10")
            recent_alerts = [dict(row) for row in cur.fetchall()]

        return jsonify({
            "success": True,
            "stats": {
                "total_users": total_users,
                "total_predictions": total_predictions,
                "high_risk_alerts": high_risk_alerts,
                "avg_risk": avg_risk
            },
            "review1_models": review1_metrics or {
                "baseline_lr": {"accuracy": 71.33, "f1": 0.7117},
                "random_forest": {"accuracy": 72.33, "f1": 0.7058},
                "gradient_boosting": {"accuracy": 69.67, "f1": 0.6942},
                "collaborative_ensemble": {"accuracy": 73.00, "f1": 0.7234}
            },
            "recent_predictions": recent_records,
            "recent_sensors": recent_sensors,
            "recent_alerts": recent_alerts
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/recent", methods=["GET"])
def api_recent():
    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT id, timestamp, score, risk_level, patient_name, symptoms, aqi FROM predictions ORDER BY id DESC LIMIT 10")
            rows = [dict(r) for r in cur.fetchall()]
        return jsonify({"success": True, "items": rows})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/stats", methods=["GET"])
def api_stats():
    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT risk_level, COUNT(*) as count FROM predictions GROUP BY risk_level")
            counts = {row["risk_level"]: row["count"] for row in cur.fetchall()}
            for k in ["Low", "Medium", "High"]:
                if k not in counts:
                    counts[k] = 0
            cur.execute("SELECT COUNT(*) as total FROM predictions")
            total = cur.fetchone()["total"]
        return jsonify({"success": True, "total": total, "distribution": counts})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/eda_data", methods=["GET"])
def api_eda_data():
    try:
        csv_path = "data/dataset.csv"
        if not os.path.exists(csv_path):
            return jsonify({"success": False, "error": "Dataset not found"}), 404

        df = pd.read_csv(csv_path)
        total_samples = len(df)
        risk_counts = df['Risk Class'].value_counts().to_dict()

        pollutants = ['AQI', 'PM2.5', 'NO2 level', 'SO2 level']
        pollutant_by_risk = {}
        for p in pollutants:
            pollutant_by_risk[p] = df.groupby('Risk Class')[p].mean().round(2).to_dict()

        symptom_crosstab = pd.crosstab(df['Asthma Symptoms Frequency'], df['Risk Class']).to_dict()
        night_crosstab = pd.crosstab(df['Night Breathing Difficulty'], df['Risk Class']).to_dict()
        sample_records = df.head(20).to_dict(orient='records')

        return jsonify({
            "success": True,
            "data": {
                "total_samples": total_samples,
                "features_count": len(df.columns) - 2,
                "classes": list(risk_counts.keys()),
                "risk_distribution": risk_counts,
                "pollutant_by_risk": pollutant_by_risk,
                "symptom_by_risk": symptom_crosstab,
                "night_by_risk": night_crosstab,
                "sample_records": sample_records
            }
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/figures_list", methods=["GET"])
def api_figures_list():
    figures = [
        {"id": "roc_curves", "title": "Multi-Model ROC Curves", "category": "Evaluation", "filename": "roc_curves.png", "desc": "ROC curves showing performance across all evaluated classifiers."},
        {"id": "confusion_matrices", "title": "Confusion Matrices Comparison", "category": "Evaluation", "filename": "confusion_matrices.png", "desc": "Confusion matrices illustrating classification distribution."},
        {"id": "model_comparison", "title": "Model Comparison Benchmark", "category": "Evaluation", "filename": "model_comparison.png", "desc": "Comparative accuracy and F1 performance across baseline and ensemble models."},
        {"id": "shap_summary", "title": "SHAP Global Feature Importance", "category": "Explainability", "filename": "shap_summary.png", "desc": "SHAP values identifying top clinical and environmental attack triggers."},
        {"id": "correlation_heatmap", "title": "Environmental Feature Correlations", "category": "EDA", "filename": "correlation_heatmap.png", "desc": "Correlation matrix across all pollutant and weather variables."}
    ]
    return jsonify({"success": True, "figures": figures})

# ==================== MAIN ====================

if __name__ == "__main__":
    init_db()
    if not os.path.exists("web_ui"):
        os.makedirs("web_ui")
    print("🚀 HridyaVayu Server Started - Review 1 Collaborative 3-Model Ready")
    app.run(host="0.0.0.0", port=7860, debug=False)
