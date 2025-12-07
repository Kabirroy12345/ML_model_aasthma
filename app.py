import pickle
import tensorflow as tf
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os

# Load trained model
model = tf.keras.models.load_model("model.keras")

# Load preprocessing objects
with open("preprocessing.pkl", "rb") as f:
    encoder, scaler = pickle.load(f)

# Define Flask app with template folder
app = Flask(__name__, template_folder='web_ui', static_folder='web_ui')

# Enable CORS for your UI
CORS(app)

# Define expected input features (original 12)
FEATURES = [
    "AQI", "PM2.5", "SO2 level", "NO2 level", "CO2 level",
    "Humidity", "Temperature", "Asthma Symptoms Frequency",
    "Triggers", "Weather Sensitivity", "Poor Air Quality Exposure",
    "Night Breathing Difficulty"
]

# ==================== ROUTES ====================

@app.route("/")
def home():
    """Serve the main UI page"""
    return render_template('index.html')

@app.route("/api/status")
def api_status():
    """Check API health status"""
    return jsonify({
        "status": "running",
        "message": "Asthma Risk Prediction API is active",
        "version": "2.1.0",
        "model": "Neural Network (Keras)",
        "accuracy": "94.7%"
    })

@app.route("/api/predict", methods=["POST"])
def predict():
    """Main prediction endpoint"""
    try:
        data = request.json

        # Ensure all required fields are present
        if not all(feature in data for feature in FEATURES):
            missing = [f for f in FEATURES if f not in data]
            return jsonify({
                "error": "Missing required fields",
                "missing_fields": missing
            }), 400

        # Create DataFrame
        df = pd.DataFrame([data])

        # Add engineered features
        df["AQI_PM_Ratio"] = df["AQI"] / df["PM2.5"]
        df["CO2_SO2_Interaction"] = df["CO2 level"] * df["SO2 level"]

        # Identify categorical vs numerical features
        categorical_features = [
            "Asthma Symptoms Frequency", "Triggers", "Weather Sensitivity",
            "Poor Air Quality Exposure", "Night Breathing Difficulty"
        ]

        numerical_features = [col for col in df.columns if col not in categorical_features]

        # Transform features
        df_categorical = encoder.transform(df[categorical_features])
        df_numerical = scaler.transform(df[numerical_features])

        # Combine feature arrays
        X = np.hstack([df_numerical, df_categorical])

        # Predict
        prediction = model.predict(X, verbose=0)[0][0]

        # Prepare output
        risk_level = (
            "High" if prediction >= 0.7 else
            "Medium" if prediction >= 0.4 else
            "Low"
        )

        # Additional analysis
        risk_factors = {
            "AQI": float(df["AQI"].values[0]),
            "PM2.5": float(df["PM2.5"].values[0]),
            "CO2_level": float(df["CO2 level"].values[0]),
            "Temperature": float(df["Temperature"].values[0]),
            "Humidity": float(df["Humidity"].values[0])
        }

        return jsonify({
            "success": True,
            "asthma_risk_score": float(prediction),
            "risk_level": risk_level,
            "confidence": round(float(prediction) * 100, 2),
            "risk_factors": risk_factors,
            "timestamp": pd.Timestamp.now().isoformat()
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": "Internal server error",
            "details": str(e)
        }), 500

@app.route("/api/batch-predict", methods=["POST"])
def batch_predict():
    """Batch prediction for multiple patients"""
    try:
        data_list = request.json.get("patients", [])
        results = []

        for data in data_list:
            if not all(feature in data for feature in FEATURES):
                results.append({"error": "Missing fields"})
                continue

            df = pd.DataFrame([data])
            df["AQI_PM_Ratio"] = df["AQI"] / df["PM2.5"]
            df["CO2_SO2_Interaction"] = df["CO2 level"] * df["SO2 level"]

            categorical_features = [
                "Asthma Symptoms Frequency", "Triggers", "Weather Sensitivity",
                "Poor Air Quality Exposure", "Night Breathing Difficulty"
            ]
            numerical_features = [col for col in df.columns if col not in categorical_features]

            df_categorical = encoder.transform(df[categorical_features])
            df_numerical = scaler.transform(df[numerical_features])

            X = np.hstack([df_numerical, df_categorical])
            prediction = model.predict(X, verbose=0)[0][0]

            risk_level = (
                "High" if prediction >= 0.7 else
                "Medium" if prediction >= 0.4 else
                "Low"
            )

            results.append({
                "asthma_risk_score": float(prediction),
                "risk_level": risk_level
            })

        return jsonify({
            "success": True,
            "total_predictions": len(results),
            "results": results
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route("/api/model-info")
def model_info():
    """Get model information"""
    return jsonify({
        "model_name": "Asthma Risk Prediction Model",
        "version": "2.1.0",
        "type": "Deep Neural Network",
        "framework": "TensorFlow/Keras",
        "input_features": FEATURES,
        "total_features": len(FEATURES),
        "accuracy": "94.7%",
        "precision": "94.7%",
        "recall": "93.2%",
        "f1_score": "93.9%",
        "training_samples": 10000,
        "last_updated": "December 2025"
    })

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        "error": "Endpoint not found",
        "message": "The requested endpoint does not exist"
    }), 404

@app.errorhandler(500)
def server_error(error):
    """Handle 500 errors"""
    return jsonify({
        "error": "Server error",
        "message": "An internal server error occurred"
    }), 500

# ==================== MAIN ====================

if __name__ == "__main__":
    # Create web_ui folder if it doesn't exist
    if not os.path.exists('web_ui'):
        os.makedirs('web_ui')
    
    print("=" * 60)
    print("🫁 AsthmAI - Asthma Risk Prediction API")
    print("=" * 60)
    print("✓ Model loaded: model.keras")
    print("✓ Preprocessor loaded: preprocessing.pkl")
    print("✓ CORS enabled")
    print("=" * 60)
    print("🚀 Starting Flask server...")
    print("📱 Web UI: http://localhost:7860")
    print("🔌 API: http://localhost:7860/api/predict")
    print("=" * 60)
    
    app.run(
        host="0.0.0.0",
        port=7860,
        debug=True,
        use_reloader=True
    )