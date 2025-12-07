import pickle
import tensorflow as tf
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

# Load trained model
model = tf.keras.models.load_model("model.keras")

# Load preprocessing objects
with open("preprocessing.pkl", "rb") as f:
    encoder, scaler = pickle.load(f)

# Define Flask app
app = Flask(__name__)

# Define expected input features (original 12)
FEATURES = [
    "AQI", "PM2.5", "SO2 level", "NO2 level", "CO2 level",
    "Humidity", "Temperature", "Asthma Symptoms Frequency",
    "Triggers", "Weather Sensitivity", "Poor Air Quality Exposure",
    "Night Breathing Difficulty"
]

@app.route("/")
def home():
    return jsonify({"message": "Asthma Risk Prediction API is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        # Ensure all required fields are present
        if not all(feature in data for feature in FEATURES):
            return jsonify({"error": "Missing required fields"}), 400

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
        prediction = model.predict(X)[0][0]

        # Prepare output
        risk_level = (
            "High" if prediction >= 0.7 else
            "Medium" if prediction >= 0.4 else
            "Low"
        )

        return jsonify({
            "asthma_risk_score": float(prediction),
            "risk_level": risk_level
        })

    except Exception as e:
        return jsonify({"error": "Internal server error: " + str(e)}), 500

# Run the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=True)

