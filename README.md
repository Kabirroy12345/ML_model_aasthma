# Asthma Attack Risk Prediction using Machine Learning

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange)
![Flask](https://img.shields.io/badge/Flask-2.0-lightgrey)

A machine learning system that predicts asthma attack risk based on environmental pollutants (CO₂, O₂, etc.) with Flask API for deployment.

## 📌 Key Features
- **Predictive Model**: Neural network trained on environmental data
- **Production-Ready**: Flask web interface + Docker support
- **Complete Pipeline**: From data preprocessing to model serving
- **Saved Artifacts**: Includes trained model (.keras) and preprocessing pipeline (.pkl)

## 🏗️ Project Structure
.
├── app.py # Flask application
├── Dockerfile # Container configuration
├── requirements.txt # Python dependencies
├── best_model.keras # Optimized model weights
├── model.keras # Trained model
├── preprocessing.pkl # Data preprocessing pipeline
├── training_history.pkl # Training metrics
├── data/ # Raw datasets
├── notebooks/ # Jupyter notebooks (analysis/training)
└── utils/ # Helper scripts
│ ├── model.py # Model architecture
│ └── preprocess.py # Data processing

text

## 🚀 Quick Deployment

### Local Setup
```bash
pip install -r requirements.txt
python app.py
Docker Deployment
bash
docker build -t asthma-ml .
docker run -p 5000:5000 asthma-ml
🔧 Technical Stack
ML Framework: TensorFlow/Keras

Web Framework: Flask

Containerization: Docker

Data Processing: Pandas, Scikit-learn

📊 Sample API Request
python
import requests
import json

data = {
    "co2_level": 420,
    "o2_level": 21,
    "pm2_5": 35
}

response = requests.post("http://localhost:5000/predict", json=data)
print(response.json())
📝 License
This project is licensed under the MIT License - see LICENSE file for details.
