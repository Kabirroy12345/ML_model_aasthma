import requests

data = {
    "AQI": 120,
    "PM2.5": 45,
    "SO2 level": 12,
    "NO2 level": 33,
    "CO2 level": 420,
    "Humidity": 65,
    "Temperature": 29,
    "Asthma Symptoms Frequency": "High",
    "Triggers": "Dust",
    "Weather Sensitivity": "Yes",
    "Poor Air Quality Exposure": "Yes",
    "Night Breathing Difficulty": "Yes"
}

response = requests.post("http://127.0.0.1:7860/predict", json=data)

print("Prediction Response:")
print(response.json())
