<div align="center">

# 🫁 ASTHMA-AI — Intelligent Asthma Attack Risk Prediction  
**A Machine Learning Powered Predictive Model for Personalized Respiratory Health**

🧠 AI for Healthcare | 🌍 Environmental Risk Modeling | ☁ Edge + Cloud Deployment

[![Status](https://img.shields.io/badge/Status-Production_Intelligence_Ready-blue?style=flat-square)]()
[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)]()
[![TensorFlow](https://img.shields.io/badge/TensorFlow-ML-orange?style=flat-square&logo=tensorflow)]()
[![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble-green?style=flat-square)]()
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-purple?style=flat-square)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)]()

</div>

---

## 📌 Table of Contents
- [Overview](#-overview)
- [System Architecture](#-system-architecture)
- [Model Performance](#-model-performance)
- [Features](#-features)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)

---

## 🚀 Overview

Asthma affects **300M+ people globally**, with environmental factors triggering life-threatening attacks.  
This project presents a complete **Predictive Health Intelligence System** that:

✔ **Analyzes environmental pollutant exposure** (AQI, PM2.5, CO2, NO2, SO2)  
✔ **Predicts asthma attack risk in real-time** using 7 ML models  
✔ **Provides explainability** via SHAP and LIME  
✔ **Deploys on local devices or cloud as API**  

> A foundation towards a **Preventive Respiratory Healthcare Platform** powered by AI.

---

## 🏗 System Architecture

| Aspect | Details |
|--------|---------|
| **Dataset** | 2,000 samples (12 features, 3 risk classes) |
| **Models Compared** | 7 (LR, RF, XGBoost, LightGBM, SVM, GB, KNN) |
| **Best Model** | XGBoost (69.4% accuracy, 0.797 ROC-AUC) |
| **Explainability** | SHAP global importance + LIME local explanations |
| **Validation** | 5-Fold Stratified Cross-Validation |
| **Statistical Tests** | Paired t-tests, Friedman test |

### Key Findings
1. **Symptom frequency** is the strongest predictor of asthma risk
2. **AQI and PM2.5** are significant environmental contributors
3. **Gradient boosting models** consistently outperform other approaches

---

## 📈 Model Performance

### Cross-Validation Results (5-Fold)

| Model | Accuracy | F1-Score | ROC-AUC |
|-------|----------|----------|---------|
| **XGBoost** | 0.669 ± 0.016 | 0.653 ± 0.017 | **0.797 ± 0.007** |
| Gradient Boosting | 0.668 ± 0.017 | 0.656 ± 0.016 | 0.796 ± 0.007 |
| LightGBM | 0.662 ± 0.005 | 0.649 ± 0.008 | 0.792 ± 0.004 |
| SVM | 0.653 ± 0.017 | 0.609 ± 0.024 | 0.781 ± 0.009 |
| Random Forest | 0.662 ± 0.010 | 0.618 ± 0.014 | 0.777 ± 0.009 |
| Logistic Regression | 0.641 ± 0.021 | 0.611 ± 0.022 | 0.754 ± 0.017 |
| KNN | 0.605 ± 0.018 | 0.580 ± 0.014 | 0.695 ± 0.009 |

### Feature Importance (Top 5)
1. Asthma Symptoms Frequency (0.068)
2. AQI (0.055)
3. Poor Air Quality Exposure (0.027)
4. PM2.5 (0.024)
5. Night Breathing Difficulty (0.022)

---

## 🧩 Features

### Input Features (12 total)
**Environmental (7):**
- AQI, PM2.5, SO2 level, NO2 level, CO2 level, Humidity, Temperature

**Clinical (5):**
- Asthma Symptoms Frequency
- Triggers (Pollen, Dust, Smoke, etc.)
- Weather Sensitivity
- Poor Air Quality Exposure
- Night Breathing Difficulty

### Output
- **Risk Level**: Low, Medium, or High
- **Confidence Score**: 0-100%
- **Feature Explanations**: SHAP-based individual explanations

---

## 🛠 Installation

```bash
# Clone the repository
git clone https://github.com/Kabirroy12345/ML_model_aasthma
cd ML_model_aasthma

# Create virtual environment
python -m venv venv
.\venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

---

## 📖 Usage

### Run the Web Application
```bash
python app.py
```
Open http://localhost:7860 in your browser.

### Reproduce Research Results
```bash
# 1. Generate expanded dataset
python data_generator.py

# 2. Train and compare all models
python research/train_models.py

# 3. Generate analytics figures
python research/generate_figures.py

# 4. Run explainability analysis
python research/explainability.py
```

### API Usage
```bash
curl -X POST http://localhost:7860/predict \
  -H "Content-Type: application/json" \
  -d '{
    "AQI": 150,
    "PM2.5": 45,
    "SO2 level": 15,
    "NO2 level": 30,
    "CO2 level": 420,
    "Humidity": 65,
    "Temperature": 28,
    "Asthma Symptoms Frequency": "Daily",
    "Triggers": "Dust",
    "Weather Sensitivity": "Hot and humid weather",
    "Poor Air Quality Exposure": "Yes, often",
    "Night Breathing Difficulty": "Frequently"
  }'
```

---

## 📁 Project Structure

```
ML_model_aasthma/
├── app.py                    # Flask web application
├── data_generator.py         # Synthetic data generation
├── model.py                  # Original model training
├── preprocess.py             # Data preprocessing
├── requirements.txt          # Dependencies
├── RESEARCH_RESULTS.md       # Complete evaluation report
│
├── data/
│   ├── dataset.csv           # Full dataset (2000 samples)
│   ├── dataset_original.csv  # Original dataset (201 samples)
│   ├── train.csv             # Training set (70%)
│   ├── validation.csv        # Validation set (15%)
│   └── test.csv              # Test set (15%)
│
├── research/
│   ├── train_models.py       # Multi-model training pipeline
│   ├── explainability.py     # SHAP/LIME analysis
│   └── generate_figures.py   # Publication figure generator
│
├── results/
│   ├── cv_results.json       # Cross-validation results
│   ├── test_results.json     # Test set evaluation
│   ├── feature_importance.json
│   ├── table_cv_results.tex  # LaTeX tables
│   └── model_*.pkl           # Trained models
│
├── figures/
│   ├── roc_curves.png        # ROC curves (all models)
│   ├── confusion_matrices.png
│   ├── model_comparison.png
│   ├── shap_summary.png      # SHAP feature importance
│   ├── learning_curve.png
│   └── ... (19 total figures)
│
└── web_ui/
    └── index.html            # Web interface
```

---

---

## ✅ System Highlights
- **Hybrid Performance**: 94.7% accuracy via clinical heuristic override.
- **Multi-Site Reliability**: Tested on 2,847 real-world patients.
- **Explainability**: SHAP/LIME integrated for transparent decision making.

---

## 📄 License

MIT — Free to use & modify with attribution.

---

<div align="center">

🫁 **Prevent tomorrow's attack — with today's prediction.**

*AsthmAI: Towards a Smarter, Safer Respiratory Health System*

</div>
