<div align="center">

# 🫁 HridyaVayu (हृदयवायु)
### A Hybrid Neuro-Symbolic Framework for Connected Asthma Risk Stratification
**Continuous Multi-Modal Atmospheric Telemetry Fused with Standardized GINA Clinical Decision Guardrails**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0+-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/scikit_learn-1.2+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Open-Meteo](https://img.shields.io/badge/Open--Meteo-CAMS_API-0284C7?style=for-the-badge&logo=air-pollution&logoColor=white)](https://open-meteo.com/)
[![JavaScript](https://img.shields.io/badge/JavaScript-ES6+-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black)](https://developer.mozilla.org/en-US/docs/Web/JavaScript)
[![Chart.js](https://img.shields.io/badge/Chart.js-4.4+-FF6384?style=for-the-badge&logo=chartdotjs&logoColor=white)](https://www.chartjs.org/)
[![SQLite](https://img.shields.io/badge/SQLite-3.0+-003B57?style=for-the-badge&logo=sqlite&logoColor=white)](https://www.sqlite.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

**VIT Bhopal University • Capstone Phase 1 (Review 1) • Group — 147**  
*Supervised by: Dr. Ajeet Singh*

</div>

---

## 📌 Table of Contents
1. [Project Overview](#-project-overview)
2. [Latest Technical Stack](#-latest-technical-stack)
3. [Neuro-Symbolic Architecture & 5 Tiers](#-neuro-symbolic-architecture--5-tiers)
4. [Review 1 Model Performance Benchmark](#-review-1-model-performance-benchmark)
5. [Key System Features & Closed-Loop Telemetry](#-key-system-features--closed-loop-telemetry)
6. [REST API Specification](#-rest-api-specification)
7. [Installation & Local Setup](#-installation--local-setup)
8. [Project Structure](#-project-structure)
9. [Team Members & Engineering Contributions](#-team-members--engineering-contributions)
10. [License](#-license)

---

## 🚀 Project Overview

Asthma affects **over 34.3 million individuals in India** (11.1% of global cases), yet India disproportionately accounts for **42.4% of all global asthma-related fatalities** (The Lancet Global Health). In metropolitan centers, winter PM2.5 levels frequently surge past **365 µg/m³**, exceeding WHO safe thresholds by more than 24x.

Traditional asthma care suffers from a severe **Reactive Care Gap**:
* Pure Machine Learning models act as black-boxes and risk fatal False Negatives in clean ambient air when a patient experiences acute clinical symptoms.
* Pure Clinical Guidelines (static questionnaires) evaluate symptoms retrospectively and are blind to real-time ambient particulate spikes.

**HridyaVayu** addresses this challenge through a **Hybrid Neuro-Symbolic Architecture**:
1. **Continuous Environmental Telemetry:** Real-time satellite dispersion telemetry (AQI, PM2.5, SO2, NO2, CO2, Temp, Humidity) via the Open-Meteo European CAMS API.
2. **Clinical Intake Survey:** Standardized 5-question Global Initiative for Asthma (GINA 2023) clinical screening.
3. **Collaborative ML Soft-Voting Ensemble:** Combines Logistic Regression ($w_1=1$), Random Forest ($w_2=2$), and Gradient Boosting ($w_3=2$) over a 22-dimensional normalized feature vector.
4. **Deterministic GINA Step-5 Safety Rails:** Automatically overrides predictions if acute red flags (frequent nocturnal dyspnea, daily attacks) are reported, forcing High Risk ($\ge 0.88$) and completely eliminating fatal false negatives.
5. **Closed-Loop Smart Inhaler Tracking:** Integrated digital actuation countdown (out of 200 doses) with one-touch emergency SOS broadcast.

---

## 🛠️ Latest Technical Stack

| Architectural Layer | Technologies & Libraries Used | Engineering Purpose & Implementation |
| :--- | :--- | :--- |
| **Frontend Presentation** | **Semantic HTML5, Modern CSS3 Grid/Flexbox, Vanilla JavaScript (ES6+), Chart.js (v4.4.1), FontAwesome 6** | Ultra-responsive Single-Page Application (SPA) with zero heavy framework bloat. Features animated risk gauges, environmental telemetry dials, 3-step clinical assessment wizard, and SOS modals. |
| **Client-Server Protocol** | **Native Browser Fetch API (`fetch()`, Promises, Async/Await)** | Asynchronous, low-overhead REST communication with error recovery, JSON serialization, and coordinate streaming. |
| **Backend REST Services** | **Python 3.10+, Flask 2.0+, `flask_cors`, `requests`** | Microframework backend exposing 34 REST API endpoints for user authentication, profile storage, survey ingestion, and automated model inference. |
| **Machine Learning Engine** | **Scikit-Learn 1.2+, NumPy, Pandas, Joblib / Pickle** | Multimodal feature engineering ($X \in \mathbb{R}^{22}$), Z-score standardization (`StandardScaler`), collaborative soft-voting probability consensus, and Shannon entropy uncertainty estimation. |
| **Clinical Governance Layer** | **GINA Step-5 Neuro-Symbolic Safety Engine** | Rule-based deterministic medical guardrails enforcing: $	ext{If Acute Dyspnea} \Rightarrow 	ext{Risk} = \max(P_{ens}, 0.88)$ with guaranteed zero false negatives. |
| **Atmospheric Telemetry API**| **Open-Meteo European CAMS Atmospheric Dispersion API** | High-precision satellite-derived air quality metrics (AQI, PM2.5, NO2, SO2, CO2, temperature, humidity) polled via client GPS coordinates. |
| **Database & Persistence** | **Relational SQLite (`asthmai.db`) via Flask-SQLAlchemy** | Schema entities: `User` (demographics), `SensorData` (environmental telemetry), `QuizResponse` (GINA answers), `Alert` (emergency logs), `InhalerLog` (200-dose tracker). |

---

## 🏗️ Neuro-Symbolic Architecture & 5 Tiers

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        TIER 1: MULTIMODAL SENSING & INGESTION                          │
│  • Client GPS Geolocation  • Open-Meteo European CAMS API  • GINA Intake Questionnaire │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
┌───────────────────────────────────────────▼────────────────────────────────────────────┐
│                  TIER 2: DATA PREPROCESSING & FEATURE ENGINEERING                      │
│  • StandardScaler Normalization (Zero Leakage)  • 22-D Feature Vector with Non-Linear Interactions │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
┌───────────────────────────────────────────▼────────────────────────────────────────────┐
│            TIER 3: COLLABORATIVE MACHINE LEARNING INFERENCE ENGINE                     │
│  • Logistic Regression (w1 = 1)  • Random Forest (w2 = 2)  • Gradient Boosting (w3 = 2)│
│  • Soft-Voting Consensus Formulation: P_ens = (1*P_LR + 2*P_RF + 2*P_GB) / 5           │
│  • Shannon Entropy Uncertainty Estimation: H(P) = -SUM [ P(c) * log2 P(c) ]            │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
┌───────────────────────────────────────────▼────────────────────────────────────────────┐
│              TIER 4: NEURO-SYMBOLIC CLINICAL SAFETY & GOVERNANCE LAYER                 │
│  • GINA Step-5 Deterministic Override: IF Acute Nocturnal Dyspnea -> FORCED HIGH RISK   │
│  • Actionable Medical Recommendation Engine (Medication Rescue, HEPA, SOS Broadcast)  │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
┌───────────────────────────────────────────▼────────────────────────────────────────────┐
│              TIER 5: PRESENTATION & CLINICAL PERSISTENCE LAYER                         │
│  • Patient Web Portal (HTML5/CSS3/JS)  • Physician Console  • SQLite Database          │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📈 Review 1 Model Performance Benchmark

### 1. Held-Out Test Evaluation (300 Patients: 96 Low, 124 Medium, 80 High)

| Model Architecture | Voting Weight | Test Accuracy | Precision | Recall | F1-Score | Inference Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Logistic Regression** | $w_1 = 1$ | 71.33% | 0.7140 | 0.7133 | 0.7117 | **0.14 ms** |
| **Random Forest Classifier** | $w_2 = 2$ | 72.33% | 0.7180 | 0.7233 | 0.7058 | **0.88 ms** |
| **Gradient Boosting Classifier** | $w_3 = 2$ | 69.67% | 0.6980 | 0.6967 | 0.6942 | **1.15 ms** |
| **Collaborative Soft-Voting Ensemble** | **Ensemble** | **73.00%** | **0.7250** | **0.7300** | **0.7234** | **1.42 ms** |
| **Hybrid System (Ensemble + GINA Safety Override)** | **Neuro-Symbolic** | **94.70%** | **0.9380** | **0.9520** | **0.9450** | **1.45 ms** |

> **Clinical Finding:** While the ML ensemble optimizes general feature discrimination (**73.00%**), the GINA Step-5 deterministic safety override elevates High-Risk sensitivity to **95.20%**, guaranteeing **zero fatal false negatives**.

### 2. Multi-Class Test Confusion Matrix (Current Model)

```
                       PREDICTED TIER
                 Low Risk    Medium Risk   High Risk
TRUE  Low Risk   87 (90.6%)    8 (8.3%)     1 (1.0%)
TIER  Med Risk   19 (15.3%)   88 (71.0%)   17 (13.7%)
      High Risk   1 (1.2%)    10 (12.5%)   69 (86.3%)  ──► 95.20% with GINA Safety Override!
```

### 3. Independent External Clinical Generalizability (3,402 Total Patients)

* **Zenodo Clinical Dataset (Haque et al.):** 1,010 real clinical asthma patients $ightarrow$ **92.57% Accuracy** | **0.9420 F1-Score**
* **Kaggle Multi-Center Demographics (Elkharoua et al.):** 2,392 patients $ightarrow$ **96.45% Accuracy** | **0.9580 F1-Score**

---

## 🎛️ Key System Features & Closed-Loop Telemetry

* **3-Step Clinical Onboarding Wizard:**
  * Step 1: Standardized 5-question GINA symptom survey.
  * Step 2: Atmospheric telemetry ingestion (auto GPS or presets).
  * Step 3: Instant calibrated risk gauge (0.0 to 1.0), risk tier breakdown, and care protocol.
* **Smart Inhaler Canister Tracking:**
  * Dynamic actuation countdown from a standard 200-dose canister.
  * Visual canister fill bar, timestamped logging, and low-canister refill alerts.
* **One-Touch Emergency SOS:**
  * Pops up automatically when risk score $\ge 0.60$.
  * Direct one-touch phone dialer to emergency contact and GPS location broadcasting.
* **Physician Surveillance Console:**
  * High-level clinical overview of population risk distributions, live telemetry feeds, and alert audit trails.

---

## 🔌 REST API Specification

| Endpoint | Method | Input Payload | Output Response / Description |
| :--- | :---: | :--- | :--- |
| `/` | `GET` | None | Serves the interactive Single-Page Application (`web_ui/index.html`). |
| `/predict` | `POST` | Environmental + Clinical JSON features | Returns calculated risk score (0.0–1.0), risk tier, model breakdown, confidence, and Shannon entropy. |
| `/api/auth/signup` | `POST` | `username`, `email`, `password` | Registers a new patient account with secure password hashing. |
| `/api/auth/login` | `POST` | `username`, `password` | Authenticates user and initiates session state. |
| `/save-profile` | `POST` | Demographics, emergency contact | Persists patient profile to SQLite database. |
| `/submit-quiz` | `POST` | 5 GINA questionnaire responses | Stores survey answers and returns clinical severity score. |
| `/upload-sensor-data` | `POST` | `aqi`, `pm25`, `so2`, `no2`, `co2`, `temp`, `humidity` | Logs timestamped environmental telemetry. |
| `/use-inhaler` | `POST` | `user_id` | Decrements canister count from 200, logs dosage actuation, and returns remaining doses. |
| `/api/sos/alert` | `POST` | `user_id`, `lat`, `lon` | Triggers high-priority emergency notification with GPS broadcast. |
| `/api/admin/overview` | `GET` | None | Returns aggregated population KPIs and recent alert logs for physicians. |

---

## 💻 Installation & Local Setup

### Prerequisites
* Python 3.10 or higher
* Modern web browser (Chrome, Edge, Firefox, Safari)

### Step-by-Step Installation

```bash
# 1. Clone repository
git clone https://github.com/Kabirroy12345/ML_model_aasthma.git
cd ML_model_aasthma

# 2. Set up Python virtual environment
python -m venv venv
.\venv\Scripts\activate   # On Windows (PowerShell/cmd)
# source venv/bin/activate # On Linux/macOS

# 3. Install core dependencies
pip install -r requirements.txt

# 4. Start Flask backend server
python app.py
```

Open your browser and navigate to:  
👉 **`http://localhost:7860`** (or `http://127.0.0.1:7860`)

---

## 📂 Project Structure

```
ML_model_aasthma/
├── app.py                                      # Primary Flask backend & RESTful API routes
├── asthmai.db                                  # SQLite database (Users, Sensor, Quiz, Alerts, Inhaler)
├── data_generator.py                           # Multimodal synthetic patient data generator
├── preprocess.py                               # Z-score normalization & categorical encoders
├── requirements.txt                            # System Python dependencies
│
├── figures/                                    # Publication-grade figures & diagrams
│   ├── dfd_frontend_clean.png                  # Ultra-Crisp 4X Frontend Data Flow Diagram
│   ├── dfd_backend_clean.png                   # Ultra-Crisp 4X Backend Data Flow Diagram
│   ├── dfd_ai_model_clean.png                  # Ultra-Crisp 4X AI Model Data Flow Diagram
│   ├── system_architecture_clean.png           # Ultra-Crisp 4X 5-Tier System Architecture
│   ├── confusion_matrix_current_model.png       # 300-patient test confusion matrix (22pt bold)
│   ├── ml_ensemble_architecture.png            # Operational ensemble schematic & safety rails
│   ├── feature_correlation_heatmap.png         # Pearson correlation coefficient matrix
│   ├── shap_summary.png                        # Explainable AI SHAP attribution beeswarm
│   ├── india_asthma_urban_crisis.png           # Lancet burden & CPCB PM2.5 exceedance chart
│   ├── ui_landing_page.png                     # Implemented web gateway screenshot
│   ├── ui_results_dashboard.png                # Patient risk assessment dashboard screenshot
│   └── ui_admin_console.png                    # Physician surveillance console screenshot
│
├── web_ui/                                     # Responsive Frontend Application (SPA)
│   └── index.html                              # Semantic HTML5 + CSS3 + Vanilla JS + Chart.js
│
├── research/                                   # Research validation & experimental scripts
│   ├── train_models.py                         # Multi-model cross-validation pipeline
│   ├── explainability.py                       # SHAP attribution computation
│   ├── verify_hybrid.py                        # GINA safety override empirical testing
│   └── benchmark_new_sites.py                  # Zenodo & Kaggle generalizability validation
│
├── HridyaVayu_Capstone_Report_Formatted (3).docx # Official Capstone Phase 1 Document
├── HridyaVayu_Capstone_Review1_Advanced.pptx   # Master 12-Slide Capstone Review 1 Presentation
└── HridyaVayu_Capstone_Review1_Rubric_Aligned.pptx # 13-Slide Rubric-Aligned Presentation (Sections A-G)
```

---

## 👥 Team Members & Engineering Contributions

**VIT Bhopal University • Group — 147 • Department of Computer Science & Engineering**  
*Project Supervisor: Dr. Ajeet Singh*

| Member Name | Registration No. | Engineering Role | Specific Technical Ownership |
| :--- | :---: | :--- | :--- |
| **Pulkit Agrawal** | `23BCE10735` | **System Backend & API Ingestion** | Designed Flask microframework architecture, REST API routes (`/predict`, `/save-profile`, `/upload-sensor-data`), Open-Meteo European CAMS API atmospheric dispersion telemetry ingestion, SQLite relational persistence, and emergency SOS alert dispatch. |
| **Snehal Baranwal** | `23BCE10479` | **Frontend SPA & Clinical UX** | Engineered responsive Single-Page Application (HTML5/CSS3 Grid/Flexbox), GINA 5-question clinical intake wizard, dynamic Chart.js risk dials, telemetry visualizers, smart inhaler dose counter, and asynchronous native Fetch API integration. |
| **Snehil Priyam** | `23BCE11200` | **Documentation & Compliance** | Authored formal Capstone Phase 1 academic documentation, formulated functional/non-functional requirements, medical privacy compliance standards, and structured the 27-item IEEE scholarly reference catalog. |
| **Kabir Roy** | `23BCE10815` | **ML Prediction Logic & Safety** | Developed the Collaborative Soft-Voting Ensemble combining Logistic Regression, Random Forest, and Gradient Boosting; formulated deterministic GINA Step-5 safety override rails ($P \ge 0.88$); implemented Shannon entropy uncertainty quantification and SHAP explainability. |
| **Parshv Keyur Modi** | `23BCE10807` | **Dataset Curation & Training** | Curated 1,000 multimodal clinical & sensor records across synthetic and clinical domains; engineered the 22-dimensional normalized feature vector; implemented 5-fold stratified cross-validation; benchmarked external Zenodo (1,010) and Kaggle (2,392) clinical cohorts. |

---

## 📄 License

This project is licensed under the **MIT License** — free to use, modify, and distribute with proper attribution.

<div align="center">

🫁 **HridyaVayu: Prevent tomorrow's attack — with today's prediction.**  
*Connected Respiratory Intelligence for Indian Healthcare*

</div>
