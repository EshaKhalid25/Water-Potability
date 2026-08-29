# 💧 Water Potability Predictor & Analysis Suite

An end-to-end machine learning pipeline, exploratory data analysis (EDA) suite, and web application that predicts whether water is **safe to drink (potable)** or **not potable** based on 9 chemical and physical properties.

Beyond live predictions, this project incorporates **Explainable AI (SHAP)**, statistical analysis, model comparison, and data visualizations to better understand the factors influencing water potability predictions.

---

## Live Demo & Endpoints

- **Frontend Application (Netlify):** [View Live Web App](https://water-potability-kweg.netlify.app/)
- **Backend API Documentation (Render):** [Explore API Endpoints (/docs)](https://water-potability-kweg.onrender.com/docs)

---

## Tech Stack

### Frontend
* React.js
* Vite
* JavaScript & CSS
* Dynamic input validation based on dataset boundaries

### Backend
* Python
* FastAPI
* Pydantic & REST API
* Uvicorn

### Machine Learning & Data Analysis
* Python
* Pandas & NumPy
* Scikit-Learn
* Random Forest Classifier & XGBoost
* Joblib

### Explainable AI & Visualization
* SHAP (SHapley Additive exPlanations)
* Matplotlib & Seaborn

---

## Dataset

The dataset contains **3,276 water samples** and **9 input features** used to predict water potability.

### Input Features
| Feature | Description |
| :--- | :--- |
| **pH** | Acidity or alkalinity level of water |
| **Hardness** | Concentration of calcium and magnesium salts |
| **Solids** | Total dissolved solids |
| **Chloramines** | Amount of chloramines present |
| **Sulfate** | Sulfate concentration |
| **Conductivity** | Electrical conductivity of water |
| **Organic Carbon** | Amount of organic carbon |
| **Trihalomethanes** | Concentration of trihalomethanes |
| **Turbidity** | Measurement of water clarity |

### Target (`Potability`)
* `0` → Not Potable (1,998 samples / 60.99%)
* `1` → Potable (1,278 samples / 39.01%)

---

## Exploratory Data Analysis (EDA)

A dedicated EDA pipeline was created to analyze the dataset before model training.

### Missing Values & Imputation
* **Sulfate:** 23.84% missing
* **pH:** 14.99% missing
* **Trihalomethanes:** 4.95% missing
* *Note:* Missing values are automatically handled using **median imputation inside the machine learning preprocessing pipeline** (no manual filling during EDA).
* **Duplicates:** No duplicate rows were found.

### Visualizations
The EDA pipeline generates distribution plots, correlation heatmaps, and feature vs. potability charts stored inside:
`machineLearning/eda_results/`

The training workflow also generates SHAP summary plots and feature-importance outputs in:
`machineLearning/explainability_results/`

---

## Data Preprocessing & Model Evaluation

* **Train/Test Split:** 80% training data and 20% testing data with stratified splitting to maintain class distribution.
* **Feature Scaling:** Applied where required (e.g., StandardScaler for Logistic Regression); tree-based models use raw features with median imputation.

### Model Comparison Table
| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 52.44% | 41.46% | 53.13% | 46.58% | 54.74% |
| **Random Forest** | 64.94% | 56.91% | 41.80% | 48.20% | 65.97% |
| **XGBoost** | 65.70% | 60.99% | 33.59% | 43.32% | 65.21% |
| **Tuned Random Forest** | **65.70%** | **58.12%** | **43.36%** | **49.66%** | **66.12%** |
| **Tuned XGBoost** | 63.87% | 55.08% | 40.23% | 46.50% | 62.61% |

> **Selected Model:** **Tuned Random Forest** has the highest F1 Score and ROC-AUC among the tested models, so it was chosen as the final model.
> *Best Parameters:* `n_estimators = 300`, `max_depth = 20`, `min_samples_leaf = 1`, `min_samples_split = 2`

---

## Explainable AI with SHAP

The project integrates **SHAP** to interpret individual predictions:
```text
Hardness       +0.1033
Chloramines    +0.0577
pH             -0.0427
Conductivity   -0.0197
```

A positive SHAP contribution pushes the prediction toward the potable (class 1) direction, while a negative contribution pushes it toward the non-potable (class 0) direction.

## Web Application & API

- **Frontend:** React + Vite interface supporting real-time value validation, confidence scores, and SHAP explanations.
- **Backend Endpoints:**

- `GET /` — Health and root check.
- `POST /predict` — Accepts 9 water parameters as JSON and returns the classification result, confidence percentage, and feature impact analysis.

## Setup & Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/EshaKhalid25/Water-Potability.git
cd Water-Potability
```

### Step 2: Set Up Python Environment & Dependencies

```bash
python -m venv venv
# On Windows:
venv\Scripts\activate

pip install pandas numpy scikit-learn joblib xgboost shap matplotlib seaborn
```

### Step 3: Run EDA & Train the Model

```bash
python eda.py
python train.py
```

This workflow runs the EDA, trains baseline and tuned models, compares performance, generates SHAP explainability plots, and saves the best model to `backend/model/water_model.pkl`. The metrics summary is saved to `machineLearning/model_comparison.csv`.

### Step 4: Start the FastAPI Backend

```bash
cd ../backend
uvicorn main:app --reload
```

*API available at `http://127.0.0.1:8000` (Docs at `/docs`)*

### Step 5: Launch the Frontend
Open a new terminal tab:

```bash
cd frontend
npm install
npm run dev
```

## Project Structure

```text
Water-Potability/
│
├── machineLearning/
│   ├── data/
│   │   └── water_potability.csv
│   ├── eda_results/
│   ├── explainability_results/
│   │   ├── shap_summary.png
│   │   ├── shap_feature_importance.png
│   │   └── shap_importance.csv
│   ├── eda.py
│   ├── train.py
│   ├── model_comparison.csv
│   └── README.md (if present in some variants)
│
├── backend/
│   ├── model/
│   │   ├── water_model.pkl
│   │   └── best_model.txt
│   ├── main.py
│   └── schemas.py
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   └── App.css
│   ├── package.json
│   └── vite.config.js
│
└── README.md
```

## Disclaimer
This project is intended for educational and demonstration purposes. Predictions are generated using machine learning models trained on historical data and should not substitute certified laboratory testing for real-world drinking water safety.

## Author
**Esha Khalid**

Software Developer | Machine Learning & AI Enthusiast

