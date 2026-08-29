# 💧 Water Potability Predictor

An end-to-end machine learning pipeline and web application that predicts whether water is safe to drink (potable) based on 9 chemical properties. The model determines water safety based on real-world historical data rather than hard-coded rules.

## Tech Stack
*   **Frontend:** React.js (Vite) featuring dynamic input validation based on strict dataset boundaries.
*   **Backend:** FastAPI (Python) acting as a bridge to process REST API requests and serve the trained `.pkl` model.
*   **Machine Learning:** Scikit-Learn (Random Forest Classifier) and Pandas for automated data cleaning.

## Model Performance
*   **Accuracy:** 65.85%
*   **F1 Score:** 40.74% (Prioritizes a balance between precision and recall for safe drinking water).
*   **Top Features:** The Random Forest algorithm identified pH (13%), Sulfate (12.4%), and Hardness (12.1%) as the most critical determining factors.
*   **Data Cleaning:** Missing values in the dataset (specifically pH, Sulfate, and Trihalomethanes) are automatically handled via median imputation during the training phase.

## Setup Instructions
*   **Step 1: Install Dependencies**  
    Create a Python virtual environment and install the required ML and API libraries by running `pip install pandas scikit-learn joblib fastapi uvicorn`.
*   **Step 2: Train the Model**  
    Navigate to the `ml/` directory and execute `python train.py`. This reads the dataset, trains 100 decision trees, and exports the `water_model.pkl` file.
*   **Step 3: Start the Backend**  
    Navigate to the `backend/` directory and execute `uvicorn main:app --reload`. The API will be live at `http://127.0.0.1:8000`.
*   **Step 4: Launch the Frontend**  
    Open a new terminal, navigate to the `frontend/` directory, run `npm install`, and then execute `npm run dev` to launch the React user interface.

## 🌐 Live Demo
* **Frontend App (Netlify):** [View Live Application](https://water-potability-kweg.netlify.app/)
* **Backend API Docs (Render):** [Explore API Endpoints (/docs)](https://water-potability-kweg.onrender.com/docs)