from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
from schemas import WaterFeatures

app = FastAPI(title="Water Potability API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Load ML model from the file
model = joblib.load("model/water_model.pkl")

# 2. Endpoint: would receive water features and return potability prediction
@app.post("/predict")
def predict_potability(features: WaterFeatures):
    # Convert Pydantic model to DataFrame for prediction
    input_data = pd.DataFrame([features.dict()])
    
    # Get prediction and probability from the model
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0].max()
    
    # Would send results back to react frontend
    return {
        "prediction": int(prediction),
        "result": "Potable" if prediction == 1 else "Not Potable",
        "confidence": round(float(probability), 2)
    }