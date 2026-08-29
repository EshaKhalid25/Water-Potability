from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import joblib
import pandas as pd
import shap

from schemas import WaterFeatures


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="Water Potability API",
    description="ML API for predicting water potability",
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load(
    "model/water_model.pkl"
)


# ============================================================
# FEATURE NAMES
# ============================================================

FEATURE_NAMES = [
    "ph",
    "Hardness",
    "Solids",
    "Chloramines",
    "Sulfate",
    "Conductivity",
    "Organic_carbon",
    "Trihalomethanes",
    "Turbidity"
]


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def root():

    return {
        "message": "Water Potability API is running",
        "status": "success"
    }


# ============================================================
# PREDICTION
# ============================================================

@app.post("/predict")
def predict_potability(
    features: WaterFeatures
):

    # --------------------------------------------------------
    # Convert request data into DataFrame
    # --------------------------------------------------------

    input_data = pd.DataFrame(
        [features.model_dump()]
    )

    # Make sure feature order is exactly the same
    # as during model training

    input_data = input_data[
        FEATURE_NAMES
    ]


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = model.predict(
        input_data
    )[0]


    # --------------------------------------------------------
    # Prediction Probability
    # --------------------------------------------------------

    probabilities = model.predict_proba(
        input_data
    )[0]

    probability = probabilities[
        int(prediction)
    ]


    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    result = (
        "Potable"
        if prediction == 1
        else "Not Potable"
    )


    # --------------------------------------------------------
    # SHAP EXPLANATION
    # --------------------------------------------------------

    explanation = []


    try:

        # Get the actual Random Forest model
        # from the saved pipeline

        rf_model = model.named_steps["model"]


        # Get preprocessing step

        if "imputer" in model.named_steps:

            imputer = model.named_steps["imputer"]

            processed_data = imputer.transform(
                input_data
            )

        elif "preprocessor" in model.named_steps:

            preprocessor = model.named_steps[
                "preprocessor"
            ]

            processed_data = preprocessor.transform(
                input_data
            )

        else:

            processed_data = input_data


        # Convert to DataFrame

        processed_data = pd.DataFrame(
            processed_data,
            columns=FEATURE_NAMES
        )


        # SHAP explainer

        explainer = shap.TreeExplainer(
            rf_model
        )

        shap_values = explainer.shap_values(
            processed_data
        )


        # Handle different SHAP versions

        if isinstance(shap_values, list):

            values = shap_values[1][0]

        elif len(shap_values.shape) == 3:

            values = shap_values[0, :, 1]

        else:

            values = shap_values[0]


        # Create feature explanation

        for feature, value in zip(
            FEATURE_NAMES,
            values
        ):

            explanation.append({
                "feature": feature,
                "impact": round(
                    float(value),
                    4
                ),
                "direction": (
                    "positive"
                    if value > 0
                    else "negative"
                )
            })


        # Sort by absolute impact

        explanation.sort(
            key=lambda x: abs(
                x["impact"]
            ),
            reverse=True
        )


        # Return top 5

        explanation = explanation[:5]


    except Exception as error:

        print(
            "SHAP explanation error:",
            error
        )


    # ========================================================
    # RESPONSE
    # ========================================================

    return {

        "prediction": int(
            prediction
        ),

        "result": result,

        "confidence": round(
            float(probability),
            4
        ),

        "confidence_percentage": round(
            float(probability) * 100,
            2
        ),

        "explanation": explanation
    }
