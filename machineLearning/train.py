import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv("data/water_potability.csv")

print("\n========== DATASET INFO ==========")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

print("\nMissing values:")
print(df.isnull().sum())

print("\nTarget distribution:")
print(df["Potability"].value_counts())

print("\nTarget percentage:")
print(df["Potability"].value_counts(normalize=True) * 100)


# ============================================================
# 2. FEATURES & TARGET
# ============================================================

X = df.drop("Potability", axis=1)
y = df["Potability"]


# ============================================================
# 3. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# 4. PREPROCESSING
# ============================================================

# IMPORTANT:
# Imputer is fitted ONLY on training data through the pipeline.
# This prevents data leakage.

logistic_preprocessor = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

tree_preprocessor = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])


# ============================================================
# 5. BASELINE MODELS
# ============================================================

models = {

    "Logistic Regression": Pipeline([
        ("preprocessor", logistic_preprocessor),
        (
            "model",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42
            )
        )
    ]),

    "Random Forest": Pipeline([
        ("preprocessor", tree_preprocessor),
        (
            "model",
            RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            )
        )
    ]),

    "XGBoost": Pipeline([
        ("preprocessor", tree_preprocessor),
        (
            "model",
            XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric="logloss"
            )
        )
    ])
}


# ============================================================
# 6. TRAIN + EVALUATE BASELINE MODELS
# ============================================================

results = []
trained_models = {}

for name, pipeline in models.items():

    print("\n" + "=" * 70)
    print(f"TRAINING BASELINE: {name}")
    print("=" * 70)

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_probability = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )
    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )
    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )
    roc_auc = roc_auc_score(
        y_test,
        y_probability
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC-AUC": roc_auc
    })

    trained_models[name] = pipeline

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


# ============================================================
# 7. HYPERPARAMETER TUNING
# ============================================================

print("\n\n")
print("=" * 80)
print("HYPERPARAMETER TUNING")
print("=" * 80)


# ------------------------------------------------------------
# Random Forest tuning
# ------------------------------------------------------------

rf_pipeline = Pipeline([
    ("preprocessor", tree_preprocessor),
    (
        "model",
        RandomForestClassifier(
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
    )
])

rf_params = {
    "model__n_estimators": [200, 300],
    "model__max_depth": [None, 10, 20],
    "model__min_samples_split": [2, 5],
    "model__min_samples_leaf": [1, 2]
}


print("\nTuning Random Forest...")

rf_grid = GridSearchCV(
    rf_pipeline,
    rf_params,
    cv=3,
    scoring="f1",
    n_jobs=-1,
    verbose=1
)

rf_grid.fit(X_train, y_train)

best_rf = rf_grid.best_estimator_

print("\nBest Random Forest parameters:")
print(rf_grid.best_params_)


# ------------------------------------------------------------
# XGBoost tuning
# ------------------------------------------------------------

xgb_pipeline = Pipeline([
    ("preprocessor", tree_preprocessor),
    (
        "model",
        XGBClassifier(
            random_state=42,
            eval_metric="logloss"
        )
    )
])

xgb_params = {
    "model__n_estimators": [200, 300],
    "model__max_depth": [3, 4, 5],
    "model__learning_rate": [0.03, 0.05, 0.1],
    "model__subsample": [0.8, 1.0],
    "model__colsample_bytree": [0.8, 1.0]
}


print("\nTuning XGBoost...")

xgb_grid = GridSearchCV(
    xgb_pipeline,
    xgb_params,
    cv=3,
    scoring="f1",
    n_jobs=-1,
    verbose=1
)

xgb_grid.fit(X_train, y_train)

best_xgb = xgb_grid.best_estimator_

print("\nBest XGBoost parameters:")
print(xgb_grid.best_params_)


# ============================================================
# 8. EVALUATE TUNED MODELS
# ============================================================

tuned_models = {
    "Tuned Random Forest": best_rf,
    "Tuned XGBoost": best_xgb
}


for name, pipeline in tuned_models.items():

    print("\n" + "=" * 70)
    print(f"EVALUATING: {name}")
    print("=" * 70)

    y_pred = pipeline.predict(X_test)
    y_probability = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_probability
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC-AUC": roc_auc
    })

    trained_models[name] = pipeline

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


# ============================================================
# 9. MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(results)

print("\n\n")
print("=" * 80)
print("FINAL MODEL COMPARISON")
print("=" * 80)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# 10. SELECT BEST MODEL
# ============================================================

best_model_name = results_df.loc[
    results_df["F1"].idxmax(),
    "Model"
]

best_model = trained_models[best_model_name]


# ============================================================
# 11. SHAP EXPLAINABILITY
# ============================================================

import shap
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import os


print("\n" + "=" * 80)
print("SHAP EXPLAINABILITY")
print("=" * 80)


# ============================================================
# 1. CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs("explainability_results", exist_ok=True)


# ============================================================
# 2. GET TUNED RANDOM FOREST
# ============================================================

rf_pipeline = best_rf


# ============================================================
# 3. GET RANDOM FOREST MODEL
# ============================================================

rf_model = rf_pipeline.named_steps["model"]


# ============================================================
# 4. PREPARE TEST DATA
# ============================================================

# Get preprocessing step automatically

if "imputer" in rf_pipeline.named_steps:

    imputer = rf_pipeline.named_steps["imputer"]

    X_test_processed = imputer.transform(X_test)

elif "preprocessor" in rf_pipeline.named_steps:

    preprocessor = rf_pipeline.named_steps["preprocessor"]

    X_test_processed = preprocessor.transform(X_test)

else:

    raise ValueError(
        "No imputer/preprocessor found in Random Forest pipeline."
    )


# Convert back to DataFrame

X_test_processed = pd.DataFrame(
    X_test_processed,
    columns=X_test.columns,
    index=X_test.index
)


# ============================================================
# 5. CREATE SHAP EXPLAINER
# ============================================================

explainer = shap.TreeExplainer(rf_model)

shap_values = explainer.shap_values(
    X_test_processed
)


# ============================================================
# 6. GET POSITIVE CLASS SHAP VALUES
# ============================================================

if isinstance(shap_values, list):

    # Older SHAP versions
    shap_values_positive = shap_values[1]

else:

    # Newer SHAP versions

    if len(shap_values.shape) == 3:

        shap_values_positive = shap_values[:, :, 1]

    else:

        shap_values_positive = shap_values


# ============================================================
# 7. SHAP SUMMARY PLOT
# ============================================================

shap.summary_plot(
    shap_values_positive,
    X_test_processed,
    show=False
)

plt.title(
    "SHAP Feature Impact on Water Potability"
)

plt.tight_layout()

plt.savefig(
    "explainability_results/shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 8. SHAP BAR PLOT
# ============================================================

shap.summary_plot(
    shap_values_positive,
    X_test_processed,
    plot_type="bar",
    show=False
)

plt.title(
    "SHAP Feature Importance"
)

plt.tight_layout()

plt.savefig(
    "explainability_results/shap_feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 9. CALCULATE FEATURE IMPORTANCE
# ============================================================

mean_shap_values = (
    abs(shap_values_positive)
    .mean(axis=0)
)


shap_importance = pd.DataFrame({

    "Feature": X_test_processed.columns,

    "Mean_Absolute_SHAP": mean_shap_values

}).sort_values(
    "Mean_Absolute_SHAP",
    ascending=False
)


# ============================================================
# 10. SAVE SHAP IMPORTANCE
# ============================================================

shap_importance.to_csv(
    "explainability_results/shap_importance.csv",
    index=False
)


# ============================================================
# 11. PRINT RESULTS
# ============================================================

print("\nSHAP Feature Importance:")

print(
    shap_importance.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


print("\nSHAP results saved successfully:")

print(
    "1. explainability_results/shap_summary.png"
)

print(
    "2. explainability_results/shap_feature_importance.png"
)

print(
    "3. explainability_results/shap_importance.csv"
)


print("\nSHAP EXPLAINABILITY COMPLETED!")

print("\n" + "=" * 80)
print(f"BEST MODEL: {best_model_name}")
print("=" * 80)


# ============================================================
# 11. CREATE MODEL DIRECTORY
# ============================================================

model_directory = "../backend/model"

os.makedirs(
    model_directory,
    exist_ok=True
)


# ============================================================
# 12. SAVE BEST MODEL
# ============================================================

model_path = os.path.join(
    model_directory,
    "water_model.pkl"
)

joblib.dump(
    best_model,
    model_path
)

print("\nModel saved successfully:")
print(model_path)


# ============================================================
# 13. SAVE MODEL COMPARISON
# ============================================================

comparison_path = "model_comparison.csv"

results_df.to_csv(
    comparison_path,
    index=False
)

print("\nModel comparison saved:")
print(comparison_path)


# ============================================================
# 14. SAVE BEST MODEL NAME
# ============================================================

with open(
    os.path.join(
        model_directory,
        "best_model.txt"
    ),
    "w"
) as file:

    file.write(best_model_name)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 80)
print("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
print("=" * 80)
