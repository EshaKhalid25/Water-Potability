import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import os

print("💧 Starting Water Potability ML Pipeline...")

# 1. Load Dataset
print("Loading dataset...")
df = pd.read_csv('data/water_potability.csv')

# ==========================================
# 2. DATA CLEANING PROCESS HAPPENS HERE
# ==========================================
print("Cleaning data (handling missing values)...")
# We fill the empty spots with the 'median' value of that specific column
# to ensure our model doesn't get confused by missing data.
df['ph'] = df['ph'].fillna(df['ph'].median())
df['Sulfate'] = df['Sulfate'].fillna(df['Sulfate'].median())
df['Trihalomethanes'] = df['Trihalomethanes'].fillna(df['Trihalomethanes'].median())

# 3. Train/Test Split
print("Splitting data into training and testing sets...")
X = df.drop('Potability', axis=1)
y = df['Potability']

# We use stratify=y to ensure the train and test sets have the same ratio of potable/not-potable water
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Train Random Forest Model
print("Training the Random Forest model (this takes a few seconds)...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 5. Evaluate Model
print("\n--- Evaluating Model ---")
y_pred = rf_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print("------------------------\n")

# 6. Save Model for FastAPI
print("Saving trained model...")
# Ensure the backend directory exists based on our folder structure
os.makedirs('../backend/model', exist_ok=True)
joblib.dump(rf_model, '../backend/model/water_model.pkl')

print("Pipeline Complete! Model saved to backend/model/water_model.pkl")