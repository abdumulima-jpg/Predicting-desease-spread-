from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd

app = FastAPI(title="Predicting Disease Spread API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained Random Forest model
model = joblib.load("models/trained_model.pkl")

# Load the feature column order
feature_columns = joblib.load("models/feature_columns.pkl")


@app.get("/")
def home():
    return {"message": "Disease Spread Prediction API is running"}


@app.post("/predict")
def predict(features: dict):

    # Convert input data into a DataFrame
    input_data = pd.DataFrame([features])

    # Arrange features in the same order used during training
    input_data = input_data[feature_columns]

    # Make prediction
    prediction = model.predict(input_data)

    return {
        "predicted_cases": float(prediction[0])
    }