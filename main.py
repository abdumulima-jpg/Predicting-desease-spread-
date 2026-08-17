from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd


# ============================================================
# API CONFIGURATION
# ============================================================

app = FastAPI(title="Predicting Disease Spread API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

model = joblib.load("models/trained_model.pkl")

feature_columns = joblib.load(
    "models/feature_columns.pkl"
)


# ============================================================
# PREDICTION INPUT SCHEMA
# ============================================================

class PredictionInput(BaseModel):
    country: int
    continent: int
    population: float
    Recovered: float
    Deaths: float
    Tests: float
    year: int
    month: int
    day_of_month: int
    hour: int


# ============================================================
# HOME ENDPOINT
# ============================================================

@app.get("/")
def home():
    return {
        "message": "Predicting Disease Spread API is running"
    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict(features: PredictionInput):

    # Convert the validated input into a dictionary
    feature_data = features.model_dump()

    # Convert input data into a DataFrame
    input_data = pd.DataFrame([feature_data])

    # Arrange features in the same order used during training
    input_data = input_data[feature_columns]

    # Make prediction
    prediction = model.predict(input_data)

    return {
        "predicted_cases": float(prediction[0])
    }