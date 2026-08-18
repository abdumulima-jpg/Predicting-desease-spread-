# Predicting Disease Spread

## Project Overview

**Predicting Disease Spread** is a machine learning project designed to predict disease cases and support early warning and monitoring of disease spread.

The system combines machine learning models with a **FastAPI backend** and a web-based frontend that allows users to enter disease-related information and receive a predicted number of cases.

## Problem Statement

Disease outbreaks can spread rapidly and place significant pressure on healthcare systems. Having a system that can estimate disease cases from available data can provide useful information for monitoring disease trends and supporting early decision-making.

## Project Goal

The goal of this project is to develop and compare machine learning models for predicting disease cases using disease-related, population, testing, and time-based features.

The project compares three different approaches:

1. **Random Forest Regressor**
2. **XGBoost Regressor**
3. **Long Short-Term Memory (LSTM)**

## Dataset

The project uses a COVID-19 dataset containing information including:

* Country
* Continent
* Population
* Cases
* Recoveries
* Deaths
* Tests
* Year
* Month
* Day of Month
* Hour

The target variable for prediction is **Cases**.

## Machine Learning Models

### 1. Random Forest Regressor

Random Forest is an ensemble learning algorithm that combines multiple decision trees to produce predictions.

### 2. XGBoost Regressor

XGBoost is a gradient boosting algorithm that builds a sequence of decision trees to improve predictive performance.

### 3. LSTM

Long Short-Term Memory (LSTM) is a type of recurrent neural network designed to learn patterns in sequential and time-dependent data.

## Model Evaluation

The models were evaluated using:

* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* R² Score

### Evaluation Results

| Model         |          MAE |                    MSE |    R² Score |
| ------------- | -----------: | ---------------------: | ----------: |
| Random Forest | 1,925,660.01 |  45,792,209,828,091.09 |  **0.9276** |
| XGBoost       | 2,785,785.75 | 122,602,559,373,312.00 |  **0.8063** |
| LSTM          | 7,695,540.50 | 692,047,275,098,112.00 | **-0.0936** |

Based on the evaluation results, **Random Forest Regressor achieved the best performance with an R² score of 0.9276** and is the model currently used by the FastAPI prediction API.

## Project Structure

```text
ml-project/
│
├── data/
│   ├── raw/
│   │   └── data.csv
│   ├── processed/
│   │   └── processed_data.csv
│   └── external/
│
├── front-end/
│   └── index.html
│
├── models/
│   ├── continent_encoder.pkl
│   ├── country_encoder.pkl
│   ├── feature_columns.pkl
│   ├── lstm_model.keras
│   ├── scaler.pkl
│   ├── trained_model.pkl
│   └── xgboost_model.pkl
│
├── notebooks/
│   ├── exploration.ipynb
│   └── other project notebooks
│
├── outputs/
│   └── plots/
│       └── .gitkeep
├── src/
│   ├── preprocess.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── generate_predictions.py
│   └── utils.py
├── frontend/
│   └── index.html
├── main.py
├── config.py
├── main.py
├── requirements.txt
└── README.md
```

## How to Run Locally

### 1. Create the virtual environment

```powershell
py -3.13 -m venv .venv
```

### 2. Activate the virtual environment

```powershell
.venv\Scripts\activate
```

### 3. Install the required libraries

```powershell
pip install -r requirements.txt
```

### 4. Start the FastAPI backend

```powershell
python -m uvicorn main:app --reload
```

The backend will run at:

```text
http://127.0.0.1:8000
```

### 5. Open the API documentation

FastAPI provides interactive API documentation at:

```text
http://127.0.0.1:8000/docs
```

The main prediction endpoint is:

```text
POST /predict
```

### 6. Open the frontend

Open `front-end/index.html` using **Live Server** in VS Code.

The local frontend is available at:

```text
http://127.0.0.1:5500/front-end/index.html
```

The frontend communicates with the FastAPI backend to send disease-related information and display the predicted number of cases.

## Prediction Output

After entering the required information, the system returns a predicted number of disease cases.

Example:

```text
Predicted Cases: 3904.91
```

The frontend also displays:

```text
AI-Powered Disease Spread Prediction
```

## API Response

The `/predict` endpoint returns the predicted number of cases in JSON format:

```json
{
    "predicted_cases": 3904.91
}
```

## Project Outputs

The project generates the following outputs:

* `outputs/predictions.csv` — contains actual and predicted case values.
* `outputs/metrics.json` — contains the evaluation results for the three models.
* `outputs/plots/` — contains project visualizations.
* `outputs/test_data.csv` — contains the test data used for model evaluation.

## Deployment

The project is being prepared for public deployment so that users can access the disease spread prediction system through the internet.

### Live Frontend

*To be added after deployment.*

### Live API

*To be added after deployment.*

## Project Status

The machine learning model, FastAPI backend, and frontend interface have been developed and connected for local testing.
