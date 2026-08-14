# Predicting Disease Spread

## Project Overview

This project uses machine learning to predict disease cases and support early warning of disease spread.

The project includes a machine learning model, a FastAPI backend, and a frontend interface for making predictions.

## Problem Statement

Disease outbreaks can spread rapidly and place pressure on healthcare systems. Predicting future disease cases can help provide early information that may support monitoring and decision-making.

## Project Goal

The goal of this project is to develop a machine learning system that predicts disease cases based on available disease-related and population data.

## Dataset

The dataset used in this project contains disease-related and population information, including country, continent, population, date, cases, recoveries, deaths, and tests.

The data was cleaned and preprocessed before being used for exploratory data analysis and machine learning model training.


## Dataset

The project uses a COVID-19 dataset containing information such as:

- Country
- Continent
- Population
- Cases
- Recoveries
- Deaths
- Tests
- Date-related information

## Machine Learning Models

Three machine learning approaches were considered for this project:

1. **Random Forest Regressor** — an ensemble learning model that combines multiple decision trees.
2. **XGBoost Regressor** — a gradient boosting model designed for strong predictive performance.
3. **LSTM (Long Short-Term Memory)** — a deep learning model suitable for sequential and time-dependent data.

The models are compared based on their predictive performance using evaluation metrics such as:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- R² Score

## Project Structure

```text
ml-project/
├── data/
│   └── raw/
│       └── data.csv
├── models/
│   ├── trained_model.pkl
│   └── feature_columns.pkl
├── notebooks/
├── outputs/
│   └── plots/
│       └── .gitkeep
├── src/
│   ├── preprocess.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── utils.py
├── frontend/
│   └── index.html
├── main.py
├── config.py
├── requirements.txt
└── README.md

## How to Run Locally

### 1. Create and activate the virtual environment

```powershell
py -3.13 -m venv .venv
.venv\Scripts\activate

### 2. Install the required libraries

```powershell
pip install -r requirements.txt

### 3. Start the FastAPI backend

```powershell
uvicorn main:app --reload

The API will run at:

http://127.0.0.1:8000

### 4. Open the frontend

Open `frondend/index.html` using Live Server in VS Code.

The frontend communicates with the FastAPI backend to make disease-case predictions.

## API Endpoint

The prediction endpoint is:

`POST /predict`

The API returns the predicted number of disease cases.

## Evaluation

The models are evaluated using:

- MAE
- MSE
- R² Score

## Project Status

The machine learning model, FastAPI backend, and frontend interface have been developed and connected for local testing.
