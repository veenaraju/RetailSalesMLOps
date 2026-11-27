
# RetailSalesMLOps

## Project Overview

A simple Retail Sales Prediction application
using Machine Learning.

## Objective

Predict retail sales using historical sales
data and Linear Regression.

## Technologies

- Python 3.11
- Pandas
- NumPy
- Scikit-learn
- Joblib

## Project Structure

- data/raw: Original dataset
- src: Python source files
- config: Configuration files
- models: Trained model files
- api: API application

## Setup

Create virtual environment:

python -m venv venv

Activate environment:

venv\Scripts\activate

Install requirements:

pip install -r requirements.txt

## Execute Training

python -m src.train

## Execute Prediction

python -m src.predict

## Model Evaluation

- MAE
- RMSE
- R2 Score

=======================================================================================================
If you want to restart the MLFlow again with an entire new experiment
Execute below commands:
1. Delete the MLflow database and remove SQLite WAL files
-----------------------------------
Remove-Item mlflow.db
Remove-Item mlflow.db-wal -ErrorAction SilentlyContinue
Remove-Item mlflow.db-shm -ErrorAction SilentlyContinue

2. Start MLflow again
------------------------------------
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000

3. Run model training
------------------------------------



## Model Deployment

The trained model is deployed using FastAPI.

### Start API Server

uvicorn api.app:app --reload --host 127.0.0.1 --port 8000

### API Documentation

http://127.0.0.1:8000/docs

### Available Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | / | API information |
| GET | /health | API health |
| POST | /predict | Sales prediction |

### Sample Prediction Input

{
    "Product": "Laptop",
    "Quantity": 3,
    "UnitPrice": 50000,
    "Discount": 5,
    "AdvertisingSpend": 2500
}
