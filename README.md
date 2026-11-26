
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