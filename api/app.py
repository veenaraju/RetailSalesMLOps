import joblib
import yaml
import pandas as pd

from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel, Field

# Project directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# Load configuration
config_path = PROJECT_ROOT / "config" / "config.yaml"

with open(config_path, "r") as file:
    config = yaml.safe_load(file)


# Load trained model
model_path = PROJECT_ROOT / config["model"]["save_path"]

model = joblib.load(model_path)


# Create FastAPI application
app = FastAPI(
    title="Retail Sales Prediction API",
    description="Predict retail sales using a trained ML model",
    version="1.0",
)


# Input schema
class RetailSalesInput(BaseModel):

    Product: str = Field(min_length=1)
    Quantity: int = Field(gt=0)
    UnitPrice: float = Field(gt=0)
    Discount: float = Field(ge=0, le=100)
    AdvertisingSpend: float = Field(ge=0)


# Home endpoint
@app.get("/")
def home():

    return {"message": "Retail Sales Prediction API is running"}


# Health endpoint
@app.get("/health")
def health():

    return {"status": "healthy"}


# Prediction endpoint
@app.post("/predict")
def predict_sales(data: RetailSalesInput):

    input_data = pd.DataFrame(
        [
            {
                "Product": data.Product,
                "Quantity": data.Quantity,
                "UnitPrice": data.UnitPrice,
                "Discount": data.Discount,
                "AdvertisingSpend": data.AdvertisingSpend,
            }
        ]
    )

    prediction = model.predict(input_data)

    return {"predicted_sales": round(float(prediction[0]), 2)}
