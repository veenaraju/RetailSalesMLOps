import mlflow.sklearn
import pandas as pd

# Replace with your actual MLflow Run ID
run_id = "94c6627290a34a4b9ca0495532046583"

# Load model from MLflow
model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

# New transaction
new_data = pd.DataFrame(
    [
        {
            "Product": "Laptop",
            "Quantity": 3,
            "UnitPrice": 50000,
            "Discount": 5,
            "AdvertisingSpend": 2500,
        }
    ]
)

# Generate prediction
prediction = model.predict(new_data)

print("Predicted Sales:", round(prediction[0], 2))
