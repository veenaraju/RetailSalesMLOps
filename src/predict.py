import joblib
import yaml
import pandas as pd

# Load configuration
with open("config/config.yaml", "r") as file:
    config = yaml.safe_load(file)


# Load trained model
model_path = config["model"]["save_path"]

model = joblib.load(model_path)


# New retail transaction
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


print("\nNew Retail Transaction:")
print(new_data)

print("\nPredicted Sales:")
print(round(prediction[0], 2))
