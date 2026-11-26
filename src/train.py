import yaml
import joblib
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from src.preprocessing import load_data, preprocess_data

# Load configuration
with open("config/config.yaml", "r") as file:
    config = yaml.safe_load(file)


# Read configuration values
raw_path = config["data"]["raw_path"]
target_column = config["data"]["target_column"]
test_size = config["training"]["test_size"]
random_state = config["training"]["random_state"]
model_path = config["model"]["save_path"]


# Load dataset
df = load_data(raw_path)


# Split dataset
X_train, X_test, y_train, y_test = preprocess_data(
    df, target_column, test_size, random_state
)


# Identify numeric and categorical columns
numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()

categorical_features = X_train.select_dtypes(include="object").columns.tolist()


# Numeric preprocessing
numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])


# Categorical preprocessing
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]
)


# Combine preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)


# Create ML pipeline
model = Pipeline(
    steps=[("preprocessor", preprocessor), ("regressor", LinearRegression())]
)


# Train model
model.fit(X_train, y_train)

print("\nModel training completed")


# Generate predictions
y_pred = model.predict(X_test)


# Evaluate model
mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

r2 = r2_score(y_test, y_pred)


print("\nModel Evaluation Results")
print("------------------------")
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 4))


# Save model
Path(model_path).parent.mkdir(parents=True, exist_ok=True)

joblib.dump(model, model_path)

print("\nModel saved successfully:", model_path)
