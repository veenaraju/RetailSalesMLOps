import yaml
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import subprocess

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

# --------------------------------------------------
# 1. Load configuration
# --------------------------------------------------

with open("config/config.yaml", "r") as file:
    config = yaml.safe_load(file)


# Read configuration
raw_path = config["data"]["raw_path"]
target_column = config["data"]["target_column"]

test_size = config["training"]["test_size"]
random_state = config["training"]["random_state"]
algorithm = config["training"]["algorithm"]

model_path = config["model"]["save_path"]

tracking_uri = config["mlflow"]["tracking_uri"]
experiment_name = config["mlflow"]["experiment_name"]


# --------------------------------------------------
# 2. Read DVC dataset version
# --------------------------------------------------

dvc_file = Path("data/raw/retail_sales.csv.dvc")

with open(dvc_file, "r") as file:
    dvc_data = yaml.safe_load(file)

dataset_hash = dvc_data["outs"][0]["md5"]

dataset_version = dataset_hash[:8]

print("\nDataset Version:", dataset_version)
print("Dataset DVC Hash:", dataset_hash)


# --------------------------------------------------
# 3. Read Git commit ID
# --------------------------------------------------

git_commit_id = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()

print("Git Commit ID:", git_commit_id)


# --------------------------------------------------
# 4. Initialize MLflow
# --------------------------------------------------

mlflow.set_tracking_uri(tracking_uri)

mlflow.set_experiment(experiment_name)


# --------------------------------------------------
# 5. Load dataset
# --------------------------------------------------

df = load_data(raw_path)


# --------------------------------------------------
# 6. Split dataset
# --------------------------------------------------

X_train, X_test, y_train, y_test = preprocess_data(
    df, target_column, test_size, random_state
)


# --------------------------------------------------
# 7. Identify feature columns
# --------------------------------------------------

numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()

categorical_features = X_train.select_dtypes(include=["object", "str"]).columns.tolist()


# --------------------------------------------------
# 8. Numeric preprocessing
# --------------------------------------------------

numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])


# --------------------------------------------------
# 9. Categorical preprocessing
# --------------------------------------------------

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]
)


# --------------------------------------------------
# 10. Combine preprocessing
# --------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)


# --------------------------------------------------
# 11. Select algorithm
# --------------------------------------------------

if algorithm == "LinearRegression":
    regressor = LinearRegression()
else:
    raise ValueError("Unsupported algorithm")


# --------------------------------------------------
# 12. Create ML pipeline
# --------------------------------------------------

model = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", regressor)])

model_name = algorithm

run_name = f"RetailSales_{model_name}_{dataset_version}"


# --------------------------------------------------
# 13. Start MLflow run
# --------------------------------------------------

# Get Git commit ID
git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()


# Read DVC metadata
dvc_file = raw_path + ".dvc"

with open(dvc_file, "r") as file:
    dvc_metadata = yaml.safe_load(file)

dataset_hash = dvc_metadata["outs"][0].get("md5")

if dataset_hash is None:
    dataset_hash = dvc_metadata["outs"][0].get("hash")


print("Git Commit:", git_commit)
print("DVC Dataset Hash:", dataset_hash)

with mlflow.start_run(run_name=run_name):

    mlflow.set_tag("git_commit", git_commit)

    mlflow.set_tag("dataset_dvc_hash", dataset_hash)

    mlflow.set_tag("dataset_path", raw_path)
    # Log model and dataset information
    mlflow.log_param("model_name", model_name)

    mlflow.log_param("dataset_version", dataset_version)

    mlflow.log_param("dataset_dvc_hash", dataset_hash)

    mlflow.log_param("git_commit_id", git_commit_id)

    # Log training configuration
    mlflow.log_param("algorithm", algorithm)

    mlflow.log_param("test_size", test_size)

    mlflow.log_param("random_state", random_state)

    mlflow.log_param("training_records", len(X_train))

    mlflow.log_param("testing_records", len(X_test))

    mlflow.log_param("target_column", target_column)

    # --------------------------------------------------
    # 14. Train model
    # --------------------------------------------------

    model.fit(X_train, y_train)

    print("\nModel training completed")

    # --------------------------------------------------
    # 15. Generate predictions
    # --------------------------------------------------

    y_pred = model.predict(X_test)

    # --------------------------------------------------
    # 16. Calculate evaluation metrics
    # --------------------------------------------------

    mae = mean_absolute_error(y_test, y_pred)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    r2 = r2_score(y_test, y_pred)

    # --------------------------------------------------
    # 17. Log metrics to MLflow
    # --------------------------------------------------

    mlflow.log_metric("MAE", mae)

    mlflow.log_metric("RMSE", rmse)

    mlflow.log_metric("R2", r2)

    # --------------------------------------------------
    # 18. Save model locally
    # --------------------------------------------------

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_path)

    # --------------------------------------------------
    # 19. Log model to MLflow
    # --------------------------------------------------

    mlflow.sklearn.log_model(
        sk_model=model, name="model", skops_trusted_types=["numpy.dtype"]
    )

    # --------------------------------------------------
    # 20. Display results
    # --------------------------------------------------

    print("\nModel Evaluation Results")

    print("------------------------")

    print("MAE:", round(mae, 2))

    print("RMSE:", round(rmse, 2))

    print("R2:", round(r2, 4))

    print("\nDataset Version:", dataset_version)

    print("Git Commit ID:", git_commit_id)

    print("\nLocal Model Saved:", model_path)

    print("\nMLflow Run ID:", mlflow.active_run().info.run_id)

    print("\nMLflow experiment logging completed")
