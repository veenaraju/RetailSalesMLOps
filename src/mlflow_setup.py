import mlflow
import yaml

# Load configuration
with open("config/config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Configure tracking URI
tracking_uri = config["mlflow"]["tracking_uri"]

mlflow.set_tracking_uri(tracking_uri)

# Create or select experiment
experiment_name = config["mlflow"]["experiment_name"]

mlflow.set_experiment(experiment_name)

print("MLflow tracking initialized")

print("Tracking URI:", mlflow.get_tracking_uri())

print("Experiment:", experiment_name)
