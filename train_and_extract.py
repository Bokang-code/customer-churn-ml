"""
Train model and show where to find artifacts for bundling into Docker image.
"""
import os
import mlflow
from pathlib import Path
from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.models.train import train_model

# Paths
DATA_PATH = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"

print("=== Training Model ===")
print("Loading data...")
df = load_data(DATA_PATH)
print(f"Data shape: {df.shape}")

print("Preprocessing data...")
df = preprocess_data(df, target_col="Churn")
print(f"Data preprocessed shape: {df.shape}")

print("Training model with MLflow (this may take a minute)...")
train_model(df, target_col="Churn")
print("✓ Model training complete!")

# Find the latest run
print("\n=== Extracting Model Info ===")
client = mlflow.tracking.MlflowClient()

# Get all experiments
experiments = client.search_experiments()
print(f"Found {len(experiments)} experiment(s)")

for exp in experiments:
    runs = client.search_runs(experiment_ids=[exp.experiment_id], order_by=["start_time DESC"])
    if not runs.empty:
        latest_run = runs[0]
        run_id = latest_run.run_id
        print(f"\nLatest run: {run_id}")
        print(f"Experiment: {exp.name}")
        
        # List artifacts
        artifacts = client.list_artifacts(run_id)
        print(f"Artifacts:")
        for artifact in artifacts:
            print(f"  - {artifact.path}")
        
        print(f"\n✓ Model artifacts are located at:")
        print(f"  mlruns/{exp.experiment_id}/{run_id}/artifacts/")
        print(f"\nTo bundle this model into Docker:")
        print(f"  1. Copy mlruns/{exp.experiment_id}/{run_id}/artifacts/ to src/serving/model/{run_id}/artifacts/")
        print(f"  2. Update Dockerfile COPY lines with run_id: {run_id}")
        break

