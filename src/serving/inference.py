"""
Inference pipeline for Telco Customer Churn prediction.

This module provides the predict() function that:
1. Takes customer data dictionary
2. Loads the trained model from MLflow
3. Preprocesses features
4. Returns churn prediction
"""

import pandas as pd
import mlflow.xgboost
from src.data.preprocess import preprocess_data
from src.data.load_data import load_data


def predict(data: dict) -> str:
    """
    Predict customer churn for a single customer.
    
    Args:
        data (dict): Customer data with keys matching CustomerData schema
                     (gender, Partner, Dependents, ..., TotalCharges)
    
    Returns:
        str: "Likely to churn" or "Not likely to churn"
    """
    try:
        # Convert dict to DataFrame (single row)
        df = pd.DataFrame([data])
        
        # Apply preprocessing
        df = preprocess_data(df, target_col=None)
        
        # Load trained model from MLflow (latest model in registry)
        # In production, you'd specify exact model version
        try:
            model = mlflow.xgboost.load_model("models:/telco-churn-model/production")
        except Exception:
            # Fallback: load most recent model from runs
            experiment = mlflow.get_experiment_by_name("churn-experiment")
            if experiment:
                runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
                if not runs.empty:
                    latest_run = runs.iloc[0]
                    model = mlflow.xgboost.load_model(f"runs:/{latest_run.run_id}/model")
                else:
                    raise RuntimeError("No trained models found in MLflow")
            else:
                raise RuntimeError("No experiment found. Please train a model first.")
        
        # Make prediction
        prediction = model.predict(df)
        
        # Convert to human-readable output
        churn_prob = prediction[0]
        return "Likely to churn" if churn_prob == 1 else "Not likely to churn"
    
    except Exception as e:
        raise RuntimeError(f"Prediction failed: {str(e)}")
