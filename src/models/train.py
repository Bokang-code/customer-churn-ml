import mlflow
import mlflow.xgboost
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def train_model(df: pd.DataFrame, target_col: str):
    """
    Train an XGBoost churn classifier and log reproducible evaluation results.

    The split is stratified so the churn/non-churn ratio is represented in
    both subsets. Metrics are calculated on the held-out test subset.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' is not present.")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    if y.isna().any():
        raise ValueError("Target column contains missing values.")
    if y.nunique() != 2:
        raise ValueError(
            f"Expected a binary target, found {y.nunique()} distinct values."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model_params = {
        "n_estimators": 300,
        "learning_rate": 0.1,
        "max_depth": 6,
        "random_state": 42,
        "n_jobs": -1,
        "eval_metric": "logloss",
    }
    model = XGBClassifier(**model_params)

    with mlflow.start_run():
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, zero_division=0),
            "recall": recall_score(y_test, predictions, zero_division=0),
            "f1": f1_score(y_test, predictions, zero_division=0),
        }
        # ROC-AUC is undefined if the held-out subset contains one class.
        if y_test.nunique() == 2:
            metrics["roc_auc"] = roc_auc_score(y_test, probabilities)

        mlflow.log_params(model_params)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("split_random_state", 42)
        mlflow.log_param("stratified_split", True)
        mlflow.log_metrics(metrics)

        cm = confusion_matrix(y_test, predictions, labels=[0, 1])
        mlflow.log_dict(
            {
                "labels": ["not_churn", "churn"],
                "matrix": cm.tolist(),
                "rows": ["actual_not_churn", "actual_churn"],
                "columns": ["predicted_not_churn", "predicted_churn"],
            },
            "evaluation/confusion_matrix.json",
        )

        mlflow.xgboost.log_model(model, "model")

        train_ds = mlflow.data.from_pandas(df, source="training_data")
        mlflow.log_input(train_ds, context="training")

        print("Model evaluation on held-out test data:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")
        print("Confusion matrix (actual rows x predicted columns):")
        print(cm)

    return model, metrics
