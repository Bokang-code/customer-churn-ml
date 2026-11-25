import pandas as pd


def preprocess_data(df: pd.DataFrame, target_col: str = "Churn") -> pd.DataFrame:
    """
    Clean the Telco churn dataset into a numeric modelling table.

    Note: this project currently performs preprocessing before the train/test
    split. For a leakage-safe production workflow, move learned imputers and
    encoders into a scikit-learn Pipeline fitted on training data only.
    """
    df = df.copy()
    df.columns = df.columns.str.strip()

    for col in ("customerID", "CustomerID", "customer_id"):
        if col in df.columns:
            df = df.drop(columns=[col])

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' is not present.")

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    if df[target_col].dtype == "object":
        target = df[target_col].astype("string").str.strip().map(
            {"No": 0, "Yes": 1}
        )
        if target.isna().any():
            raise ValueError(
                f"Target '{target_col}' contains values other than Yes/No."
            )
        df[target_col] = target.astype("int64")

    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = pd.to_numeric(
            df["SeniorCitizen"], errors="coerce"
        )

    categorical_cols = [
        col for col in df.select_dtypes(include=["object", "string", "category"]).columns
        if col != target_col
    ]
    df = pd.get_dummies(df, columns=categorical_cols, dtype="int64")

    numeric_cols = df.select_dtypes(include=["number"]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    return df
