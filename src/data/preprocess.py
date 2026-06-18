import pandas as pd
from sklearn.preprocessing import LabelEncoder


def preprocess_data(df: pd.DataFrame, target_col: str = "Churn") -> pd.DataFrame:
    """
    Preprocessing for Telco churn with feature encoding.
    - trim column names
    - drop obvious ID cols
    - fix TotalCharges to numeric
    - map target Churn to 0/1 if needed
    - encode categorical features
    - simple NA handling
    """
    df = df.copy()
    
    # tidy headers
    df.columns = df.columns.str.strip()  # Remove leading/trailing whitespace

    # drop ids if present
    for col in ["customerID", "CustomerID", "customer_id"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # target to 0/1 if it's Yes/No
    if target_col in df.columns and df[target_col].dtype == "object":
        df[target_col] = df[target_col].str.strip().map({"No": 0, "Yes": 1})

    # TotalCharges often has blanks in this dataset -> coerce to float
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # SeniorCitizen should be 0/1 ints if present
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].fillna(0).astype(int)

    # Encode categorical columns (except target)
    categorical_cols = df.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        if col != target_col:
            # Strip whitespace and encode categorical features
            if df[col].dtype == "object":
                df[col] = df[col].astype(str).str.strip()
                # Use LabelEncoder for categorical features
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])

    # Fill NA numeric values
    num_cols = df.select_dtypes(include=["number"]).columns
    df[num_cols] = df[num_cols].fillna(0)

    return df