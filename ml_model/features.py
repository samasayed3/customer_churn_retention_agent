"""Shared feature definitions (used by BOTH training and prediction -> no mismatch)."""
import pandas as pd

BASE_NUMERIC = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
ENGINEERED = ["NumAddons", "AvgCharge", "IsNew", "AutoPay"]
NUMERIC = BASE_NUMERIC + ENGINEERED
CATEGORICAL = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]
ADDONS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
          "TechSupport", "StreamingTV", "StreamingMovies"]
FEATURES = NUMERIC + CATEGORICAL


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["NumAddons"] = (d[ADDONS] == "Yes").sum(axis=1)          # how many add-on services
    d["AvgCharge"] = d["TotalCharges"] / (d["tenure"] + 1)      # avg paid per month so far
    d["IsNew"] = (d["tenure"] <= 6).astype(int)                 # new customer flag
    d["AutoPay"] = d["PaymentMethod"].str.contains("automatic").astype(int)
    return d
