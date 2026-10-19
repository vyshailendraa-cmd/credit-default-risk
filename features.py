"""Shared data preparation for the credit-risk model.

Used by the notebooks and, later, by the deployed API, so training and
prediction always apply exactly the same steps.
"""
import numpy as np
import pandas as pd

# Excluded from the model on fairness grounds: many lenders and regulators
# do not allow gender in credit decisions.
PROTECTED = ["CODE_GENDER"]
ID_COL = "SK_ID_CURR"
TARGET = "TARGET"
EMPLOYED_PLACEHOLDER = 365243


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Clean known anomalies and add lender-style features. Returns a new frame."""
    df = df.copy()

    # DAYS_EMPLOYED = 365243 is a placeholder (~1,000 years), mostly pensioners/unemployed.
    anomaly = df["DAYS_EMPLOYED"] == EMPLOYED_PLACEHOLDER
    df.loc[anomaly, "DAYS_EMPLOYED"] = np.nan

    new = pd.DataFrame(index=df.index)
    new["DAYS_EMPLOYED_ANOM"] = anomaly.astype(int)
    new["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
    new["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    new["CREDIT_TERM"] = df["AMT_ANNUITY"] / df["AMT_CREDIT"]
    # Borrowing more than the item costs: the signal behind the AMT_CREDIT / AMT_GOODS_PRICE pair in Step 1.
    new["CREDIT_TO_GOODS"] = df["AMT_CREDIT"] / df["AMT_GOODS_PRICE"]
    new["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365.25
    new["EMPLOYED_YEARS"] = -df["DAYS_EMPLOYED"] / 365.25
    new["EMPLOYED_TO_AGE"] = new["EMPLOYED_YEARS"] / new["AGE_YEARS"]

    ext = [c for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"] if c in df.columns]
    if ext:
        new["EXT_SOURCE_MEAN"] = df[ext].mean(axis=1)
        new["EXT_SOURCE_MIN"] = df[ext].min(axis=1)

    df = pd.concat([df, new], axis=1)  # one concat instead of many inserts: no fragmentation warnings
    df = df.replace([np.inf, -np.inf], np.nan)
    return df.drop(columns=[c for c in PROTECTED if c in df.columns])


def split_xy(df: pd.DataFrame):
    """Return features X (categoricals as pandas 'category') and target y."""
    y = df[TARGET]
    X = df.drop(columns=[TARGET, ID_COL])
    cat_cols = X.select_dtypes(include=["object", "string"]).columns
    X[cat_cols] = X[cat_cols].astype("category")
    return X, y
