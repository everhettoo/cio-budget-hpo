"""Data preparation: clean -> remove PII -> balance -> 70/30 split -> encode.

Methods 3.1. Every step is printed by notebook 01 and checked by tests.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config

NUMERIC = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
TARGET = "Churn"
PII = ["customerID"]


def load_raw(path=config.RAW_CSV) -> pd.DataFrame:
    """Read the Telco CSV exactly as downloaded (7,043 rows)."""
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """TotalCharges -> number; drop the 11 blank rows; drop customerID (PII)."""
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"])
    df = df.drop(columns=PII)
    df[TARGET] = (df[TARGET] == "Yes").astype(int)
    return df.reset_index(drop=True)


def balance(df: pd.DataFrame, seed=config.DATA_SEED) -> pd.DataFrame:
    """Downsample the majority class (no churn) to the size of the minority."""
    pos = df[df[TARGET] == 1]
    neg = df[df[TARGET] == 0].sample(n=len(pos), random_state=seed)
    return pd.concat([pos, neg]).sample(frac=1, random_state=seed).reset_index(drop=True)


def split(df: pd.DataFrame, seed=config.DATA_SEED):
    """70/30 stratified hold-out, fixed once for all 300 runs."""
    X, y = df.drop(columns=[TARGET]), df[TARGET].to_numpy()
    return train_test_split(X, y, test_size=config.VAL_SIZE, stratify=y, random_state=seed)


def _encoder(scale: bool, categorical):
    """Build the encoder: one-hot for text columns; for SVM, also scale every column."""
    num = StandardScaler() if scale else "passthrough"
    cat_steps = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    if scale:  # SVM: scale every column, dummies included
        cat_steps = make_pipeline(cat_steps, StandardScaler())
    return ColumnTransformer([("num", num, NUMERIC), ("cat", cat_steps, categorical)])


@dataclass
class Dataset:
    """Holds the prepared data for one model: training and validation features, their labels, and the feature names."""
    X_train: np.ndarray
    X_val: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    feature_names: list


def prepare(model: str, path=config.RAW_CSV) -> Dataset:
    """Full pipeline for one model family. SVM is scaled; trees are not.

    Encoders are fitted on the training rows only (no leakage into validation).
    """
    df = balance(clean(load_raw(path)))
    X_tr, X_va, y_tr, y_va = split(df)
    categorical = [c for c in X_tr.columns if c not in NUMERIC]
    enc = _encoder(scale=(model == "svm"), categorical=categorical)
    Xt = enc.fit_transform(X_tr)
    Xv = enc.transform(X_va)
    names = list(enc.get_feature_names_out())
    return Dataset(Xt.astype(np.float64), Xv.astype(np.float64), y_tr, y_va, names)


def summary(path=config.RAW_CSV) -> dict:
    """Row counts at every step (notebook 01 and tests)."""
    raw = load_raw(path)
    cl = clean(raw)
    bal = balance(cl)
    X_tr, X_va, y_tr, y_va = split(bal)
    return {
        "raw_rows": len(raw),
        "clean_rows": len(cl),
        "dropped_blank_totalcharges": len(raw) - len(cl),
        "pii_columns_left": [c for c in PII if c in cl.columns],
        "churn_after_clean": int(cl[TARGET].sum()),
        "balanced_rows": len(bal),
        "balanced_churn": int(bal[TARGET].sum()),
        "train_rows": len(X_tr),
        "val_rows": len(X_va),
        "train_churn_share": float(np.mean(y_tr)),
        "val_churn_share": float(np.mean(y_va)),
    }
