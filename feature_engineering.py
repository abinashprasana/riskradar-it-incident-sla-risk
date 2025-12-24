from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer


def make_train_test(
    df_incidents: pd.DataFrame,
    target_col: str = "sla_breached",
    test_size: float = 0.2,
    seed: int = 42
):
    df = df_incidents.copy()

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")

    y = df[target_col].astype(int)

    # drop non-features / leaky fields
    drop_cols = ["number", target_col, "made_sla", "opened_at_min", "resolved_at_max"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    return X_train, X_test, y_train, y_test


def build_preprocess_pipeline(X: pd.DataFrame) -> ColumnTransformer:
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop"
    )

    return pre
