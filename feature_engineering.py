import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer


def make_train_test(df: pd.DataFrame, target_col: str = "sla_breached", test_size: float = 0.2, seed: int = 42):
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")

    y = df[target_col].astype(int)
    drop = [c for c in ["number", target_col, "made_sla", "opened_at_min", "resolved_at_max"] if c in df.columns]
    X = df.drop(columns=drop)

    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


def build_preprocess_pipeline(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]

    return ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),
            ("cat", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", OneHotEncoder(handle_unknown="ignore")),
            ]), cat_cols),
        ],
        remainder="drop",
    )
