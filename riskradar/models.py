"""Candidate classifiers and model selection.

Carries over `ModelResult`, `evaluate_model` and `pick_best` from the original
`model_training.py`, with one change that matters: selection happens on the
calibration fold, never on test.  The original picked the winner by test AUC,
which makes the reported test number a best-of-N and no longer held out.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from .encoding import build_preprocess_pipeline


@dataclass
class ModelResult:
    name: str
    pipeline: Pipeline
    roc_auc: float


def make_candidates(
    seed: int = 42,
    balanced: bool = True,
    pos_weight: float | None = None,
) -> list[tuple[str, object]]:
    """The candidate learners, all CPU-only and all available in this env.

    `balanced=False` drops every class-weighting term, which is the control
    arm for the imbalance question. AUC is invariant to a monotone rescaling
    of scores, so re-weighting is expected to move discrimination barely at
    all while moving calibration a lot -- and demonstrating that is the point
    of running the arm rather than assuming it.
    """
    cw = "balanced" if balanced else None
    cw_forest = "balanced_subsample" if balanced else None
    cands: list[tuple[str, object]] = [
        (
            "logreg",
            LogisticRegression(max_iter=1000, class_weight=cw, random_state=seed),
        ),
        (
            "random_forest",
            RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=5,
                random_state=seed,
                class_weight=cw_forest,
                n_jobs=-1,
            ),
        ),
    ]
    try:
        from xgboost import XGBClassifier

        cands.append(
            (
                "xgboost",
                XGBClassifier(
                    n_estimators=400,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    eval_metric="logloss",
                    tree_method="hist",
                    # XGBoost has no `class_weight`; re-weighting goes here.
                    scale_pos_weight=(pos_weight if balanced and pos_weight else 1.0),
                    random_state=seed,
                    n_jobs=-1,
                ),
            )
        )
    except ImportError:
        pass
    return cands


def train_single_bucket(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    seed: int = 42,
    only: list[str] | None = None,
    balanced: bool = True,
) -> list[tuple[str, Pipeline]]:
    """Fit every candidate as preprocessing + classifier, one model for all k.

    `only` restricts the candidate set by name.  It exists so the encoding
    comparison can hold the learner fixed: comparing best-of-three aggregation
    against logreg-only index encoding would measure the learner as much as the
    encoding.
    """
    # n_neg / n_pos, the standard XGBoost balancing ratio.
    n_pos = float((y_train == 1).sum())
    pos_weight = float((y_train == 0).sum()) / n_pos if n_pos else 1.0
    candidates = make_candidates(seed, balanced=balanced, pos_weight=pos_weight)
    if only:
        candidates = [(n, c) for n, c in candidates if n in set(only)]
        if not candidates:
            raise ValueError(f"no candidates match {only}")
    fitted = []
    for name, clf in candidates:
        pipe = Pipeline([("preprocess", build_preprocess_pipeline(X_train)), ("model", clf)])
        pipe.fit(X_train, y_train)
        fitted.append((name, pipe))
    return fitted


def make_index_bucket_model(seed: int = 42) -> Pipeline:
    """The learner used for every index-encoded prefix-length bucket.

    Deliberately just L2 logistic regression, and deliberately not selected per
    bucket.  Index encoding multiplies categorical width by k while the bucket's
    training set shrinks with k -- UCI bucket 8 has roughly 1,500 training cases
    against several hundred one-hot columns.  Running a model search in that
    regime would mostly be selecting noise on the calibration fold, and the
    point of this arm is to compare encodings, not to find the best learner for
    each of eight separate small problems.
    """
    return Pipeline(
        [
            ("preprocess", None),  # filled per bucket, needs that bucket's columns
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    C=0.5,
                    random_state=seed,
                ),
            ),
        ]
    )


def train_prefix_length_buckets(
    X_by_k: dict[int, pd.DataFrame],
    y_by_k: dict[int, pd.Series],
    seed: int = 42,
    min_rows: int = 100,
    min_frequency: int = 50,
) -> dict[int, Pipeline]:
    """Fit one model per prefix length.

    Buckets with too few rows, or with only one class present, are skipped and
    simply absent from the returned mapping -- an unfitted bucket must produce
    no prediction rather than a fabricated one.
    """
    models: dict[int, Pipeline] = {}
    for k, X in X_by_k.items():
        y = y_by_k[k]
        if X is None or len(X) < min_rows or y.nunique() < 2:
            continue
        pipe = make_index_bucket_model(seed)
        pipe.set_params(
            preprocess=build_preprocess_pipeline(X, min_frequency=min_frequency)
        )
        pipe.fit(X, y)
        models[k] = pipe
    return models


def evaluate_model(pipe: Pipeline, X, y, name: str) -> ModelResult:
    proba = pipe.predict_proba(X)[:, 1]
    return ModelResult(name=name, pipeline=pipe, roc_auc=float(roc_auc_score(y, proba)))


def pick_best(results: list[ModelResult]) -> ModelResult:
    """Highest AUC on whatever fold the caller scored -- which must be calib."""
    return max(results, key=lambda r: r.roc_auc)


def predict_proba(pipe: Pipeline, X: pd.DataFrame) -> np.ndarray:
    return pipe.predict_proba(X)[:, 1]
