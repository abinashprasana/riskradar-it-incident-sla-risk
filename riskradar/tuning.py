"""Randomised hyperparameter search, scored on the validation fold.

Deliberately not `RandomizedSearchCV`. That would run its own internal
cross-validation over the training rows, which does two unwanted things here:
it ignores the val fold this project went to some trouble to create, and its
folds are drawn at random over *prefix rows*, so prefixes of the same case land
on both sides and every reported score is inflated by the exact leak the
splitting module exists to prevent.

So the search is written out: fit on train, score on val, keep the best. Small
enough to read in one sitting, and correct for this data.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from .encoding import build_preprocess_pipeline


@dataclass
class SearchResult:
    name: str
    best_params: dict
    best_val_auc: float
    pipeline: Pipeline
    trials: list[dict] = field(default_factory=list)
    seconds: float = 0.0

    def summary(self) -> dict:
        aucs = [t["val_auc"] for t in self.trials]
        return {
            "learner": self.name,
            "n_trials": len(self.trials),
            "best_val_auc": self.best_val_auc,
            "worst_val_auc": float(min(aucs)) if aucs else float("nan"),
            "median_val_auc": float(np.median(aucs)) if aucs else float("nan"),
            "best_params": self.best_params,
            "seconds": round(self.seconds, 1),
        }


def _logreg_space(rng: np.random.Generator) -> dict:
    return {
        "C": float(10 ** rng.uniform(-3, 2)),
        "class_weight": rng.choice([None, "balanced"]),
        "solver": "lbfgs",
        "max_iter": 2000,
    }


def _xgb_space(rng: np.random.Generator, pos_weight: float) -> dict:
    return {
        "max_depth": int(rng.integers(3, 9)),
        "learning_rate": float(10 ** rng.uniform(-2, np.log10(0.2))),
        "n_estimators": int(rng.integers(150, 801)),
        "min_child_weight": int(rng.integers(1, 21)),
        "subsample": float(rng.uniform(0.6, 1.0)),
        "colsample_bytree": float(rng.uniform(0.6, 1.0)),
        "reg_lambda": float(10 ** rng.uniform(-1, 1.5)),
        # Sampled, not fixed: the imbalance control arm showed weighting costs
        # calibration for no AUC gain, so let the search decide rather than
        # imposing it.
        "scale_pos_weight": float(rng.choice([1.0, pos_weight])),
    }


def _build(name: str, params: dict, X: pd.DataFrame, seed: int) -> Pipeline:
    if name == "logreg":
        clf: Any = LogisticRegression(random_state=seed, **params)
    elif name == "xgboost":
        from xgboost import XGBClassifier

        clf = XGBClassifier(
            eval_metric="logloss",
            tree_method="hist",
            random_state=seed,
            n_jobs=-1,
            **params,
        )
    else:
        raise ValueError(f"no search space for {name!r}")
    return Pipeline([("preprocess", build_preprocess_pipeline(X)), ("model", clf)])


def random_search(
    name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = 60,
    seed: int = 42,
    on_trial: Callable[[int, float, float], None] | None = None,
) -> SearchResult:
    """Sample `n_trials` configurations, fit on train, score on val.

    Returns the best pipeline already fitted, plus every trial, so the search
    itself can be reported. A search whose worst and best trials are barely
    apart says the learner is insensitive to its hyperparameters on this data,
    which is worth knowing and is invisible if only the winner is kept.
    """
    rng = np.random.default_rng(seed)
    n_pos = float((y_train == 1).sum())
    pos_weight = float((y_train == 0).sum()) / n_pos if n_pos else 1.0

    best: SearchResult | None = None
    trials: list[dict] = []
    t0 = time.time()

    for i in range(n_trials):
        params = _logreg_space(rng) if name == "logreg" else _xgb_space(rng, pos_weight)
        try:
            pipe = _build(name, params, X_train, seed)
            pipe.fit(X_train, y_train)
            auc = float(roc_auc_score(y_val, pipe.predict_proba(X_val)[:, 1]))
        except Exception as exc:  # a bad corner of the space must not kill the run
            trials.append({"trial": i, "params": params, "val_auc": float("nan"), "error": str(exc)})
            continue

        trials.append({"trial": i, "params": params, "val_auc": auc})
        if best is None or auc > best.best_val_auc:
            best = SearchResult(name, params, auc, pipe)
        if on_trial:
            on_trial(i, auc, best.best_val_auc)

    if best is None:
        raise RuntimeError(f"every trial failed for {name!r}")

    best.trials = [t for t in trials if t["val_auc"] == t["val_auc"]]
    best.seconds = time.time() - t0
    return best
