from __future__ import annotations

from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix


@dataclass
class ModelResult:
    name: str
    pipeline: Pipeline
    roc_auc: float


def train_models(preprocess, X_train, y_train, seed: int = 42):
    models = []

    # baseline model (easy to explain)
    logreg = LogisticRegression(max_iter=250, class_weight="balanced", random_state=seed)

    # stronger model (still easy enough to explain)
    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=seed,
        class_weight="balanced_subsample",
        n_jobs=-1
    )

    for name, model in [("logreg", logreg), ("random_forest", rf)]:
        pipe = Pipeline(steps=[
            ("preprocess", preprocess),
            ("model", model)
        ])
        pipe.fit(X_train, y_train)
        models.append((name, pipe))

    return models


def evaluate_model(pipe: Pipeline, X_test, y_test, name: str):
    proba = pipe.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)

    preds = (proba >= 0.5).astype(int)
    cm = confusion_matrix(y_test, preds)
    report = classification_report(y_test, preds, digits=3)

    print(f"\n=== {name} ===")
    print("ROC-AUC:", round(auc, 4))
    print("Confusion matrix:\n", cm)
    print("Report:\n", report)

    return ModelResult(name=name, pipeline=pipe, roc_auc=auc)


def pick_best(results: list[ModelResult]) -> ModelResult:
    return sorted(results, key=lambda r: r.roc_auc, reverse=True)[0]