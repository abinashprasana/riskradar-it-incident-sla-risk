"""Reproduce the original RiskRadar result, as shipped, for comparison.

    python scripts/run_leaky_baseline.py

This imports the untouched root-level `data_processing`, `feature_engineering`
and `model_training` modules rather than re-implementing them, so the number it
produces is the real one and not a strawman built to lose.

What it reproduces, and why each part is wrong:

* one feature vector per COMPLETED incident, including `resolution_hours`,
  `total_events` and whole-trace maxima -- none of which exist at the moment a
  triage decision is actually made;
* a random stratified split, which lets the model learn from the same weeks it
  is tested on, in a log whose breach rate runs 0.642 in March against 0.234 in
  May;
* model selection on the test set itself, making the reported figure a
  best-of-N rather than a held-out estimate.

The output belongs beside the prefix-based result as the "what naive evaluation
claims" row. The gap between them is the finding.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_processing import build_incident_summary, load_event_log  # noqa: E402
from feature_engineering import build_preprocess_pipeline, make_train_test  # noqa: E402


def main() -> None:
    out_dir = ROOT / "artifacts" / "baseline_leaky"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("loading event log via the original pipeline...")
    df_inc = build_incident_summary(load_event_log(str(ROOT / "incident_event_log.csv")))
    print(f"  incidents: {len(df_inc):,}")

    X_train, X_test, y_train, y_test = make_train_test(df_inc)
    pre = build_preprocess_pipeline(X_train)

    # The app trains 100 trees at startup (app.py get_model); run_train.py uses
    # 300. The README quotes the 100-tree figures, so that is what we reproduce.
    pipe = Pipeline(
        [
            ("preprocess", pre),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                ),
            ),
        ]
    )
    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)[:, 1]
    auc = float(roc_auc_score(y_test, proba))
    acc = float(accuracy_score(y_test, (proba >= 0.5).astype(int)))

    summary = {
        "arm": "leaky_baseline_as_shipped",
        "n_incidents": int(len(df_inc)),
        "n_features_raw": int(X_train.shape[1]),
        "split": "random stratified 80/20, seed 42",
        "roc_auc": auc,
        "accuracy": acc,
        "known_defects": [
            "post-hoc whole-trace features (resolution_hours, total_events, *_max)",
            "random split ignores strong month-to-month base-rate drift",
            "model selected on the test set",
        ],
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))

    print(f"\nROC-AUC {auc:.4f}   accuracy {acc:.4f}")
    print(f"artifacts -> {out_dir}")


if __name__ == "__main__":
    main()
