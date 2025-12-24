from __future__ import annotations

import joblib
from dotenv import load_dotenv
load_dotenv()

from data_processing import load_event_log, build_incident_summary
from feature_engineering import make_train_test, build_preprocess_pipeline
from model_training import train_models, evaluate_model, pick_best

CSV_PATH = "incident_event_log.csv"
MODEL_OUT = "best_model.joblib"


def main():
    print("Loading event log...")
    df_events = load_event_log(CSV_PATH)
    print("Rows:", len(df_events), "| Cols:", len(df_events.columns))

    print("Building incident summary (one row per ticket)...")
    df_inc = build_incident_summary(df_events)
    print("Incidents:", len(df_inc))

    if "sla_breached" not in df_inc.columns:
        raise ValueError("sla_breached not found. Is 'made_sla' present in the CSV?")

    X_train, X_test, y_train, y_test = make_train_test(df_inc, target_col="sla_breached")
    pre = build_preprocess_pipeline(X_train)

    print("Training models...")
    models = train_models(pre, X_train, y_train)

    results = []
    for name, pipe in models:
        results.append(evaluate_model(pipe, X_test, y_test, name=name))

    best = pick_best(results)
    print(f"\nBest model: {best.name} | AUC: {best.roc_auc:.4f}")

    joblib.dump(best.pipeline, MODEL_OUT)
    print("Saved ->", MODEL_OUT)


if __name__ == "__main__":
    main()