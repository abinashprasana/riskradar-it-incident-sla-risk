"""Run one prefix-based predictive process monitoring experiment.

    python scripts/run_experiment.py --log uci_servicenow

Writes per-k metrics, a run summary and the full configuration to
`artifacts/<log>/<encoding>/`.  Test data is scored exactly once, at the end,
after the model has been chosen on the calibration fold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from riskradar.adapters import get_adapter
from riskradar.encoding import aggregation_encode, index_encode
from riskradar.evaluation import (
    case_level_metrics,
    earliness_curve,
    prefix_weighted_summary,
    reliability_by_k,
)
from riskradar.models import (
    ModelResult,
    evaluate_model,
    pick_best,
    train_prefix_length_buckets,
    train_single_bucket,
)
from riskradar.prefix_log import assert_no_leakage, build_prefix_log
from riskradar.tuning import random_search
from riskradar.splitting import (
    build_case_index,
    random_case_split,
    slice_prefixes,
    temporal_case_split,
)

DEFAULT_PATHS = {
    "uci_servicenow": "incident_event_log.csv",
    "bpi2013": "csv_files/VINST cases incidents.csv",
}
# 99.3% of UCI cases start between 2016-03 and 2016-05; the 274 later cases are
# a thin, differently-behaved tail that would otherwise dominate the test
# window's calendar span while contributing almost none of its mass.
DENSE_WINDOW = {"uci_servicenow": ("2016-02-01", "2016-06-01")}
# BPI 2013's cases all arrive inside a ten-day window while running a median of
# 7.5 days, so every chronological boundary is straddled by almost every case.
# See riskradar/adapters/bpi2013.py for the measurements behind this.
DEFAULT_SPLIT_MODE = {"uci_servicenow": "temporal", "bpi2013": "random_case"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="uci_servicenow", choices=sorted(DEFAULT_PATHS))
    ap.add_argument("--path", default=None)
    ap.add_argument("--encoding", default="agg", choices=["agg", "index"])
    ap.add_argument("--k-max", type=int, default=None)
    ap.add_argument("--train-frac", type=float, default=0.50)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--calib-frac", type=float, default=0.10)
    ap.add_argument(
        "--split-mode",
        default=None,
        choices=["temporal", "random_case"],
        help="defaults per log; bpi2013 cannot support an out-of-time split",
    )
    ap.add_argument(
        "--learner",
        default=None,
        choices=["logreg", "random_forest", "xgboost"],
        help="force one learner; use logreg on both arms for a controlled encoding comparison",
    )
    ap.add_argument(
        "--dump-scored",
        action="store_true",
        help="also write per-prefix test scores to scored_test.csv",
    )
    ap.add_argument(
        "--imbalance",
        default="balanced",
        choices=["balanced", "none"],
        help="class weighting; `none` is the control arm",
    )
    ap.add_argument(
        "--tune",
        type=int,
        default=0,
        metavar="N",
        help="randomised search of N configs per learner on the val fold (0 = untuned defaults)",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    path = args.path or str(root / DEFAULT_PATHS[args.log])
    out_dir = Path(args.out or root / "artifacts" / args.log / args.encoding)
    out_dir.mkdir(parents=True, exist_ok=True)

    adapter = get_adapter(args.log)
    spec = adapter.SPEC
    k_max = int(args.k_max or spec.k_max)

    print(f"[1/6] loading {args.log} from {path}")
    events = adapter.load_events(path)
    print(f"      {len(events):,} non-terminal events / {events['case_id'].nunique():,} cases")

    # The split is computed WITHOUT labels, because a constructed label (BPI's
    # duration deadline) must be fitted on training cases only.
    case_index = build_case_index(events)
    n_all = len(case_index)
    if args.log in DENSE_WINDOW:
        lo, hi = DENSE_WINDOW[args.log]
        case_index = case_index[
            (case_index["case_start_ts"] >= lo) & (case_index["case_start_ts"] < hi)
        ]
    n_dense = len(case_index)

    split_mode = args.split_mode or DEFAULT_SPLIT_MODE.get(args.log, "temporal")
    print(f"[2/6] {split_mode} split (train {args.train_frac}, calib {args.calib_frac})")
    if split_mode == "temporal":
        split = temporal_case_split(
            case_index,
            args.train_frac,
            args.calib_frac,
            val_frac=args.val_frac,
            seed=args.seed,
        )
    else:
        split = random_case_split(
            case_index,
            args.train_frac,
            args.calib_frac,
            val_frac=args.val_frac,
            seed=args.seed,
        )
    print(f"      {json.dumps(split.as_dict())}")

    labels = adapter.make_labels(events, spec, fit_cases=split.train)
    label_meta = dict(labels.attrs)
    if label_meta:
        print(f"      label: {json.dumps(label_meta, default=float)}")
    case_index = case_index.join(labels["y"], how="inner")

    print(f"[3/6] building prefix log (k_max={k_max})")
    prefix = build_prefix_log(events, labels, spec, k_max=k_max)
    assert_no_leakage(prefix, adapter.FORBIDDEN)
    prefix = prefix[prefix["case_id"].isin(set(case_index.index))]

    folds = {
        n: slice_prefixes(prefix, ix)
        for n, ix in (
            ("train", split.train),
            ("val", split.val),
            ("calib", split.calib),
            ("test", split.test),
        )
    }
    for name, f in folds.items():
        print(f"      {name}: {len(f):,} prefixes / {f['case_id'].nunique():,} cases "
              f"/ base rate {f['y'].mean():.3f}")

    test = folds["test"].copy()
    model_note: dict = {}

    if args.encoding == "agg":
        X = {n: aggregation_encode(f) for n, f in folds.items()}

        if args.tune:
            print(f"[4/6] randomised search, {args.tune} configs per learner, scored on val")
            searches = []
            for learner in ([args.learner] if args.learner else ["logreg", "xgboost"]):
                res = random_search(
                    learner,
                    X["train"], folds["train"]["y"],
                    X["val"], folds["val"]["y"],
                    n_trials=args.tune,
                    seed=args.seed,
                )
                s_ = res.summary()
                print(f"      {learner:9s} best {s_['best_val_auc']:.4f} "
                      f"(median {s_['median_val_auc']:.4f}, worst {s_['worst_val_auc']:.4f}) "
                      f"in {s_['seconds']:.0f}s")
                searches.append(res)

            best_search = max(searches, key=lambda r: r.best_val_auc)
            best = ModelResult(best_search.name, best_search.pipeline, best_search.best_val_auc)
            print(f"      selected: {best.name}")
            model_note = {
                "selected_model": best.name,
                "tuned": True,
                "n_trials_per_learner": args.tune,
                "search": {r.name: r.summary() for r in searches},
                "bucketing": "single",
                "imbalance": args.imbalance,
            }
        else:
            print("[4/6] training candidates on train, selecting on val")
            fitted = train_single_bucket(
                X["train"],
                folds["train"]["y"],
                seed=args.seed,
                only=[args.learner] if args.learner else None,
                balanced=args.imbalance == "balanced",
            )
            # Selection happens on val. calib is reserved for the calibrator
            # and the conformal quantiles -- choosing a model on the same rows
            # that later calibrate it makes the reported calibration optimistic
            # for the same reason selecting on test does.
            results = [
                evaluate_model(pipe, X["val"], folds["val"]["y"], name)
                for name, pipe in fitted
            ]
            for r in results:
                print(f"      {r.name:14s} val AUC {r.roc_auc:.4f}")
            best = pick_best(results)
            print(f"      selected: {best.name}")
            model_note = {
                "selected_model": best.name,
                "tuned": False,
                "val_auc_by_model": {r.name: r.roc_auc for r in results},
                "bucketing": "single",
                "imbalance": args.imbalance,
            }

        print("[5/6] scoring test (touched once)")
        test["p"] = best.pipeline.predict_proba(X["test"])[:, 1]

        # Calibration and conformal both need scores on the calib fold, and a
        # refit would not reproduce this one exactly. Persist the fitted
        # pipeline and its calib scores rather than making later phases
        # re-derive them.
        joblib.dump(best.pipeline, out_dir / "model.joblib", compress=3)
        for fold_name in ("calib", "val"):
            sc = folds[fold_name][["case_id", "k", "_n_nonterm", "y"]].copy()
            sc["p"] = best.pipeline.predict_proba(X[fold_name])[:, 1]
            sc.to_csv(out_dir / f"scored_{fold_name}.csv", index=False)
        print(f"      wrote model.joblib, scored_calib.csv, scored_val.csv")

    else:
        print("[4/6] training one model per prefix length (index encoding)")
        enc = {
            n: {k: index_encode(f, events, k) for k in range(1, k_max + 1)}
            for n, f in folds.items()
        }
        y_by_k = {
            k: folds["train"].loc[folds["train"]["k"] == k, "y"] for k in range(1, k_max + 1)
        }
        models = train_prefix_length_buckets(enc["train"], y_by_k, seed=args.seed)
        widths = {k: int(enc["train"][k].shape[1]) for k in models}
        for k in sorted(models):
            print(f"      k={k}: {len(enc['train'][k]):,} train cases, "
                  f"{widths[k]} raw columns")
        skipped = [k for k in range(1, k_max + 1) if k not in models]
        if skipped:
            print(f"      buckets too small to fit, left unscored: {skipped}")
        model_note = {
            "selected_model": "logreg (fixed per bucket)",
            "bucketing": "prefix_length",
            "buckets_fitted": sorted(models),
            "buckets_skipped": skipped,
            "raw_columns_by_bucket": widths,
        }

        print("[5/6] scoring test (touched once)")
        # A bucket with no model contributes no prediction, so its rows leave
        # the evaluation rather than receiving a fabricated score.
        test["p"] = np.nan
        for k, pipe in models.items():
            Xk = enc["test"][k]
            if Xk is None or Xk.empty:
                continue
            mask = test["k"] == k
            test.loc[mask, "p"] = pipe.predict_proba(Xk)[:, 1]
        dropped = int(test["p"].isna().sum())
        if dropped:
            print(f"      {dropped:,} test prefixes had no fitted bucket, dropped")
        test = test[test["p"].notna()].copy()

    # Where the label is a duration deadline, a prefix whose elapsed time has
    # already passed that deadline has a determined outcome -- scoring it
    # measures clock-reading, not prediction. Restrict the primary curve.
    already_breached_share = None
    if hasattr(adapter, "mark_still_at_risk") and "deadline_hours" in label_meta:
        at_risk = adapter.mark_still_at_risk(test, label_meta["deadline_hours"])
        already_breached_share = float((~at_risk).mean())
        print(f"      excluding {already_breached_share:.1%} of test prefixes "
              f"already past the {label_meta['deadline_hours']:.0f}h deadline")
        test = test[at_risk].copy()

    if args.dump_scored:
        # Per-prefix test scores, for downstream consumers that need the raw
        # predictions rather than aggregate metrics (the web explorer). Written
        # here rather than recomputed elsewhere so nothing has to refit the
        # model to get at what this run already produced.
        cols = ["case_id", "k", "_n_nonterm", "y", "p"]
        test[cols].to_csv(out_dir / "scored_test.csv", index=False)
        print(f"      wrote scored_test.csv ({len(test):,} rows)")

    curves = pd.concat(
        [
            earliness_curve(test, cohort="variable", k_max=k_max),
            earliness_curve(test, cohort="fixed", k_max=k_max),
        ],
        ignore_index=True,
    )
    fixed = curves[curves["cohort"] == "fixed"]
    variable = curves[curves["cohort"] == "variable"]

    cfg_hash = hashlib.sha1(
        json.dumps(vars(args), sort_keys=True, default=str).encode()
    ).hexdigest()[:10]

    summary = {
        "run_id": cfg_hash,
        "run_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "log": args.log,
        "encoding": args.encoding,
        "learner_forced": args.learner,
        **model_note,
        "k_max": k_max,
        "n_cases_total": int(n_all),
        "n_cases_dense_window": int(n_dense),
        "n_cases_dropped_all_terminal": int(events.attrs.get("n_cases_dropped_all_terminal", 0)),
        "breach_rate_dropped_cases": events.attrs.get("breach_rate_dropped"),
        "breach_rate_kept_cases": events.attrs.get("breach_rate_kept"),
        "split_mode": split_mode,
        "split": split.as_dict(),
        "label_construction": label_meta or {"kind": "native flag"},
        "test_prefixes_already_past_deadline": already_breached_share,
        "base_rate_by_fold": {n: float(f["y"].mean()) for n, f in folds.items()},
        "test_variable_cohort": prefix_weighted_summary(variable),
        "test_fixed_cohort": prefix_weighted_summary(fixed),
        "test_case_level_at_k1": case_level_metrics(test, at_k=1),
    }

    print("[6/6] writing artifacts")
    curves.to_csv(out_dir / "per_k.csv", index=False)
    reliability_by_k(test).to_csv(out_dir / "reliability_by_k.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2, default=float))
    (out_dir / "run_config.json").write_text(
        json.dumps(
            {
                "argv": vars(args),
                "python": platform.python_version(),
                "pandas": pd.__version__,
                "numpy": np.__version__,
                "n_sentinel_events_dropped": events.attrs.get("n_sentinel_events_dropped"),
                "dense_window": DENSE_WINDOW.get(args.log),
            },
            indent=2,
        )
    )

    print()
    print(curves.to_string(index=False))
    print()
    print(json.dumps(summary["test_fixed_cohort"], indent=2))
    print(f"\nartifacts -> {out_dir}")


if __name__ == "__main__":
    main()
