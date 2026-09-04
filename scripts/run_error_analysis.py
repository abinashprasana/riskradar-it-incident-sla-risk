"""Phase 5: what the model learns as a case unfolds, and where it stays wrong.

    python scripts/run_error_analysis.py --log uci_servicenow

Four questions, in order of how much they change what you would say about the
model:

1. **SHAP per prefix length.** Which features carry the prediction at k=1, and
   does that change by k=8? A model whose attribution is stable across k is
   learning one thing better; a model whose attribution moves is learning
   something different as evidence arrives.
2. **Who flips.** Cases wrong at k=1 and right at k=8 are where the earliness
   gain actually lives. If they are a coherent group, the curve has a
   mechanism; if they are scattered, it is diffuse improvement.
3. **Who never flips.** Cases still wrong at k=8, and what they share.
4. **The `elapsed_h` question.** Its median at k=1 is two minutes yet it scores
   AUC 0.64 alone, which is implausible as urgency and very plausible as
   creation channel. Cross-tabulating it against `contact_type` and
   `sys_created_by` settles it, and that open item has been carried in the
   model card since the first audit.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from riskradar.adapters import get_adapter  # noqa: E402
from riskradar.encoding import aggregation_encode  # noqa: E402
from riskradar.evaluation import fast_auc  # noqa: E402
from riskradar.prefix_log import build_prefix_log  # noqa: E402
from riskradar.splitting import build_case_index, slice_prefixes, temporal_case_split  # noqa: E402

DENSE = ("2016-02-01", "2016-06-01")
SHAP_SAMPLE = 400  # per k; exact TreeSHAP is fast but not free


def shap_by_k(model, X_test, prefix_test, ks, out_dir) -> pd.DataFrame | None:
    try:
        import shap
    except ImportError:
        print("  shap not installed, skipping attribution")
        return None

    booster = model.named_steps["model"]
    pre = model.named_steps["preprocess"]
    names = list(pre.get_feature_names_out())

    rows = []
    explainer = shap.TreeExplainer(booster)
    for k in ks:
        m = (prefix_test["k"] == k).to_numpy()
        if m.sum() < 50:
            continue
        idx = np.where(m)[0]
        if len(idx) > SHAP_SAMPLE:
            idx = np.random.default_rng(42).choice(idx, SHAP_SAMPLE, replace=False)
        Xk = pre.transform(X_test.iloc[idx])
        vals = explainer.shap_values(Xk)
        mean_abs = np.abs(vals).mean(axis=0)
        for name, v in zip(names, mean_abs):
            rows.append({"k": int(k), "feature": name, "mean_abs_shap": float(v)})

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "shap_by_k.csv", index=False)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="uci_servicenow")
    ap.add_argument("--arm", default="agg")
    args = ap.parse_args()

    run = ROOT / "artifacts" / args.log / args.arm
    adapter = get_adapter(args.log)
    spec = adapter.SPEC

    print(f"[1/4] rebuilding the test fold for {args.log}/{args.arm}")
    events = adapter.load_events(str(ROOT / "incident_event_log.csv"))
    ci = build_case_index(events)
    ci = ci[(ci["case_start_ts"] >= DENSE[0]) & (ci["case_start_ts"] < DENSE[1])]
    split = temporal_case_split(ci)
    labels = adapter.make_labels(events, spec, fit_cases=split.train)
    prefix = build_prefix_log(events, labels, spec, k_max=spec.k_max)
    prefix = prefix[prefix["case_id"].isin(set(ci.index))]
    test = slice_prefixes(prefix, split.test).reset_index(drop=True)

    # Trusted input: this artifact is written by scripts/run_experiment.py in
    # this same repository and is never fetched from anywhere. Joblib
    # unpickling is only a hazard for files of unknown provenance.
    model = joblib.load(run / "model.joblib")
    X_test = aggregation_encode(test)
    scored = pd.read_csv(run / "scored_test.csv")

    print("[2/4] SHAP attribution per prefix length")
    shap_df = shap_by_k(model, X_test, test, range(1, spec.k_max + 1), run)
    if shap_df is not None:
        top = (
            shap_df[shap_df["k"].isin([1, spec.k_max])]
            .pivot(index="feature", columns="k", values="mean_abs_shap")
            .fillna(0)
        )
        top["shift"] = top[spec.k_max] - top[1]
        print(f"      top features at k=1:")
        for f, v in top[1].nlargest(6).items():
            print(f"        {f[:52]:<54} {v:.4f}")
        print(f"      largest gain in importance from k=1 to k={spec.k_max}:")
        for f, v in top["shift"].nlargest(5).items():
            print(f"        {f[:52]:<54} {v:+.4f}")

    print("[3/4] who flips, who never does")
    k_max = spec.k_max
    wide = scored.pivot_table(index="case_id", columns="k", values="p")
    ycase = scored.groupby("case_id")["y"].first()
    full = wide.dropna(subset=[1, k_max])
    ycase = ycase.reindex(full.index)

    thr = 0.5
    right1 = (full[1] >= thr).astype(int) == ycase
    rightK = (full[k_max] >= thr).astype(int) == ycase
    groups = {
        "flipped_to_right": (~right1) & rightK,
        "stayed_right": right1 & rightK,
        "stayed_wrong": (~right1) & (~rightK),
        "flipped_to_wrong": right1 & (~rightK),
    }
    meta = test.groupby("case_id").first()
    prof_rows = []
    for name, mask in groups.items():
        ids = full.index[mask]
        sub = meta.reindex(ids)
        prof_rows.append({
            "group": name,
            "n": int(len(ids)),
            "share": float(len(ids) / len(full)),
            "breach_rate": float(ycase[mask].mean()) if len(ids) else float("nan"),
            "median_trace_len": float(sub["_n_nonterm"].median()) if len(ids) else float("nan"),
            "median_elapsed_h_at_k1": float(sub["num__elapsed_h"].median()) if len(ids) else float("nan"),
            "median_distinct_resources": float(sub["num__n_distinct_resources"].median()) if len(ids) else float("nan"),
            "top_priority": (sub["dyn_cat__priority"].mode().iloc[0]
                             if len(ids) and sub["dyn_cat__priority"].notna().any() else None),
        })
    prof = pd.DataFrame(prof_rows)
    prof.to_csv(run / "error_groups.csv", index=False)
    print(prof.to_string(index=False))

    print("\n[4/4] the elapsed_h question")
    # The full test period, not the fixed cohort. The fixed cohort is 478
    # long cases and would answer a narrower question than the one asked.
    y_all = scored.groupby("case_id")["y"].first()
    k1 = test[test["k"] == 1].set_index("case_id")
    k1 = k1.join(y_all.rename("y_case"), how="inner")
    auc_elapsed = fast_auc(k1["y_case"].to_numpy(dtype=np.int8), k1["num__elapsed_h"].to_numpy())
    print(f"      elapsed_h alone at k=1: AUC {auc_elapsed:.4f}, "
          f"median {k1['num__elapsed_h'].median():.3f}h")

    tab_rows = []
    for col in ("static__contact_type", "static__sys_created_by"):
        if col not in k1.columns:
            continue
        g = k1.groupby(col)["num__elapsed_h"].agg(["count", "median"])
        g = g[g["count"] >= 30].sort_values("median")
        print(f"\n      median elapsed_h at k=1 by {col.replace('static__', '')}:")
        for lvl, r in g.head(6).iterrows():
            br = k1[k1[col] == lvl]["y_case"].mean()
            print(f"        {str(lvl)[:28]:<30} n={int(r['count']):>5}  "
                  f"median {r['median']:>8.3f}h  breach {br:.3f}")
            tab_rows.append({"column": col, "level": str(lvl), "n": int(r["count"]),
                             "median_elapsed_h": float(r["median"]), "breach_rate": float(br)})
    pd.DataFrame(tab_rows).to_csv(run / "elapsed_h_crosstab.csv", index=False)

    summary = {
        "elapsed_h_auc_at_k1": auc_elapsed,
        "elapsed_h_median_at_k1": float(k1["num__elapsed_h"].median()),
        "error_groups": prof.to_dict(orient="records"),
    }
    (run / "error_analysis.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"\nartifacts -> {run}")


if __name__ == "__main__":
    main()
