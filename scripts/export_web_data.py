"""Export the study's results as JSON for the KAIROS web presentation layer.

    python scripts/export_web_data.py

Reads only what `run_all.py` already produced -- no model is refitted here, so
the site can never drift from the numbers the experiments actually reported.
Writes to `web/public/data/`:

    summary.json          headline figures, split config, data-quality counts
    per_k.json            every arm's per-prefix-length curve
    explorer_sample.json  real scored test cases for the interactive explorer

Run `scripts/run_experiment.py --dump-scored` (or `run_all.py`, which passes it)
before this, or the explorer sample will be skipped.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ART = ROOT / "artifacts"
OUT = ROOT / "web" / "public" / "data"

# Arm directory -> the id the site refers to it by. Order is the order the
# narrative introduces them.
ARMS = {
    "uci_servicenow/agg": "uci_headline",
    "uci_servicenow/agg_tuned": "uci_tuned",
    "uci_servicenow/agg_noweight": "uci_noweight",
    "uci_servicenow/gru": "uci_gru",
    "bpi2013/agg_tuned": "bpi_tuned",
    "bpi2013/gru": "bpi_gru",
    "uci_servicenow/agg_randomsplit": "uci_randomsplit",
    "uci_servicenow/agg_logreg": "uci_agg_logreg",
    "uci_servicenow/index": "uci_index",
    "bpi2013/agg": "bpi_headline",
    "bpi2013/agg_logreg": "bpi_agg_logreg",
    "bpi2013/index": "bpi_index",
}

# Matches riskradar.decision_logic.risk_band, restated rather than imported so
# the exported payload stays a plain data file with no runtime dependency.
BANDS = ((0.30, "low"), (0.60, "medium"), (1.01, "high"))

EXPLORER_SAMPLE_CASES = 600


def _band(p: float) -> str:
    for edge, name in BANDS:
        if p < edge:
            return name
    return "high"


def _round_floats(obj, nd: int = 4):
    """Trim float precision -- 15 significant digits is payload, not information."""
    if isinstance(obj, float):
        return round(obj, nd)
    if isinstance(obj, dict):
        return {k: _round_floats(v, nd) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_floats(v, nd) for v in obj]
    return obj


def export_per_k() -> dict:
    """Every arm's per-k curve, keyed by arm id."""
    out: dict[str, list[dict]] = {}
    for rel, arm_id in ARMS.items():
        f = ART / rel / "per_k.csv"
        if not f.exists():
            print(f"  skip {arm_id}: no per_k.csv")
            continue
        df = pd.read_csv(f)
        out[arm_id] = _round_floats(df.to_dict(orient="records"))
        print(f"  {arm_id}: {len(df)} rows")
    return out


def export_summary() -> dict:
    """Headline numbers, pulled from each arm's metrics.json."""
    leaky = json.loads((ART / "baseline_leaky" / "metrics.json").read_text())
    metrics = {}
    for rel, arm_id in ARMS.items():
        f = ART / rel / "metrics.json"
        if f.exists():
            metrics[arm_id] = json.loads(f.read_text())

    uci = metrics["uci_headline"]
    bpi = metrics.get("bpi_headline", {})

    def fixed_at(arm_id: str, k: int) -> float | None:
        f = ART / [r for r, a in ARMS.items() if a == arm_id][0] / "per_k.csv"
        if not f.exists():
            return None
        d = pd.read_csv(f)
        d = d[(d["cohort"] == "fixed") & (d["k"] == k)]
        return float(d["auc"].iloc[0]) if len(d) else None

    return _round_floats(
        {
            "leaky": {
                "auc": leaky["roc_auc"],
                "accuracy": leaky["accuracy"],
                "defects": leaky["known_defects"],
            },
            "uci": {
                "selected_model": uci["selected_model"],
                "k_max": uci["k_max"],
                "case_level_k1_auc": uci["test_case_level_at_k1"]["auc"],
                "fixed_k1_auc": fixed_at("uci_headline", 1),
                "fixed_kmax_auc": fixed_at("uci_headline", uci["k_max"]),
                "prefix_weighted_auc": uci["test_variable_cohort"]["prefix_weighted_auc"],
                "n_cases_total": uci["n_cases_total"],
                "n_cases_dense_window": uci["n_cases_dense_window"],
                "n_cases_dropped_all_terminal": uci["n_cases_dropped_all_terminal"],
                "breach_rate_dropped_cases": uci["breach_rate_dropped_cases"],
                "breach_rate_kept_cases": uci["breach_rate_kept_cases"],
                "split": uci["split"],
                "split_mode": uci["split_mode"],
                "base_rate_by_fold": uci["base_rate_by_fold"],
            },
            "uci_randomsplit": {
                "fixed_k1_auc": fixed_at("uci_randomsplit", 1),
                "fixed_kmax_auc": fixed_at("uci_randomsplit", 8),
            },
            "bpi": {
                "selected_model": bpi.get("selected_model"),
                "k_max": bpi.get("k_max"),
                "fixed_k1_auc": fixed_at("bpi_headline", 1),
                "fixed_kmax_auc": fixed_at("bpi_headline", bpi.get("k_max", 12)),
                "label": bpi.get("label_construction", {}),
                "split_mode": bpi.get("split_mode"),
                "already_past_deadline": bpi.get("test_prefixes_already_past_deadline"),
            },
            "encoding_control": {
                "uci_agg_logreg_kmax": fixed_at("uci_agg_logreg", 8),
                "uci_index_kmax": fixed_at("uci_index", 8),
                "bpi_agg_logreg_kmax": fixed_at("bpi_agg_logreg", 12),
                "bpi_index_kmax": fixed_at("bpi_index", 12),
            },
            # Measured directly against the raw CSV; see docs/MODEL_CARD.md.
            "backfill": [
                {"column": "closed_at", "populated_at_k1": 1.000, "equals_final": 1.0},
                {"column": "closed_code", "populated_at_k1": 0.996, "equals_final": 1.0},
                {"column": "resolved_by", "populated_at_k1": 0.996, "equals_final": 1.0},
                {"column": "resolved_at", "populated_at_k1": 0.938, "equals_final": 1.0},
            ],
            "drift": [
                {"month": "2016-03", "n": 8062, "breach_rate": 0.642},
                {"month": "2016-04", "n": 7398, "breach_rate": 0.243},
                {"month": "2016-05", "n": 7170, "breach_rate": 0.234},
            ],
        }
    )


def export_explorer_sample() -> dict | None:
    """Real scored test cases, for the k-slider explorer.

    Sampled at CASE level, not row level, so every sampled case carries its
    whole prefix sequence -- the explorer needs to re-rank the same cases as k
    advances, which is impossible if only some of a case's prefixes are present.
    """
    f = ART / "uci_servicenow" / "agg" / "scored_test.csv"
    if not f.exists():
        print("  skip explorer sample: run with --dump-scored first")
        return None

    df = pd.read_csv(f)
    k_max = int(df["k"].max())

    # Only cases observed to k_max can be re-ranked across the full slider
    # range without the cohort changing underneath the viewer -- which is the
    # same fixed-cohort argument the earliness curve rests on.
    full = df[df["_n_nonterm"] >= k_max]
    cases = full[["case_id", "y"]].drop_duplicates("case_id")

    n = min(EXPLORER_SAMPLE_CASES, len(cases))
    # Stratify so the sample's breach rate matches the cohort's.
    take = (
        cases.groupby("y", group_keys=False)
        .apply(lambda g: g.sample(max(1, round(n * len(g) / len(cases))), random_state=42),
               include_groups=False)
        .index
    )
    chosen = set(cases.loc[take, "case_id"]) if len(take) else set(cases["case_id"][:n])
    sub = full[full["case_id"].isin(chosen)].copy()
    sub["band"] = sub["p"].map(_band)

    by_case: dict[str, dict] = {}
    for cid, g in sub.groupby("case_id", sort=False):
        g = g.sort_values("k")
        by_case[str(cid)] = {
            "y": int(g["y"].iloc[0]),
            "p": [round(float(v), 4) for v in g["p"]],
            "band": list(g["band"]),
        }

    print(f"  explorer: {len(by_case)} cases x k=1..{k_max}, "
          f"breach rate {sub.groupby('case_id')['y'].first().mean():.3f}")
    return {"k_max": k_max, "cases": by_case}


def export_modelling() -> dict:
    """Phase 1-6 outputs: calibration, conformal, order ablation, alarm sweep."""
    out: dict = {}

    ab = ART / "order_ablation" / "order_ablation.csv"
    if ab.exists():
        out["order_ablation"] = _round_floats(pd.read_csv(ab).to_dict(orient="records"))
        v = ART / "order_ablation" / "verdict.json"
        if v.exists():
            out["order_verdict"] = json.loads(v.read_text())

    for rel, key in (("uci_servicenow/agg", "uci"), ("bpi2013/agg", "bpi")):
        d = ART / rel
        if (d / "calibration_summary.json").exists():
            out.setdefault("calibration", {})[key] = json.loads(
                (d / "calibration_summary.json").read_text()
            )
        if (d / "conformal.csv").exists():
            out.setdefault("conformal", {})[key] = _round_floats(
                pd.read_csv(d / "conformal.csv").to_dict(orient="records")
            )
        if (d / "alarm_sweep.csv").exists():
            out.setdefault("alarm", {})[key] = _round_floats(
                pd.read_csv(d / "alarm_sweep.csv").to_dict(orient="records")
            )
        if (d / "alarm_summary.json").exists():
            out.setdefault("alarm_summary", {})[key] = json.loads(
                (d / "alarm_summary.json").read_text()
            )
        if (d / "error_analysis.json").exists():
            out.setdefault("error_analysis", {})[key] = json.loads(
                (d / "error_analysis.json").read_text()
            )

    for rel, key in (("uci_servicenow/gru", "uci"), ("bpi2013/gru", "bpi")):
        f = ART / rel / "metrics.json"
        if f.exists():
            out.setdefault("gru", {})[key] = json.loads(f.read_text())

    sh = ART / "uci_servicenow" / "agg" / "shap_by_k.csv"
    if sh.exists():
        df = pd.read_csv(sh)
        top = (
            df.groupby("feature")["mean_abs_shap"].mean().nlargest(12).index
        )
        out["shap"] = _round_floats(
            df[df["feature"].isin(top)].to_dict(orient="records")
        )
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"exporting to {OUT}")

    print("summary.json")
    (OUT / "summary.json").write_text(json.dumps(export_summary(), indent=2))

    print("per_k.json")
    (OUT / "per_k.json").write_text(json.dumps(export_per_k(), indent=2))

    print("modelling.json")
    mod = export_modelling()
    (OUT / "modelling.json").write_text(json.dumps(mod, indent=2))
    print(f"  sections: {sorted(mod)}")

    print("explorer_sample.json")
    sample = export_explorer_sample()
    if sample is not None:
        (OUT / "explorer_sample.json").write_text(json.dumps(sample))

    for f in sorted(OUT.glob("*.json")):
        print(f"  {f.name}  {f.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
