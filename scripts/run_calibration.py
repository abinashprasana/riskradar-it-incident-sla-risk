"""Phase 2: fit calibrators on the calib fold, evaluate them on test.

    python scripts/run_calibration.py --log uci_servicenow

Reads `scored_calib.csv` and `scored_test.csv` from an experiment run, so no
model is refitted and the calibrators see exactly the scores the reported model
produced. Writes `calibration.csv` (per calibrator x cohort x k) and
`calibration_summary.json` next to them.

The expected result is that Platt, isotonic and beta all leave ECE roughly
where they found it, because the miscalibration is a shifted prior rather than
a mis-shaped score. That is the finding, not a failed experiment -- and the
prior-shift arm is there to test the diagnosis rather than to assume it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from riskradar.calibration import fit_all, prior_shift_offset  # noqa: E402
from riskradar.evaluation import expected_calibration_error, fast_auc  # noqa: E402


def evaluate(df: pd.DataFrame, cal, cohort: str, k_max: int) -> list[dict]:
    d = df[df["_n_nonterm"] >= k_max] if cohort == "fixed" else df
    rows = []
    for k, g in d.groupby("k", sort=True):
        p = cal(g["p"].to_numpy())
        y = g["y"].to_numpy()
        stats = expected_calibration_error(y, p)
        rows.append(
            {
                "cohort": cohort,
                "k": int(k),
                "n_k": int(len(g)),
                "base_rate_k": float(y.mean()),
                "mean_pred": float(p.mean()),
                "auc": fast_auc(y.astype(np.int8), p),
                "brier": float(np.mean((p - y) ** 2)),
                "ece": stats["ece"],
                "mce": stats["mce"],
                "bias": stats["bias"],
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="uci_servicenow")
    ap.add_argument("--arm", default="agg")
    args = ap.parse_args()

    run = ROOT / "artifacts" / args.log / args.arm
    calib = pd.read_csv(run / "scored_calib.csv")
    test = pd.read_csv(run / "scored_test.csv")
    k_max = int(test["k"].max())

    print(f"calibrating {args.log}/{args.arm}")
    print(f"  calib {len(calib):,} rows, base rate {calib['y'].mean():.4f}")
    print(f"  test  {len(test):,} rows, base rate {test['y'].mean():.4f}")
    print(f"  the gap between those two priors is what the calibrators are up against")

    # The EM arm sees test SCORES only. No test label is read here.
    cals = fit_all(calib["p"].to_numpy(), calib["y"].to_numpy(), p_target=test["p"].to_numpy())

    # Diagnostic only, and labelled as such: the shift a *perfect* prior
    # estimate would apply. It answers "is the residual error the prior, or the
    # shape of the score distribution?" -- a question EM's own error otherwise
    # confounds. Never a deployable arm; it reads the test labels.
    cals["prior_shift_oracle"] = prior_shift_offset(
        float(test["y"].mean()), float(calib["y"].mean())
    )

    em = cals.get("prior_shift_em")
    if em is not None:
        print(f"  EM prior estimate: {em.estimated_prior:.4f} "
              f"(calib prior {em.source_prior:.4f}, true test prior {test['y'].mean():.4f})")

    rows = []
    for name, cal in cals.items():
        for cohort in ("variable", "fixed"):
            for r in evaluate(test, cal, cohort, k_max):
                rows.append({"calibrator": name, **r})
    out = pd.DataFrame(rows)
    out.to_csv(run / "calibration.csv", index=False)

    # Prefix-weighted headline per calibrator per cohort.
    summary = {}
    for (name, cohort), g in out.groupby(["calibrator", "cohort"]):
        w = g["n_k"].to_numpy(dtype=float)
        summary[f"{name}|{cohort}"] = {
            "ece": float(np.average(g["ece"], weights=w)),
            "brier": float(np.average(g["brier"], weights=w)),
            "bias": float(np.average(g["bias"], weights=w)),
            "auc": float(np.average(g["auc"].fillna(0.5), weights=w)),
        }
    meta = {
        "log": args.log,
        "arm": args.arm,
        "calib_prior": float(calib["y"].mean()),
        "test_prior": float(test["y"].mean()),
        "em_prior_estimate": float(em.estimated_prior) if em is not None else None,
        "by_calibrator": summary,
    }
    (run / "calibration_summary.json").write_text(json.dumps(meta, indent=2))

    print()
    print("prefix-weighted, VARIABLE cohort (the operational population):")
    print(f"  {'calibrator':<24} {'ECE':>8} {'Brier':>8} {'bias':>8} {'AUC':>8}")
    for name in cals:
        v = summary.get(f"{name}|variable")
        if v:
            print(f"  {name:<24} {v['ece']:>8.4f} {v['brier']:>8.4f} {v['bias']:>+8.4f} {v['auc']:>8.4f}")

    print()
    print("prefix-weighted, FIXED cohort (long cases only):")
    for name in cals:
        v = summary.get(f"{name}|fixed")
        if v:
            print(f"  {name:<24} {v['ece']:>8.4f} {v['brier']:>8.4f} {v['bias']:>+8.4f} {v['auc']:>8.4f}")

    print(f"\nartifacts -> {run}")


if __name__ == "__main__":
    main()
