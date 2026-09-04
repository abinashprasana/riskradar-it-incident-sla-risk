"""Phase 4: conformal coverage per prefix length.

    python scripts/run_conformal.py --log uci_servicenow

Calibrates on `scored_calib.csv` and measures empirical coverage on
`scored_test.csv`. Reports across several confidence levels, because a method
that holds at 80% and fails at 95% is telling you something a single level
hides.
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

from riskradar.conformal import (  # noqa: E402
    calibrate,
    coverage_report,
    predict_sets,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="uci_servicenow")
    ap.add_argument("--arm", default="agg")
    ap.add_argument("--levels", default="0.20,0.10,0.05")
    args = ap.parse_args()

    run = ROOT / "artifacts" / args.log / args.arm
    calib = pd.read_csv(run / "scored_calib.csv")
    test = pd.read_csv(run / "scored_test.csv")

    print(f"conformal calibration for {args.log}/{args.arm}")
    print(f"  calib {len(calib):,} rows, test {len(test):,} rows")

    all_rows, meta = [], {}
    for eps in [float(x) for x in args.levels.split(",")]:
        cal = calibrate(
            calib["p"].to_numpy(), calib["y"].to_numpy(), calib["k"].to_numpy(), epsilon=eps
        )
        sets = predict_sets(test["p"].to_numpy(), test["k"].to_numpy(), cal)
        rep = coverage_report(sets, test["y"].to_numpy(), test["k"].to_numpy(), cal)
        rep.insert(0, "epsilon", eps)
        all_rows.append(rep)
        meta[f"eps={eps}"] = cal.as_dict()

        overall = float(sets[np.arange(len(test)), test["y"].to_numpy().astype(int)].mean())
        print(f"\n  nominal coverage {1 - eps:.0%}  (calibration bins: "
              f"{ {k: v for k, v in sorted(cal.n_per_bin.items())} })")
        print(f"  {'k':>3} {'n':>6} {'coverage':>9} {'gap':>8} {'set size':>9} "
              f"{'singleton':>10} {'abstain':>8}")
        for _, r in rep.iterrows():
            flag = "" if abs(r["gap"]) < 0.03 else "  <-- off"
            print(f"  {int(r['k']):>3} {int(r['n']):>6} {r['coverage']:>9.3f} "
                  f"{r['gap']:>+8.3f} {r['mean_set_size']:>9.2f} "
                  f"{r['singleton_rate']:>10.3f} {r['abstain_rate']:>8.3f}{flag}")
        print(f"  overall coverage {overall:.4f} against nominal {1 - eps:.2f}")
        meta[f"eps={eps}"]["overall_coverage"] = overall

    out = pd.concat(all_rows, ignore_index=True)
    out.to_csv(run / "conformal.csv", index=False)
    (run / "conformal_summary.json").write_text(json.dumps(meta, indent=2))

    print("\nThe exchangeability caveat is not a formality here: this split is")
    print("deliberately temporal and the log drifts (train prior 0.56, test 0.40),")
    print("so any coverage shortfall measures that drift rather than a broken method.")
    print(f"\nartifacts -> {run}")


if __name__ == "__main__":
    main()
