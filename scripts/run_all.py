"""Run every experimental arm, then render the figures.

    python scripts/run_all.py

Roughly fifteen minutes on CPU. The GRU arms (scripts/run_gru.py) are run
separately because five seeds on two logs takes another ten. Each arm writes to its own directory under
`artifacts/`, so a failure part-way through leaves the completed arms intact.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ARMS: list[tuple[str, list[str]]] = [
    ("leaky baseline (reproduces the as-shipped 0.9674)", ["scripts/run_leaky_baseline.py"]),
    ("UCI, aggregation, out-of-time  [headline]",
     ["scripts/run_experiment.py", "--log", "uci_servicenow", "--dump-scored"]),
    ("UCI, aggregation, random split  [ablation: price of honest evaluation]",
     ["scripts/run_experiment.py", "--log", "uci_servicenow", "--split-mode", "random_case",
      "--out", "artifacts/uci_servicenow/agg_randomsplit"]),
    ("UCI, aggregation, logreg  [encoding control]",
     ["scripts/run_experiment.py", "--log", "uci_servicenow", "--learner", "logreg",
      "--out", "artifacts/uci_servicenow/agg_logreg"]),
    ("UCI, index encoding, prefix-length buckets",
     ["scripts/run_experiment.py", "--log", "uci_servicenow", "--encoding", "index"]),
    ("BPI 2013, aggregation  [cross-log]",
     ["scripts/run_experiment.py", "--log", "bpi2013", "--dump-scored"]),
    ("BPI 2013, aggregation, logreg  [encoding control]",
     ["scripts/run_experiment.py", "--log", "bpi2013", "--learner", "logreg",
      "--out", "artifacts/bpi2013/agg_logreg"]),
    ("BPI 2013, index encoding, prefix-length buckets",
     ["scripts/run_experiment.py", "--log", "bpi2013", "--encoding", "index"]),
    ("UCI, no class weighting  [imbalance control]",
     ["scripts/run_experiment.py", "--log", "uci_servicenow", "--imbalance", "none",
      "--out", "artifacts/uci_servicenow/agg_noweight"]),
    # --- Phase 1: tuned baselines ---
    ("UCI, tuned (60 configs on val)",
     ["scripts/run_experiment.py", "--log", "uci_servicenow", "--tune", "60",
      "--dump-scored", "--out", "artifacts/uci_servicenow/agg_tuned"]),
    ("BPI 2013, tuned (60 configs on val)",
     ["scripts/run_experiment.py", "--log", "bpi2013", "--tune", "60",
      "--dump-scored", "--out", "artifacts/bpi2013/agg_tuned"]),
    # --- Phase 2: calibration under label shift ---
    ("UCI calibration", ["scripts/run_calibration.py", "--log", "uci_servicenow"]),
    ("BPI 2013 calibration", ["scripts/run_calibration.py", "--log", "bpi2013"]),
    # --- Phase 3a: does event order carry signal? ---
    ("order ablation, both logs", ["scripts/run_order_ablation.py"]),
    # --- Phase 4: conformal coverage ---
    ("UCI conformal", ["scripts/run_conformal.py", "--log", "uci_servicenow"]),
    ("BPI 2013 conformal", ["scripts/run_conformal.py", "--log", "bpi2013"]),
    # --- Phase 5: attribution and error analysis ---
    ("UCI error analysis + SHAP", ["scripts/run_error_analysis.py", "--log", "uci_servicenow"]),
    # --- Phase 6: decision layer ---
    ("UCI alarm sweep", ["scripts/run_alarm.py", "--log", "uci_servicenow"]),
    ("BPI 2013 alarm sweep", ["scripts/run_alarm.py", "--log", "bpi2013"]),
    ("figures and headline table", ["scripts/make_figures.py"]),
    ("web data export", ["scripts/export_web_data.py"]),
]


def main() -> None:
    failures = []
    for i, (label, cmd) in enumerate(ARMS, 1):
        print(f"\n{'=' * 78}\n[{i}/{len(ARMS)}] {label}\n{'=' * 78}")
        t0 = time.time()
        proc = subprocess.run([sys.executable, *cmd], cwd=ROOT)
        if proc.returncode != 0:
            print(f"  FAILED (exit {proc.returncode})")
            failures.append(label)
        else:
            print(f"  done in {time.time() - t0:.1f}s")

    print(f"\n{'=' * 78}")
    if failures:
        print(f"{len(failures)} arm(s) failed:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("all arms complete -- see reports/ for figures and headline_comparison.csv")


if __name__ == "__main__":
    main()
