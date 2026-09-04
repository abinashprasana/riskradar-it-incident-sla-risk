"""Phase 6: cost-sensitive alarm thresholds.

    python scripts/run_alarm.py --log uci_servicenow

Tunes the firing threshold on the validation fold and applies it unchanged to
test, across a grid of cost assumptions. Writes `alarm_sweep.csv` and
`alarm_summary.json`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from riskradar.alarm import CostModel, baselines, evaluate_alarm, optimise_threshold, sweep  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="uci_servicenow")
    ap.add_argument("--arm", default="agg")
    args = ap.parse_args()

    run = ROOT / "artifacts" / args.log / args.arm
    val = pd.read_csv(run / "scored_val.csv")
    test = pd.read_csv(run / "scored_test.csv")

    print(f"alarm layer for {args.log}/{args.arm}")
    print(f"  tuning on val ({val['case_id'].nunique():,} cases), "
          f"applying to test ({test['case_id'].nunique():,} cases)")

    ref = CostModel(c_intervene=1.0, c_breach=10.0, effectiveness=0.5)
    tau, curve = optimise_threshold(val, ref)
    at_test = evaluate_alarm(test, tau, ref)
    base = baselines(test, ref)

    print(f"\n  reference point: escalation costs 1, a missed breach costs 10,")
    print(f"  and escalating averts half the breaches it catches.")
    print(f"    tau* (tuned on val)        {tau:.2f}")
    print(f"    closed-form break-even p*  {ref.break_even_threshold():.2f}")
    print(f"    test mean cost per case    {at_test['mean_cost']:.4f}")
    print(f"    never alarm                {base['never_alarm']:.4f}")
    print(f"    always alarm at k=1        {base['always_alarm_at_k1']:.4f}")
    print(f"    alarm rate                 {at_test['alarm_rate']:.3f}")
    print(f"    mean alarm prefix          k={at_test['mean_alarm_k']:.2f}")
    print(f"    recall / precision         {at_test['recall']:.3f} / {at_test['precision']:.3f}")

    grid = sweep(val, test)
    grid.to_csv(run / "alarm_sweep.csv", index=False)

    print(f"\n  sensitivity across cost assumptions:")
    print(f"  {'c_i/c_b':>8} {'effect':>7} {'tau*':>6} {'saving':>8} {'alarm%':>7} {'mean k':>7}")
    for _, r in grid.iterrows():
        print(f"  {r['cost_ratio']:>8.2f} {r['effectiveness']:>7.1f} {r['tau_star']:>6.2f} "
              f"{r['saving_vs_trivial']:>+8.1%} {r['alarm_rate']:>7.1%} {r['mean_alarm_k']:>7.2f}")

    beats = grid[grid["saving_vs_trivial"] > 0]
    summary = {
        "reference": {"tau_star": tau, "cost_model": ref.__dict__, **at_test, **base},
        "sweep_cells": int(len(grid)),
        "cells_beating_trivial": int(len(beats)),
        "tau_range": [float(grid["tau_star"].min()), float(grid["tau_star"].max())],
        "median_saving": float(grid["saving_vs_trivial"].median()),
    }
    (run / "alarm_summary.json").write_text(json.dumps(summary, indent=2, default=float))

    print(f"\n  the model beats the better trivial policy in "
          f"{len(beats)}/{len(grid)} cost cells")
    print(f"  tau* ranges {grid['tau_star'].min():.2f} to {grid['tau_star'].max():.2f} "
          f"across those assumptions")
    print(f"\nartifacts -> {run}")


if __name__ == "__main__":
    main()
