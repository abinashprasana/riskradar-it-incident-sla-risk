"""Phase 3a: does the ORDER of events carry signal, or only the multiset?

    python scripts/run_order_ablation.py

This is the measurement that decides whether a sequence model is worth building
at all. For each prefix length k it target-encodes three views of the same
trace and compares how well each separates the outcome:

    ordered    the activity sequence as it happened      (New > Active > ...)
    multiset   the same activities, sorted               (Active > New > ...)
    last       only the most recent activity

`ordered - multiset` is exactly what an order-aware architecture can buy that
aggregation encoding cannot. If that gap is zero, a GRU or a transformer is
modelling a property the data does not have.

Everything is **cross-fitted** (5-fold, out-of-fold encoding, smoothed toward
the global mean). That is not a detail. In-sample target encoding of a
high-cardinality key memorises the training labels: on UCI at k=8 there are 283
distinct sequences across 2,979 cases, and the in-sample gap looks like +0.06
purely because of that. Both numbers are reported side by side, because the
contrast between them is the pedagogical point.

Trace length is held fixed within each k (only cases reaching k are used), so
the comparison is not contaminated by the fact that longer cases breach more.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from riskradar.adapters import get_adapter  # noqa: E402
from riskradar.evaluation import bootstrap_ci, fast_auc  # noqa: E402

SMOOTH = 20.0  # pseudo-counts pulling a rare key toward the global mean
N_FOLDS = 5
SEED = 42


def target_encode_cv(keys: np.ndarray, y: np.ndarray, n_folds: int = N_FOLDS) -> np.ndarray:
    """Out-of-fold smoothed target encoding.

    Each row's encoding is computed from the other folds only, so a key seen
    once contributes nothing to its own score. This is the difference between
    measuring signal and measuring memorisation.
    """
    rng = np.random.default_rng(SEED)
    n = len(y)
    fold = rng.permutation(n) % n_folds
    out = np.zeros(n, dtype=float)

    for f in range(n_folds):
        tr, te = fold != f, fold == f
        prior = y[tr].mean()
        df = pd.DataFrame({"k": keys[tr], "y": y[tr]})
        agg = df.groupby("k")["y"].agg(["sum", "count"])
        enc = (agg["sum"] + SMOOTH * prior) / (agg["count"] + SMOOTH)
        out[te] = pd.Series(keys[te]).map(enc).fillna(prior).to_numpy()
    return out


def target_encode_insample(keys: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The naive version, kept only to show what memorisation looks like."""
    prior = y.mean()
    df = pd.DataFrame({"k": keys, "y": y})
    agg = df.groupby("k")["y"].agg(["sum", "count"])
    enc = (agg["sum"] + SMOOTH * prior) / (agg["count"] + SMOOTH)
    return pd.Series(keys).map(enc).fillna(prior).to_numpy()


def views_at_k(events: pd.DataFrame, labels: pd.DataFrame, k: int) -> pd.DataFrame | None:
    """Ordered / multiset / last-activity keys for every case reaching length k."""
    ev = events[events["event_idx"] < k].sort_values(["case_id", "event_idx"])
    n_seen = ev.groupby("case_id").size()
    keep = set(n_seen[n_seen >= k].index)
    if len(keep) < 100:
        return None

    ev = ev[ev["case_id"].isin(keep)]
    g = ev.groupby("case_id")["activity"]
    ordered = g.apply(lambda s: ">".join(s))
    multiset = g.apply(lambda s: ">".join(sorted(s)))
    last = g.last()

    out = pd.DataFrame({"ordered": ordered, "multiset": multiset, "last": last})
    return out.join(labels["y"], how="inner").dropna()


def main() -> None:
    logs = {
        "uci_servicenow": ("incident_event_log.csv", [1, 2, 3, 4, 5, 6, 7, 8]),
        "bpi2013": ("csv_files/VINST cases incidents.csv", [1, 2, 4, 6, 8, 10, 12]),
    }

    rows = []
    for log, (rel, ks) in logs.items():
        adapter = get_adapter(log)
        print(f"\n=== {log} ===")
        events = adapter.load_events(str(ROOT / rel))
        labels = adapter.make_labels(events, adapter.SPEC)

        print(f"{'k':>3} {'n':>6} {'base':>6} | {'ordered':>8} {'multiset':>8} {'last':>7} "
              f"| {'D(order)':>9} {'95% CI':>18} | {'in-sample D':>12}")
        for k in ks:
            v = views_at_k(events, labels, k)
            if v is None:
                continue
            y = v["y"].to_numpy(dtype=np.int8)
            if len(np.unique(y)) < 2:
                continue

            enc = {c: target_encode_cv(v[c].to_numpy(), y) for c in ("ordered", "multiset", "last")}
            auc = {c: fast_auc(y, e) for c, e in enc.items()}
            delta = auc["ordered"] - auc["multiset"]

            # Interval on the DIFFERENCE, resampling cases jointly so the two
            # views always see the same bootstrap sample.
            rng = np.random.default_rng(SEED)
            idx = rng.integers(0, len(y), size=(500, len(y)))
            diffs = [
                d for d in (
                    fast_auc(y[r], enc["ordered"][r]) - fast_auc(y[r], enc["multiset"][r])
                    for r in idx
                ) if d == d
            ]
            lo, hi = (np.percentile(diffs, [2.5, 97.5]) if diffs else (np.nan, np.nan))

            ins = {c: fast_auc(y, target_encode_insample(v[c].to_numpy(), y))
                   for c in ("ordered", "multiset")}
            ins_delta = ins["ordered"] - ins["multiset"]

            print(f"{k:>3} {len(y):>6} {y.mean():>6.3f} | {auc['ordered']:>8.4f} "
                  f"{auc['multiset']:>8.4f} {auc['last']:>7.4f} | {delta:>+9.4f} "
                  f"[{lo:>+6.3f},{hi:>+6.3f}] | {ins_delta:>+12.4f}")

            rows.append({
                "log": log, "k": k, "n": int(len(y)), "base_rate": float(y.mean()),
                "auc_ordered": auc["ordered"], "auc_multiset": auc["multiset"],
                "auc_last": auc["last"], "delta_order": float(delta),
                "delta_lo": float(lo), "delta_hi": float(hi),
                "n_distinct_ordered": int(v["ordered"].nunique()),
                "n_distinct_multiset": int(v["multiset"].nunique()),
                "insample_delta_order": float(ins_delta),
            })

    df = pd.DataFrame(rows)
    out_dir = ROOT / "artifacts" / "order_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "order_ablation.csv", index=False)

    verdict = {}
    for log, g in df.groupby("log"):
        sig = g[(g["delta_lo"] > 0) | (g["delta_hi"] < 0)]
        verdict[log] = {
            "mean_delta": float(g["delta_order"].mean()),
            "max_abs_delta": float(g["delta_order"].abs().max()),
            "mean_insample_delta": float(g["insample_delta_order"].mean()),
            "k_with_ci_excluding_zero": [int(x) for x in sig["k"]],
            "order_carries_signal": bool(len(sig) > 0 and sig["delta_order"].mean() > 0),
        }
    (out_dir / "verdict.json").write_text(json.dumps(verdict, indent=2))

    print("\n=== verdict ===")
    for log, v in verdict.items():
        print(f"{log}: mean cross-fitted D = {v['mean_delta']:+.4f}, "
              f"in-sample D = {v['mean_insample_delta']:+.4f}, "
              f"k where CI excludes 0: {v['k_with_ci_excluding_zero'] or 'none'}")
    print(f"\nartifacts -> {out_dir}")


if __name__ == "__main__":
    main()
