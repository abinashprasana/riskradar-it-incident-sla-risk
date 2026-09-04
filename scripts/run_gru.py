"""Phase 3b: train the GRU across five seeds and compare it to the tabular baseline.

    python scripts/run_gru.py --log uci_servicenow --seeds 5

The prediction, written down before the run: on UCI the GRU ties the tuned
XGBoost baseline inside its bootstrap interval, because the order ablation
already showed event order carries no signal there. On BPI the gap should be
smaller still or slightly positive at mid prefix lengths.

Five seeds, mean and standard deviation. A single-seed neural number on 23k
cases is noise, and if the seed spread exceeds the model gap then the
comparison is underpowered and the honest report says so.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from riskradar.adapters import get_adapter  # noqa: E402
from riskradar.evaluation import bootstrap_ci, earliness_curve, fast_auc  # noqa: E402
from riskradar.prefix_log import build_prefix_log  # noqa: E402
from riskradar.splitting import (  # noqa: E402
    build_case_index,
    random_case_split,
    slice_prefixes,
    temporal_case_split,
)
from riskradar.torch_seq import PrefixGRU, PrefixTensors, spec_for  # noqa: E402

DEFAULT_PATHS = {
    "uci_servicenow": "incident_event_log.csv",
    "bpi2013": "csv_files/VINST cases incidents.csv",
}
DENSE_WINDOW = {"uci_servicenow": ("2016-02-01", "2016-06-01")}
DEFAULT_SPLIT_MODE = {"uci_servicenow": "temporal", "bpi2013": "random_case"}


def run_seed(tensors, folds, seed, epochs=100, patience=10, batch=256, lr=1e-3, verbose=False):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = PrefixGRU(tensors)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    ytr = folds["train"]["y"].to_numpy(dtype=np.float32)
    pos = float(ytr.sum())
    pos_weight = torch.tensor([(len(ytr) - pos) / pos if pos else 1.0])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    tr_ids = folds["train"]["case_id"].to_numpy()
    tr_ks = folds["train"]["k"].to_numpy()
    n = len(ytr)

    best_auc, best_state, bad = -1.0, None, 0
    for epoch in range(epochs):
        model.train()
        order = np.random.permutation(n)
        for s in range(0, n, batch):
            sl = order[s : s + batch]
            b = tensors.batch(tr_ids[sl], tr_ks[sl])
            opt.zero_grad()
            loss = loss_fn(model(b), torch.from_numpy(ytr[sl]))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        auc = evaluate(model, tensors, folds["val"])
        if auc > best_auc + 1e-5:
            best_auc, bad = auc, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
        if verbose and epoch % 5 == 0:
            print(f"        epoch {epoch:3d} val AUC {auc:.4f} (best {best_auc:.4f})")

    model.load_state_dict(best_state)
    return model, best_auc, epoch + 1


@torch.no_grad()
def predict(model, tensors, fold, batch=1024) -> np.ndarray:
    model.eval()
    ids, ks = fold["case_id"].to_numpy(), fold["k"].to_numpy()
    out = []
    for s in range(0, len(ids), batch):
        b = tensors.batch(ids[s : s + batch], ks[s : s + batch])
        out.append(torch.sigmoid(model(b)).numpy())
    return np.concatenate(out)


def evaluate(model, tensors, fold) -> float:
    return fast_auc(fold["y"].to_numpy(dtype=np.int8), predict(model, tensors, fold))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="uci_servicenow", choices=sorted(DEFAULT_PATHS))
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=60)
    args = ap.parse_args()

    adapter = get_adapter(args.log)
    spec = adapter.SPEC
    k_max = spec.k_max

    print(f"[1/4] loading {args.log}")
    events = adapter.load_events(str(ROOT / DEFAULT_PATHS[args.log]))
    ci = build_case_index(events)
    if args.log in DENSE_WINDOW:
        lo, hi = DENSE_WINDOW[args.log]
        ci = ci[(ci["case_start_ts"] >= lo) & (ci["case_start_ts"] < hi)]

    split_fn = (
        temporal_case_split
        if DEFAULT_SPLIT_MODE[args.log] == "temporal"
        else random_case_split
    )
    split = split_fn(ci)
    labels = adapter.make_labels(events, spec, fit_cases=split.train)
    ci = ci.join(labels["y"], how="inner")

    prefix = build_prefix_log(events, labels, spec, k_max=k_max)
    prefix = prefix[prefix["case_id"].isin(set(ci.index))]
    folds = {
        n: slice_prefixes(prefix, ix)
        for n, ix in (("train", split.train), ("val", split.val), ("test", split.test))
    }

    print("[2/4] building tensors")
    seq_spec = spec_for(events)
    tensors = PrefixTensors(events, seq_spec, k_max)
    print(f"      event channels: {list(tensors.cat_vocabs)} + {len(tensors.num_names)} numeric")
    print(f"      vocab sizes: { {c: v.size for c, v in tensors.cat_vocabs.items()} }")
    print(f"      static: {len(tensors.static_vocabs)} categorical")

    n_params = sum(p.numel() for p in PrefixGRU(tensors).parameters())
    print(f"      parameters: {n_params:,}")

    print(f"[3/4] training {args.seeds} seeds")
    preds, val_aucs, test_aucs = [], [], []
    for s in range(args.seeds):
        t0 = time.time()
        model, val_auc, n_epochs = run_seed(tensors, folds, seed=42 + s, epochs=args.epochs)
        p = predict(model, tensors, folds["test"])
        t_auc = fast_auc(folds["test"]["y"].to_numpy(dtype=np.int8), p)
        preds.append(p)
        val_aucs.append(val_auc)
        test_aucs.append(t_auc)
        print(f"      seed {42 + s}: val {val_auc:.4f}  test {t_auc:.4f}  "
              f"({n_epochs} epochs, {time.time() - t0:.0f}s)")

    print("[4/4] evaluating the seed ensemble")
    test = folds["test"].copy()
    test["p"] = np.mean(preds, axis=0)

    curves = pd.concat(
        [
            earliness_curve(test, cohort="variable", k_max=k_max),
            earliness_curve(test, cohort="fixed", k_max=k_max),
        ],
        ignore_index=True,
    )
    out_dir = ROOT / "artifacts" / args.log / "gru"
    out_dir.mkdir(parents=True, exist_ok=True)
    curves.to_csv(out_dir / "per_k.csv", index=False)
    test[["case_id", "k", "_n_nonterm", "y", "p"]].to_csv(out_dir / "scored_test.csv", index=False)

    # Compare against the tuned tabular arm on the same test rows.
    baseline_dir = ROOT / "artifacts" / args.log / "agg_tuned"
    comparison = {}
    if (baseline_dir / "per_k.csv").exists():
        base = pd.read_csv(baseline_dir / "per_k.csv")
        bf = base[base["cohort"] == "fixed"].set_index("k")
        gf = curves[curves["cohort"] == "fixed"].set_index("k")
        print(f"\n{'k':>3} {'GRU':>8} {'tuned XGB':>10} {'delta':>8}  inside baseline CI?")
        inside = []
        for k in sorted(gf.index):
            g, b = gf.loc[k], bf.loc[k]
            ok = bool(b["auc_lo"] <= g["auc"] <= b["auc_hi"])
            inside.append(ok)
            print(f"{k:>3} {g['auc']:>8.4f} {b['auc']:>10.4f} {g['auc'] - b['auc']:>+8.4f}"
                  f"  {'yes' if ok else 'NO'}")
        comparison = {
            "baseline": "agg_tuned",
            "gru_minus_baseline_mean": float((gf["auc"] - bf["auc"]).mean()),
            "inside_baseline_ci_at_all_k": bool(all(inside)),
            "n_k_inside": int(sum(inside)),
            "n_k_total": len(inside),
        }

    summary = {
        "log": args.log,
        "model": "PrefixGRU",
        "n_parameters": int(n_params),
        "seeds": args.seeds,
        "val_auc_mean": float(np.mean(val_aucs)),
        "val_auc_std": float(np.std(val_aucs)),
        "test_auc_mean": float(np.mean(test_aucs)),
        "test_auc_std": float(np.std(test_aucs)),
        "test_auc_ensemble": fast_auc(
            folds["test"]["y"].to_numpy(dtype=np.int8), test["p"].to_numpy()
        ),
        "comparison": comparison,
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))

    print(f"\nseed spread: test AUC {np.mean(test_aucs):.4f} +/- {np.std(test_aucs):.4f}")
    if comparison:
        gap = abs(comparison["gru_minus_baseline_mean"])
        if np.std(test_aucs) > gap:
            print(f"seed sd ({np.std(test_aucs):.4f}) exceeds the model gap ({gap:.4f}) "
                  f"-- the comparison is underpowered, and that is the honest reading")
    print(f"\nartifacts -> {out_dir}")


if __name__ == "__main__":
    main()
