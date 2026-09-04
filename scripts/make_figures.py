"""Render the earliness figures and the headline comparison table.

    python scripts/make_figures.py

Reads `artifacts/` and writes to `reports/`.  Run these first:

    python scripts/run_leaky_baseline.py
    python scripts/run_experiment.py --log uci_servicenow
    python scripts/run_experiment.py --log uci_servicenow --split-mode random_case \
        --out artifacts/uci_servicenow/agg_randomsplit
    python scripts/run_experiment.py --log bpi2013
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ART = ROOT / "artifacts"
REP = ROOT / "reports"

C_FIXED = "#c1121f"
C_VAR = "#457b9d"
C_LEAK = "#8d99ae"
C_BASE = "#e9c46a"
C_RAND = "#2a9d8f"


def _load(rel: str):
    d = ART / rel
    return (
        pd.read_csv(d / "per_k.csv"),
        json.loads((d / "metrics.json").read_text()),
    )


def figure_uci(leaky: dict) -> None:
    per_k, _ = _load("uci_servicenow/agg")
    fixed = per_k[per_k["cohort"] == "fixed"].sort_values("k")
    var = per_k[per_k["cohort"] == "variable"].sort_values("k")

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(8.2, 7.0), sharex=True, gridspec_kw={"height_ratios": [2.6, 1]}
    )
    ax.axhline(leaky["roc_auc"], ls="--", lw=1.6, color=C_LEAK,
               label=f"as-shipped, leaky evaluation ({leaky['roc_auc']:.3f})")
    ax.plot(var["k"], var["auc"], "o-", color=C_VAR, lw=2, ms=6,
            label="variable cohort (all prefixes at k)")
    ax.plot(fixed["k"], fixed["auc"], "o-", color=C_FIXED, lw=2.4, ms=7,
            label="fixed cohort (cases reaching k=8)")
    ax.axhline(0.5, color="#adb5bd", lw=1, ls=":")
    ax.set_ylabel("ROC-AUC on held-out test period")
    ax.set_ylim(0.45, 1.0)
    ax.set_title(
        "How early can an SLA breach be called?\n"
        "UCI ServiceNow incidents, prefix-based, out-of-time split",
        fontsize=11.5, loc="left",
    )
    ax.legend(fontsize=8.5, loc="lower right", framealpha=0.95)
    ax.grid(alpha=0.25)

    # The variable cohort's climb is mostly composition, not skill: short cases
    # exhaust and the survivors are the breach-prone ones.
    ax2.bar(var["k"], var["base_rate_k"], color=C_BASE, edgecolor="white", width=0.6,
            label="variable cohort base rate")
    ax2.plot(fixed["k"], fixed["base_rate_k"], "o-", color=C_FIXED, lw=2, ms=5,
             label="fixed cohort base rate (constant by construction)")
    ax2.set_xlabel("prefix length k (events observed)")
    ax2.set_ylabel("breach rate")
    ax2.set_ylim(0, 1)
    ax2.legend(fontsize=8, loc="upper left", framealpha=0.95)
    ax2.grid(alpha=0.25)
    for a in (ax, ax2):
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(REP / "earliness_curve_uci.png", dpi=160)
    print(f"wrote {REP / 'earliness_curve_uci.png'}")


def figure_crosslog() -> None:
    uci_t, _ = _load("uci_servicenow/agg")
    uci_r, _ = _load("uci_servicenow/agg_randomsplit")
    bpi, bpi_m = _load("bpi2013/agg")

    f = lambda d: d[d["cohort"] == "fixed"].sort_values("k")  # noqa: E731

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.6), sharey=True)

    ax.plot(f(uci_t)["k"], f(uci_t)["auc"], "o-", color=C_FIXED, lw=2.4, ms=7,
            label="out-of-time split (honest)")
    ax.plot(f(uci_r)["k"], f(uci_r)["auc"], "s--", color=C_RAND, lw=2, ms=6,
            label="random case split (optimistic)")
    ax.set_title("UCI ServiceNow: what the split protocol costs", fontsize=10.5, loc="left")
    ax.set_xlabel("prefix length k")
    ax.set_ylabel("ROC-AUC, fixed cohort")
    ax.legend(fontsize=8.5, loc="lower right")

    ax2.plot(f(bpi)["k"], f(bpi)["auc"], "o-", color=C_VAR, lw=2.4, ms=7,
             label="BPI 2013 (random split only)")
    ax2.plot(f(uci_r)["k"], f(uci_r)["auc"], "s--", color=C_RAND, lw=2, ms=6,
             label="UCI, random split (comparable protocol)")
    ax2.set_title(
        f"BPI 2013 Volvo IT: flat from k=1 "
        f"(deadline {bpi_m['label_construction']['deadline_hours']:.0f}h)",
        fontsize=10.5, loc="left",
    )
    ax2.set_xlabel("prefix length k")
    ax2.legend(fontsize=8.5, loc="lower right")

    for a in (ax, ax2):
        a.set_ylim(0.5, 1.0)
        a.grid(alpha=0.25)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)

    fig.suptitle(
        "Cross-log replication. BPI 2013 cannot support an out-of-time split: "
        "all cases arrive inside ten days while running a median of 7.5,\n"
        "so every boundary is straddled. Its curve is therefore compared "
        "against UCI under the same random protocol, not against the honest one.",
        fontsize=8.6, y=1.06, x=0.01, ha="left", color="#495057",
    )
    fig.tight_layout()
    fig.savefig(REP / "crosslog_comparison.png", dpi=160, bbox_inches="tight")
    print(f"wrote {REP / 'crosslog_comparison.png'}")


def figure_encoding() -> None:
    """Aggregation vs index encoding, with the learner held fixed at logreg."""
    pairs = [
        ("UCI ServiceNow (out-of-time split)", "uci_servicenow/agg_logreg",
         "uci_servicenow/index", "uci_servicenow/agg"),
        ("BPI 2013 (random split)", "bpi2013/agg_logreg", "bpi2013/index", "bpi2013/agg"),
    ]
    f = lambda d: d[d["cohort"] == "fixed"].sort_values("k")  # noqa: E731

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.6), sharey=True)
    for ax, (title, agg_l, idx_l, agg_best) in zip(axes, pairs):
        a, _ = _load(agg_l)
        i, _ = _load(idx_l)
        b, mb = _load(agg_best)
        ax.plot(f(a)["k"], f(a)["auc"], "o-", color=C_FIXED, lw=2.2, ms=6,
                label="aggregation + logreg")
        ax.plot(f(i)["k"], f(i)["auc"], "s--", color=C_VAR, lw=2.2, ms=6,
                label="index + logreg")
        ax.plot(f(b)["k"], f(b)["auc"], "^:", color=C_RAND, lw=1.8, ms=6, alpha=0.9,
                label=f"aggregation + {mb['selected_model']} (best arm)")
        ax.set_title(title, fontsize=10.5, loc="left")
        ax.set_xlabel("prefix length k")
        ax.grid(alpha=0.25)
        ax.set_ylim(0.5, 1.0)
        ax.legend(fontsize=8.5, loc="lower right")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("ROC-AUC, fixed cohort")

    fig.suptitle(
        "Encoding comparison, learner held fixed. On UCI the two encodings are within noise of "
        "each other; on BPI index encoding loses badly,\nbecause pure index encoding carries only "
        "static and per-position attributes and discards the running aggregates BPI's signal lives in.\n"
        "Changing the learner moves the curve more than changing the encoding does.",
        fontsize=8.6, y=1.09, x=0.01, ha="left", color="#495057",
    )
    fig.tight_layout()
    fig.savefig(REP / "encoding_comparison.png", dpi=160, bbox_inches="tight")
    print(f"wrote {REP / 'encoding_comparison.png'}")


def table(leaky: dict) -> None:
    uci_t, m_t = _load("uci_servicenow/agg")
    uci_r, m_r = _load("uci_servicenow/agg_randomsplit")
    bpi, m_b = _load("bpi2013/agg")
    f = lambda d: d[d["cohort"] == "fixed"].sort_values("k")  # noqa: E731

    uci_al, _ = _load("uci_servicenow/agg_logreg")
    uci_ix, _ = _load("uci_servicenow/index")
    bpi_al, _ = _load("bpi2013/agg_logreg")
    bpi_ix, _ = _load("bpi2013/index")

    rows = [
        ("UCI", "As shipped: leaky features, random split, selected on test",
         "one row per completed incident", leaky["roc_auc"]),
        ("UCI", "Prefix-based, out-of-time, case decision at k=1",
         "one decision per ticket, first event", m_t["test_case_level_at_k1"]["auc"]),
        ("UCI", "Prefix-based, out-of-time, fixed cohort k=1",
         "long cases only, first event", float(f(uci_t).iloc[0]["auc"])),
        ("UCI", "Prefix-based, out-of-time, fixed cohort k=8",
         "same cases, eight events", float(f(uci_t).iloc[-1]["auc"])),
        ("UCI", "Same, but random case split (ablation)",
         "price of honest evaluation, k=1", float(f(uci_r).iloc[0]["auc"])),
        ("BPI 2013", "Prefix-based, random split, fixed cohort k=1",
         "no out-of-time split possible", float(f(bpi).iloc[0]["auc"])),
        ("BPI 2013", "Prefix-based, random split, fixed cohort k=12",
         "curve is flat: signal is present at intake", float(f(bpi).iloc[-1]["auc"])),
        ("UCI", "Encoding control: aggregation + logreg, k=8",
         "learner held fixed", float(f(uci_al).iloc[-1]["auc"])),
        ("UCI", "Encoding control: index + logreg, k=8",
         "learner held fixed -- within noise of aggregation", float(f(uci_ix).iloc[-1]["auc"])),
        ("BPI 2013", "Encoding control: aggregation + logreg, k=12",
         "learner held fixed", float(f(bpi_al).iloc[-1]["auc"])),
        ("BPI 2013", "Encoding control: index + logreg, k=12",
         "index drops the running aggregates BPI depends on", float(f(bpi_ix).iloc[-1]["auc"])),
    ]
    t = pd.DataFrame(rows, columns=["log", "arm", "what it measures", "ROC-AUC"])
    t["ROC-AUC"] = t["ROC-AUC"].round(4)
    t.to_csv(REP / "headline_comparison.csv", index=False)
    print(f"wrote {REP / 'headline_comparison.csv'}\n")
    print(t.to_string(index=False))


def main() -> None:
    REP.mkdir(exist_ok=True)
    leaky = json.loads((ART / "baseline_leaky" / "metrics.json").read_text())
    figure_uci(leaky)
    figure_crosslog()
    figure_encoding()
    table(leaky)


if __name__ == "__main__":
    main()
