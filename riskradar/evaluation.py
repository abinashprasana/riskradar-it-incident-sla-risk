"""Prefix-conditioned evaluation: the earliness curve.

The headline question is not "how accurate is the model" but "how early in an
incident's life can a reliable decision be made".  That is answered by scoring
each prefix length k separately.

One trap governs the whole module.  In the UCI log the base rate climbs from
0.382 at k=1 to 0.818 at k=8, purely because short cases exhaust and the
survivors are the long, breach-prone ones.  A curve computed over "all prefixes
available at k" therefore improves with k even for a model that learned
nothing.  So every curve is reported two ways:

``variable``  all prefixes available at each k.  Operationally realistic --
              this is the population a live queue would present -- but the
              cohort changes at every k, so movement along it conflates skill
              with composition.
``fixed``     only cases long enough to reach k_max.  The cohort is identical
              at every k, so a change in AUC is a change in what the model
              knows.  This is the curve that answers the earliness question.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _safe_auc(y, p) -> float:
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def fast_auc(y: np.ndarray, p: np.ndarray) -> float:
    """AUC via the rank-sum identity, for use inside a bootstrap loop.

    Equivalent to `roc_auc_score` (ties averaged) but skips input validation and
    ROC-curve construction, which `roc_auc_score` redoes on every call. Across a
    thousand resamples per cell that difference is the whole runtime of the
    experiment suite, not a micro-optimisation.
    """
    n_pos = int(y.sum())
    n_neg = y.shape[0] - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    # rankdata averages tied ranks in C. Bootstrap resamples duplicate rows and
    # therefore create ties by construction, so tie handling is not optional
    # here -- getting it wrong biases every interval.
    ranks = rankdata(p)
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def bootstrap_ci(
    y,
    p,
    metric=None,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Percentile bootstrap interval for a metric, resampling rows with replacement.

    Necessary because the fixed cohort is 478 test cases. A difference of 0.02
    AUC between two learners on that sample is not obviously distinguishable
    from noise, and without an interval there is no way to say whether a model
    comparison means anything. Resamples that end up single-class produce no
    defined AUC and are skipped rather than counted as a degenerate score.
    """
    y = np.asarray(y, dtype=np.int8)
    p = np.asarray(p, dtype=np.float64)
    n = len(y)
    if n == 0 or len(np.unique(y)) < 2:
        return float("nan"), float("nan")

    score = metric or fast_auc
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))

    stats = []
    for row in idx:
        s = score(y[row], p[row])
        if s == s:  # skip NaN from a single-class resample
            stats.append(s)

    if not stats:
        return float("nan"), float("nan")
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def expected_calibration_error(y, p, bins: int = 10) -> dict:
    """ECE, MCE and signed bias against equal-width probability bins.

    Signed bias is reported alongside the absolute measures because the
    direction is the diagnostic here: a uniformly positive bias across every
    bin points at a shifted prior rather than at an overconfident learner, and
    those two problems have different fixes.
    """
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    if len(y) == 0:
        return {"ece": float("nan"), "mce": float("nan"), "bias": float("nan"), "n": 0}

    edges = np.linspace(0.0, 1.0, bins + 1)
    which = np.clip(np.digitize(p, edges[1:-1], right=False), 0, bins - 1)

    ece = mce = bias = 0.0
    for b in range(bins):
        m = which == b
        nb = int(m.sum())
        if nb == 0:
            continue
        gap = p[m].mean() - y[m].mean()
        w = nb / len(y)
        ece += w * abs(gap)
        bias += w * gap
        mce = max(mce, abs(gap))

    return {"ece": float(ece), "mce": float(mce), "bias": float(bias), "n": int(len(y))}


def per_prefix_metrics(
    y: pd.Series,
    p: np.ndarray,
    k: pd.Series,
    threshold: float = 0.5,
    with_ci: bool = True,
    n_boot: int = 1000,
) -> pd.DataFrame:
    """Metrics at each prefix length, with the cohort size and base rate.

    `n_k` and `base_rate_k` are not decoration: without them a reader cannot
    tell whether a rising AUC reflects a better-informed model or a different
    population, and in this data it is usually the latter.

    `auc_lo` / `auc_hi` are a 95% percentile bootstrap. At k=8 the fixed cohort
    is 478 cases, so the interval is wide enough to change how several model
    comparisons should be read.
    """
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p), "k": np.asarray(k)})
    rows = []
    for kk, g in df.groupby("k", sort=True):
        pred = (g["p"] >= threshold).astype(int)
        cal = expected_calibration_error(g["y"], g["p"])
        lo, hi = bootstrap_ci(g["y"], g["p"], n_boot=n_boot) if with_ci else (np.nan, np.nan)
        rows.append(
            {
                "k": int(kk),
                "n_k": int(len(g)),
                "base_rate_k": float(g["y"].mean()),
                "auc": _safe_auc(g["y"], g["p"]),
                "auc_lo": lo,
                "auc_hi": hi,
                "ap": float(average_precision_score(g["y"], g["p"]))
                if g["y"].nunique() == 2
                else float("nan"),
                "brier": float(brier_score_loss(g["y"], g["p"])),
                "ece": cal["ece"],
                "bias": cal["bias"],
                "precision": float(precision_score(g["y"], pred, zero_division=0)),
                "recall": float(recall_score(g["y"], pred, zero_division=0)),
                "f1": float(f1_score(g["y"], pred, zero_division=0)),
            }
        )
    return pd.DataFrame(rows)


def earliness_curve(
    scored: pd.DataFrame,
    cohort: str = "fixed",
    k_max: int | None = None,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Per-k metrics on either the variable or the fixed cohort.

    `scored` needs columns `case_id`, `k`, `y`, `p`, `_n_nonterm`.
    """
    df = scored
    k_max = int(k_max if k_max is not None else df["k"].max())
    if cohort == "fixed":
        df = df[df["_n_nonterm"] >= k_max]
    elif cohort != "variable":
        raise ValueError(f"unknown cohort {cohort!r}")

    out = per_prefix_metrics(df["y"], df["p"].to_numpy(), df["k"], threshold)
    out.insert(0, "cohort", cohort)
    return out


def prefix_weighted_summary(per_k: pd.DataFrame) -> dict:
    """Single numbers for the abstract, weighted by how many prefixes each k has."""
    ok = per_k.dropna(subset=["auc"])
    w = ok["n_k"].to_numpy(dtype=float)
    if not len(ok) or w.sum() == 0:
        return {"prefix_weighted_auc": float("nan"), "prefix_weighted_ap": float("nan")}
    return {
        "prefix_weighted_auc": float(np.average(ok["auc"], weights=w)),
        "prefix_weighted_ap": float(np.average(ok["ap"], weights=w)),
        "n_prefix_rows": int(ok["n_k"].sum()),
    }


def case_level_metrics(scored: pd.DataFrame, at_k: int = 1) -> dict:
    """Metrics for one decision per case, taken at prefix length `at_k`.

    The operationally honest summary: a triage tool scores each ticket once, so
    weighting a 48-event incident 8x a 2-event one is a reporting artifact.
    """
    g = scored[scored["k"] == at_k]
    return {
        "at_k": at_k,
        "n_cases": int(len(g)),
        "base_rate": float(g["y"].mean()),
        "auc": _safe_auc(g["y"], g["p"]),
        "ap": float(average_precision_score(g["y"], g["p"]))
        if g["y"].nunique() == 2
        else float("nan"),
    }


def reliability_by_k(scored: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    """Predicted vs observed breach rate, per prefix length."""
    df = scored.copy()
    df["bin"] = pd.cut(df["p"], bins=np.linspace(0, 1, bins + 1), include_lowest=True)
    return (
        df.groupby(["k", "bin"], observed=True)
        .agg(n=("y", "size"), mean_pred=("p", "mean"), observed=("y", "mean"))
        .reset_index()
        .dropna()
    )
