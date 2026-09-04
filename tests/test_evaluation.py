"""Evaluation semantics, especially the fixed/variable cohort distinction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from riskradar.evaluation import (
    case_level_metrics,
    earliness_curve,
    per_prefix_metrics,
    prefix_weighted_summary,
)


def _scored(n_cases=200, k_max=4, seed=0):
    """Prefix rows for cases of differing length, with a separable signal."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_cases):
        n = rng.integers(1, k_max + 1)
        y = int(i % 2)
        for k in range(1, n + 1):
            rows.append(
                {
                    "case_id": f"c{i}",
                    "k": k,
                    "_n_nonterm": n,
                    "y": y,
                    "p": float(np.clip(y * 0.9 + rng.normal(0, 0.01), 0.0, 1.0)),
                }
            )
    return pd.DataFrame(rows)


def test_per_prefix_metrics_reports_cohort_size_and_base_rate():
    out = per_prefix_metrics(pd.Series([0, 1, 0, 1]), np.array([0.1, 0.9, 0.2, 0.8]), pd.Series([1, 1, 2, 2]))
    assert list(out["k"]) == [1, 2]
    assert (out["n_k"] == 2).all()
    assert (out["base_rate_k"] == 0.5).all()


def test_separable_scores_give_auc_one():
    out = per_prefix_metrics(pd.Series([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]), pd.Series([1] * 4))
    assert out.loc[0, "auc"] == 1.0


def test_auc_is_nan_when_one_class_present():
    out = per_prefix_metrics(pd.Series([1, 1]), np.array([0.3, 0.7]), pd.Series([1, 1]))
    assert np.isnan(out.loc[0, "auc"])


def test_fixed_cohort_holds_population_constant():
    """The whole point: n_k must not move with k on the fixed cohort."""
    sc = _scored()
    fixed = earliness_curve(sc, cohort="fixed", k_max=4)
    assert fixed["n_k"].nunique() == 1
    assert fixed["base_rate_k"].nunique() == 1


def test_variable_cohort_shrinks_with_k():
    sc = _scored()
    var = earliness_curve(sc, cohort="variable", k_max=4)
    assert var["n_k"].is_monotonic_decreasing
    assert var["n_k"].nunique() > 1


def test_fixed_cohort_only_keeps_full_length_cases():
    sc = _scored()
    fixed = earliness_curve(sc, cohort="fixed", k_max=4)
    assert fixed["n_k"].iloc[0] == (sc.groupby("case_id")["_n_nonterm"].first() >= 4).sum()


def test_prefix_weighted_summary_weights_by_cohort_size():
    per_k = pd.DataFrame({"k": [1, 2], "n_k": [90, 10], "auc": [0.9, 0.5], "ap": [0.9, 0.5]})
    s = prefix_weighted_summary(per_k)
    assert abs(s["prefix_weighted_auc"] - (0.9 * 90 + 0.5 * 10) / 100) < 1e-9
    assert s["n_prefix_rows"] == 100


def test_case_level_metrics_counts_each_case_once():
    sc = _scored()
    m = case_level_metrics(sc, at_k=1)
    assert m["n_cases"] == sc["case_id"].nunique()
