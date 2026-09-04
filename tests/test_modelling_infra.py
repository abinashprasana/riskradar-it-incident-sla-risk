"""Phase 0 guarantees: four disjoint folds, a stable test set, and error bars.

The claim this project rests on is that no model saw test data during
selection. That is only worth anything if it is asserted.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from riskradar.adapters import uci_servicenow as ad
from riskradar.evaluation import (
    bootstrap_ci,
    expected_calibration_error,
    per_prefix_metrics,
)
from riskradar.splitting import (
    build_case_index,
    random_case_split,
    slice_prefixes,
    temporal_case_split,
)

# Measured before the fourth fold existed. The 4-way default sums to the same
# 0.75 modelling fraction, so these must not move.
EXPECTED_TEST_CASES = 5707
EXPECTED_TEST_PREFIXES = 17118
EXPECTED_STRADDLERS = 2718
DENSE = ("2016-02-01", "2016-06-01")


@pytest.fixture(scope="module")
def dense_index(uci_events):
    ci = build_case_index(uci_events)
    return ci[(ci["case_start_ts"] >= DENSE[0]) & (ci["case_start_ts"] < DENSE[1])]


@pytest.fixture(scope="module")
def split(dense_index):
    return temporal_case_split(dense_index)


def test_four_folds_are_pairwise_disjoint_at_case_level(split):
    sets = [set(split.train), set(split.val), set(split.calib), set(split.test)]
    assert all(not (a & b) for a, b in itertools.combinations(sets, 2))


def test_four_folds_are_pairwise_disjoint_at_prefix_level(uci_prefix, split):
    ids = [
        set(slice_prefixes(uci_prefix, ix)["case_id"])
        for ix in (split.train, split.val, split.calib, split.test)
    ]
    assert all(not (a & b) for a, b in itertools.combinations(ids, 2))


def test_adding_val_does_not_move_the_test_fold(split, uci_prefix, dense_index):
    """The whole point of 0.50/0.15/0.10: the test boundary is unchanged.

    If this fails, the fourth fold silently invalidated every previously
    published test number.
    """
    assert len(split.test) == EXPECTED_TEST_CASES
    px = uci_prefix[uci_prefix["case_id"].isin(set(dense_index.index))]
    assert len(slice_prefixes(px, split.test)) == EXPECTED_TEST_PREFIXES
    assert split.n_dropped_straddling["modelling_period"] == EXPECTED_STRADDLERS


def test_modelling_folds_close_before_the_test_period(dense_index, split):
    modelling = split.train.union(split.val).union(split.calib)
    assert (dense_index.loc[modelling, "case_end_ts"] < split.test_boundary).all()


def test_val_and_calib_are_both_non_empty(split):
    assert len(split.val) > 0
    assert len(split.calib) > 0


def test_fraction_validation_rejects_impossible_splits(dense_index):
    with pytest.raises(ValueError):
        temporal_case_split(dense_index, train_frac=0.7, val_frac=0.2, calib_frac=0.2)


def test_random_case_split_also_yields_four_disjoint_folds(dense_index):
    s = random_case_split(dense_index)
    sets = [set(s.train), set(s.val), set(s.calib), set(s.test)]
    assert all(not (a & b) for a, b in itertools.combinations(sets, 2))
    assert sum(len(x) for x in sets) == len(dense_index)


def test_split_is_reproducible(dense_index):
    a = temporal_case_split(dense_index, seed=7)
    b = temporal_case_split(dense_index, seed=7)
    assert set(a.val) == set(b.val)
    assert set(a.calib) == set(b.calib)


def test_as_dict_reports_every_fold(split):
    d = split.as_dict()
    for key in ("n_train_cases", "n_val_cases", "n_calib_cases", "n_test_cases"):
        assert key in d


# --- error bars ---------------------------------------------------------


def test_bootstrap_interval_brackets_the_point_estimate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    p = np.clip(y * 0.6 + rng.normal(0.2, 0.2, 400), 0, 1)
    from riskradar.evaluation import _safe_auc

    point = _safe_auc(y, p)
    lo, hi = bootstrap_ci(y, p, n_boot=300)
    assert lo <= point <= hi


def test_bootstrap_interval_is_wider_on_less_data():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 2000)
    p = np.clip(y * 0.5 + rng.normal(0.25, 0.25, 2000), 0, 1)
    wide = np.subtract(*reversed(bootstrap_ci(y[:150], p[:150], n_boot=300)))
    narrow = np.subtract(*reversed(bootstrap_ci(y, p, n_boot=300)))
    assert wide > narrow


def test_bootstrap_returns_nan_on_single_class():
    lo, hi = bootstrap_ci(np.ones(50), np.linspace(0, 1, 50), n_boot=50)
    assert np.isnan(lo) and np.isnan(hi)


def test_per_prefix_metrics_carries_ci_and_calibration_columns():
    y = pd.Series([0, 1] * 100)
    p = np.tile([0.2, 0.8], 100)
    out = per_prefix_metrics(y, p, pd.Series([1] * 200), n_boot=100)
    for col in ("auc_lo", "auc_hi", "ece", "bias"):
        assert col in out.columns
    assert out.loc[0, "auc_lo"] <= out.loc[0, "auc"] <= out.loc[0, "auc_hi"]


# --- calibration measurement -------------------------------------------


def test_perfect_calibration_scores_zero_ece():
    # 300 cases at p=0.5 whose outcomes really are half-and-half.
    y = np.array([0, 1] * 150)
    p = np.full(300, 0.5)
    assert expected_calibration_error(y, p)["ece"] < 1e-9


def test_ece_detects_a_uniform_overprediction():
    y = np.zeros(200)
    y[:40] = 1  # true rate 0.20
    p = np.full(200, 0.60)
    out = expected_calibration_error(y, p)
    assert out["ece"] == pytest.approx(0.40, abs=1e-6)
    # Positive bias means the model predicts higher than reality.
    assert out["bias"] == pytest.approx(0.40, abs=1e-6)


def test_bias_is_signed_where_ece_is_not():
    """Two models, equally miscalibrated, opposite directions.

    ECE cannot tell them apart; bias is what says one over-predicts and the
    other under-predicts, which is the distinction that picks the fix.
    """
    y = np.concatenate([np.ones(50), np.zeros(50)])
    over = expected_calibration_error(y, np.full(100, 0.9))
    under = expected_calibration_error(y, np.full(100, 0.1))
    assert over["ece"] == pytest.approx(under["ece"], abs=1e-9)
    assert over["bias"] > 0 > under["bias"]


# --- class-imbalance control arm ---------------------------------------


def test_xgboost_actually_receives_the_weighting():
    """Regression guard.

    `class_weight` is a scikit-learn parameter; XGBoost ignores it and
    re-weights through `scale_pos_weight`. The first version of this arm set
    only `class_weight`, so toggling it changed nothing for the learner that
    actually gets selected and the comparison reported a meaningless zero.
    """
    from riskradar.models import make_candidates

    on = dict(make_candidates(balanced=True, pos_weight=3.0))
    off = dict(make_candidates(balanced=False, pos_weight=3.0))
    if "xgboost" not in on:
        pytest.skip("xgboost not installed")
    assert on["xgboost"].get_params()["scale_pos_weight"] == 3.0
    assert off["xgboost"].get_params()["scale_pos_weight"] == 1.0


def test_unbalanced_candidates_carry_no_class_weight():
    from riskradar.models import make_candidates

    off = dict(make_candidates(balanced=False))
    assert off["logreg"].get_params()["class_weight"] is None
    assert off["random_forest"].get_params()["class_weight"] is None
