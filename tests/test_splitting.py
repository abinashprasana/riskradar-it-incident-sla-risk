"""Split integrity.

The failure this guards against is silent: if a single case has prefixes on both
sides of the split, the model sees its own future and every downstream number is
inflated with no visible symptom.
"""

from __future__ import annotations

import pandas as pd
import pytest

from riskradar.splitting import build_case_index, slice_prefixes, temporal_case_split


@pytest.fixture(scope="module")
def case_index(uci_events, uci_labels):
    return build_case_index(uci_events, uci_labels)


@pytest.fixture(scope="module")
def split(case_index):
    return temporal_case_split(case_index)


def test_folds_are_disjoint_at_case_level(split):
    a, b, c = set(split.train), set(split.calib), set(split.test)
    assert not (a & b) and not (a & c) and not (b & c)


def test_folds_are_disjoint_at_prefix_level(uci_prefix, split):
    ids = {
        name: set(slice_prefixes(uci_prefix, ix)["case_id"])
        for name, ix in (("train", split.train), ("calib", split.calib), ("test", split.test))
    }
    assert not (ids["train"] & ids["test"])
    assert not (ids["calib"] & ids["test"])
    assert not (ids["train"] & ids["calib"])


def test_every_training_case_closes_before_the_test_period(case_index, split):
    """The straddle rule: nothing may still be running when test begins."""
    modelling = split.train.union(split.calib)
    assert (case_index.loc[modelling, "case_end_ts"] < split.test_boundary).all()


def test_test_cases_all_start_after_the_boundary(case_index, split):
    assert (case_index.loc[split.test, "case_start_ts"] >= split.test_boundary).all()


def test_straddling_cases_are_counted(split):
    assert split.n_dropped_straddling["modelling_period"] > 0


def test_disabling_straddle_drop_keeps_more_cases(case_index):
    kept = temporal_case_split(case_index, drop_straddling=False)
    dropped = temporal_case_split(case_index, drop_straddling=True)
    n = lambda s: len(s.train) + len(s.calib)  # noqa: E731
    assert n(kept) > n(dropped)


def test_random_calib_matches_train_base_rate(case_index):
    """Random calib exists so selection is not made on a duration-skewed fold."""
    s = temporal_case_split(case_index, calib_mode="random")
    tr = case_index.loc[s.train, "y"].mean()
    ca = case_index.loc[s.calib, "y"].mean()
    assert abs(tr - ca) < 0.05

    tail = temporal_case_split(case_index, calib_mode="temporal_tail")
    gap_tail = abs(
        case_index.loc[tail.train, "y"].mean() - case_index.loc[tail.calib, "y"].mean()
    )
    assert gap_tail > abs(tr - ca)


def test_split_is_reproducible(case_index):
    a = temporal_case_split(case_index, seed=7)
    b = temporal_case_split(case_index, seed=7)
    assert set(a.calib) == set(b.calib)


def test_rejects_impossible_fractions(case_index):
    with pytest.raises(ValueError):
        temporal_case_split(case_index, train_frac=0.9, calib_frac=0.2)
