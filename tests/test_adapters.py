"""Adapter contract: every log must arrive as the same canonical frame."""

from __future__ import annotations

import pandas as pd
import pytest

from riskradar.adapters import get_adapter
from riskradar.adapters import uci_servicenow as ad
from riskradar.logspec import CANONICAL_BOOKKEEPING

EXPECTED_CASES_FULL = 24918
EXPECTED_CASES_KEPT = 23102
EXPECTED_SENTINEL_DROPPED = 5


def test_registry_resolves_known_logs():
    assert get_adapter("uci_servicenow") is ad
    with pytest.raises(ValueError):
        get_adapter("nope")


def test_canonical_columns_present(uci_events):
    for c in (*CANONICAL_BOOKKEEPING, "activity", "resource"):
        assert c in uci_events.columns


def test_question_mark_is_parsed_as_missing(uci_events):
    """`?` is the missing token; left raw it becomes a real one-hot level."""
    assert not (uci_events["static__category"].dropna() == "?").any()
    assert uci_events["static__cmdb_ci_known"].mean() < 0.05


def test_terminal_events_removed(uci_events):
    assert not uci_events["activity"].isin(ad.SPEC.terminal_activities).any()


def test_sentinel_events_removed_and_counted(uci_events):
    assert uci_events.attrs["n_sentinel_events_dropped"] == EXPECTED_SENTINEL_DROPPED
    assert "-100" not in set(uci_events["activity"])


def test_case_counts_and_selection_bias_recorded(uci_events):
    assert uci_events.attrs["n_cases_full"] == EXPECTED_CASES_FULL
    assert uci_events["case_id"].nunique() == EXPECTED_CASES_KEPT
    assert uci_events.attrs["n_cases_dropped_all_terminal"] == (
        EXPECTED_CASES_FULL - EXPECTED_CASES_KEPT
    )
    # Dropped cases are NOT a random subset -- that is why the number is kept.
    assert uci_events.attrs["breach_rate_dropped"] < uci_events.attrs["breach_rate_kept"]


def test_event_idx_is_dense_and_zero_based(uci_events):
    g = uci_events.groupby("case_id")["event_idx"]
    assert (g.min() == 0).all()
    assert (g.max() + 1 == g.size()).all()


def test_events_are_chronological_within_case(uci_events):
    d = uci_events.groupby("case_id")["event_ts"].diff().dropna()
    assert (d >= pd.Timedelta(0)).all()


def test_case_start_is_constant_within_case(uci_events):
    assert (uci_events.groupby("case_id")["_case_start_ts"].nunique() == 1).all()


def test_labels_are_binary_and_cover_every_case(uci_events, uci_labels):
    assert set(uci_labels["y"].unique()) <= {0, 1}
    assert uci_labels.index.nunique() == uci_events["case_id"].nunique()
    assert abs(uci_labels["y"].mean() - 0.382) < 0.01


def test_forbidden_columns_absent_from_canonical_frame(uci_events):
    assert not (set(uci_events.columns) & ad.FORBIDDEN)
