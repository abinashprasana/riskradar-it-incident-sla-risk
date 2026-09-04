"""The leakage tests.

These exist because the UCI log back-fills its closure fields onto every event
of a trace, so a column that looks innocuous under inspection can carry the
outcome at k=1.  Inspection missed it; an assertion will not.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from riskradar.adapters import uci_servicenow as ad
from riskradar.logspec import feature_columns
from riskradar.prefix_log import assert_no_leakage, build_prefix_log, leakage_screen

# Verified directly against the CSV; see docs/MODEL_CARD.md.
EXPECTED_N_K = {1: 23102, 2: 16643, 4: 9434, 6: 5423, 8: 3013}
EXPECTED_CASES = 23102


def test_no_forbidden_column_reaches_features(uci_prefix):
    assert_no_leakage(uci_prefix, ad.FORBIDDEN)


def test_feature_whitelist_is_closed(uci_prefix):
    """Every feature carries a known prefix -- nothing arrives by accident."""
    for c in feature_columns(uci_prefix):
        assert c.startswith(("static__", "dyn_cat__", "dyn_num__", "num__")), c


def test_label_and_bookkeeping_are_not_features(uci_prefix):
    feats = set(feature_columns(uci_prefix))
    for c in ("y", "case_id", "k", "event_ts", "_case_start_ts", "_case_end_ts", "_n_nonterm"):
        assert c not in feats


def test_prefix_counts_match_verified_values(uci_prefix):
    n_nonterm = uci_prefix.groupby("case_id")["_n_nonterm"].first()
    assert uci_prefix["case_id"].nunique() == EXPECTED_CASES
    for k, expected in EXPECTED_N_K.items():
        assert int((n_nonterm >= k).sum()) == expected, f"k={k}"


def test_prefix_row_count_is_sum_of_capped_lengths(uci_prefix):
    n_nonterm = uci_prefix.groupby("case_id")["_n_nonterm"].first()
    assert len(uci_prefix) == int(n_nonterm.clip(upper=8).sum())


def test_k_is_one_based_and_capped(uci_prefix):
    assert uci_prefix["k"].min() == 1
    assert uci_prefix["k"].max() <= 8


def test_running_counters_are_monotonic_within_case(uci_prefix):
    """A running count may never decrease as more of the trace is observed."""
    sub = uci_prefix.sort_values(["case_id", "k"]).head(50_000)
    for col in ("num__n_distinct_activities", "num__n_distinct_resources", "num__elapsed_h"):
        d = sub.groupby("case_id")[col].diff().dropna()
        assert (d >= -1e-9).all(), col


def test_elapsed_is_never_negative(uci_prefix):
    assert (uci_prefix["num__elapsed_h"].dropna() >= 0).all()


def test_leakage_screen_rejects_backfilled_timestamps():
    """closed_at/resolved_at describe a time after the observation point."""
    raw = pd.DataFrame(
        {
            "case": ["a", "a", "b", "b"],
            "seq": [0, 1, 0, 1],
            "ts": pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-03"]
            ),
            # back-filled: known at event 0 but refers to the future
            "closed_at": pd.to_datetime(
                ["2024-01-09", "2024-01-09", "2024-01-09", "2024-01-09"]
            ),
            # legitimately fixed at creation
            "category": ["x", "x", "y", "y"],
        }
    )
    y = pd.Series({"a": 1, "b": 0})
    rep = leakage_screen(raw, "case", ("seq",), y, "ts").set_index("column")

    assert rep.loc["closed_at", "verdict"] == "REJECT_FUTURE"
    assert rep.loc["closed_at", "future_ref_rate"] == 1.0
    # A constant attribute must NOT be rejected merely for being constant.
    assert rep.loc["category", "verdict"] != "REJECT_FUTURE"
    assert rep.loc["category", "constant_rate"] == 1.0


def test_assert_no_leakage_catches_prefixed_column():
    df = pd.DataFrame({"static__closed_code": ["a"], "num__k": [1]})
    with pytest.raises(AssertionError, match="closed_code"):
        assert_no_leakage(df, frozenset({"closed_code"}))


def test_build_prefix_log_respects_k_max(uci_events, uci_labels):
    small = build_prefix_log(uci_events, uci_labels, ad.SPEC, k_max=3)
    assert small["k"].max() == 3
