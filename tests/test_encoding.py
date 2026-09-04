"""Encoding contracts, for both the aggregation and index arms."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from riskradar.adapters import uci_servicenow as ad
from riskradar.encoding import aggregation_encode, index_encode, make_buckets
from riskradar.logspec import feature_columns, split_feature_columns


def test_aggregation_encode_selects_only_whitelisted_columns(uci_prefix):
    X = aggregation_encode(uci_prefix)
    assert set(X.columns) == set(feature_columns(uci_prefix))
    for junk in ("y", "case_id", "_case_end_ts"):
        assert junk not in X.columns


def test_aggregation_encode_uses_object_not_nullable_string(uci_prefix):
    """sklearn's imputer tests missingness with `X != X`, which raises on pd.NA."""
    _, cat_cols = split_feature_columns(uci_prefix)
    X = aggregation_encode(uci_prefix)
    for c in cat_cols:
        assert X[c].dtype == object
        assert not any(v is pd.NA for v in X[c].to_numpy())


def test_aggregation_encode_row_count_is_preserved(uci_prefix):
    assert len(aggregation_encode(uci_prefix)) == len(uci_prefix)


def test_index_encode_widens_with_k(uci_events, uci_prefix):
    """One column block per position, so width grows with prefix length."""
    w2 = index_encode(uci_prefix, uci_events, 2).shape[1]
    w4 = index_encode(uci_prefix, uci_events, 4).shape[1]
    w8 = index_encode(uci_prefix, uci_events, 8).shape[1]
    assert w2 < w4 < w8


def test_index_encode_aligns_with_its_bucket(uci_events, uci_prefix):
    bucket = uci_prefix[uci_prefix["k"] == 4]
    X = index_encode(uci_prefix, uci_events, 4)
    assert len(X) == len(bucket)
    assert list(X.index) == list(bucket["case_id"])


def test_index_encode_positions_are_one_based_and_bounded(uci_events, uci_prefix):
    X = index_encode(uci_prefix, uci_events, 3)
    positions = {int(c.rsplit("_", 1)[1]) for c in X.columns if not c.startswith("static__")}
    assert positions == {1, 2, 3}


def test_index_encode_emits_only_whitelisted_prefixes(uci_events, uci_prefix):
    X = index_encode(uci_prefix, uci_events, 3)
    for c in X.columns:
        assert c.startswith(("idx_cat__", "idx_num__", "static__")), c


def test_index_encode_carries_no_forbidden_column(uci_events, uci_prefix):
    X = index_encode(uci_prefix, uci_events, 3)
    bare = {c.split("__", 1)[-1].rsplit("_", 1)[0] for c in X.columns}
    assert not (bare & ad.FORBIDDEN)


def test_index_encode_elapsed_is_nondecreasing_across_positions(uci_events, uci_prefix):
    X = index_encode(uci_prefix, uci_events, 4)
    cols = [f"idx_num__elapsed_h_{p}" for p in range(1, 5)]
    vals = X[cols].to_numpy(dtype=float)
    diffs = np.diff(vals, axis=1)
    assert np.nanmin(diffs) >= -1e-9


def test_index_encode_empty_bucket_returns_empty(uci_events, uci_prefix):
    assert index_encode(uci_prefix, uci_events, 999).empty


def test_make_buckets_partitions_the_prefix_log(uci_prefix):
    buckets = make_buckets(uci_prefix, 8)
    assert sum(len(b) for b in buckets.values()) == len(uci_prefix)
    for k, b in buckets.items():
        assert (b["k"] == k).all()
