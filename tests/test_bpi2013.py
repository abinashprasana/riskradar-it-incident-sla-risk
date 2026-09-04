"""BPI 2013 adapter, including the two documented deviations."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from riskradar.adapters import bpi2013 as ad
from riskradar.logspec import CANONICAL_BOOKKEEPING, feature_columns
from riskradar.prefix_log import assert_no_leakage, build_prefix_log
from riskradar.splitting import build_case_index

BPI_CSV = Path(__file__).resolve().parents[1] / "csv_files" / "VINST cases incidents.csv"

EXPECTED_CASES_FULL = 7546  # 7,554 shipped, less the 8 still-open cases
EXPECTED_CENSORED = 8


@pytest.fixture(scope="module")
def events():
    if not BPI_CSV.exists():
        pytest.skip(f"missing {BPI_CSV}")
    return ad.load_events(str(BPI_CSV))


@pytest.fixture(scope="module")
def labels(events):
    return ad.make_labels(events)


def test_parses_semicolon_latin1_export(events):
    assert len(events) > 50_000
    for c in (*CANONICAL_BOOKKEEPING, "activity", "resource"):
        assert c in events.columns


def test_censored_cases_dropped_and_counted(events):
    """8 cases end in `Accepted` -- still open, so duration is undefined."""
    assert events.attrs["n_cases_censored_dropped"] == EXPECTED_CENSORED
    assert events.attrs["n_cases_full"] == EXPECTED_CASES_FULL


def test_terminal_events_removed(events):
    assert not events["activity"].isin(ad.SPEC.terminal_activities).any()


def test_impact_is_excluded(events):
    """`SR Latest Impact` is constant and unverifiable; it stays out."""
    assert not any("impact" in c.lower() for c in feature_columns(events))


def test_deadline_is_a_constant_not_a_fitted_quantile(events):
    """A fitted percentile collapses to 0.6h once straddlers are removed."""
    everything = ad.make_labels(events, fit_cases=None)
    subset = ad.make_labels(events, fit_cases=events["case_id"].unique()[:500])
    assert everything.attrs["deadline_hours"] == subset.attrs["deadline_hours"] == 240.0
    assert everything["y"].equals(subset["y"])


def test_label_matches_deadline(events, labels):
    dur = labels["_case_duration_h"]
    assert (labels["y"] == (dur > 240.0).astype(int)).all()
    assert 0.25 < labels["y"].mean() < 0.40  # near the UCI base rate of 0.366


def test_still_at_risk_flags_determined_prefixes(events, labels):
    px = build_prefix_log(events, labels, ad.SPEC)
    at_risk = ad.mark_still_at_risk(px, 240.0)
    # Anything past the deadline has already breached by construction.
    assert (px.loc[~at_risk, "y"] == 1).all()


def test_prefix_log_has_no_forbidden_columns(events, labels):
    px = build_prefix_log(events, labels, ad.SPEC)
    assert_no_leakage(px, ad.FORBIDDEN)


def test_arrival_window_is_too_short_for_temporal_split(events):
    """The measurement behind the random_case deviation."""
    ci = build_case_index(events)
    span = ci["case_start_ts"].quantile(0.75) - ci["case_start_ts"].quantile(0.25)
    median_duration = (ci["case_end_ts"] - ci["case_start_ts"]).median()
    assert span < pd.Timedelta(days=10)
    assert median_duration > span / 2
