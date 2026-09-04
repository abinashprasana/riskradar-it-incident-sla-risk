"""Adapter for BPI Challenge 2013, incidents (Volvo IT, VINST).

Steeman, W. (2013). BPI Challenge 2013, incidents. 4TU.ResearchData,
DOI 10.4121/500573e6-accc-4b0c-9576-aa5468b10cee. 7,554 cases / 65,533 events,
2010-03-31 to 2012-05-22.  Used here as cross-log replication: a second ITSM
incident process, a different organisation, and a 2.1-year window against the
UCI log's three dense months.

The `.xes.gz` in the distribution is redundant -- the shipped CSV export carries
the same 7,554 traces, so no XES parser is needed.

Two things differ from the UCI log and both are modelling choices that must be
stated rather than assumed:

**There is no SLA field.**  The label is constructed: a case breaches if its
duration exceeds a fixed deadline of 240 hours (10 days), which yields a breach
rate of 0.310 against the UCI log's 0.366 -- close enough that a cross-log AUC
comparison is not mostly a comparison of prevalence.

The deadline is a stated constant, not a fitted quantile.  A constant matches
how service targets actually work: an SLA is a policy someone wrote down, not a
quantile of last quarter.  It also keeps the label independent of the split, so
no part of it can move when the fold boundaries do.

**Arrivals are bunched and cases run long, so a chronological boundary cuts
through a lot of open work.**  Event timestamps span 2010 to 2012, but the
middle 50% of cases *start* inside a six-day window, 2012-04-27 to 2012-05-03,
while the median case runs 181h.  Under `temporal_case_split` that costs
1,456 of 5,658 modelling cases to straddle removal (25.7%), against 2,718 of
17,121 on UCI (15.9%).  `run_experiment.py --split-mode random_case` is the
documented deviation: a case-level random split, which still prevents a case's
prefixes from spanning folds but makes no out-of-time claim.

    Measured 2026-09-05.  Three figures previously stated here did not
    reproduce and have been corrected: "7,483 of 7,545 cases start inside a
    ten-day window" (those 7,483 starts actually span 124 days; the densest
    ten-day window holds 6,284), "straddle removal discards 4,081 of 5,658"
    (it discards 1,456), and "the fitted p65 lands at 0.6h, labelling 82% of
    test as breached" (with the current four-fold defaults it lands at 197.4h
    and labels 13.5%).  A 25.7% straddle rate is a real cost but not a
    disqualifying one, so the choice of a random split for this log is worth
    re-examining rather than treated as settled.

**The deadline makes some prefixes deterministic.**  A case whose elapsed time
at event k already exceeds the deadline has, by construction, breached: `y` is
knowable from `num__elapsed_h` alone.  `still_at_risk` marks those rows so the
primary curve can exclude them.  Reporting them would quietly inflate AUC
toward 1.0 at high k and is the single most likely way this replication gets
falsified.
"""

from __future__ import annotations

import pandas as pd

from ..logspec import LogSpec

SPEC = LogSpec(
    name="bpi2013",
    case_id_col="SR Number",
    order_cols=("ts",),
    timestamp_col="ts",
    activity_col="Status",
    resource_col="Involved ST",
    terminal_activities=frozenset({"Completed"}),
    # VINST records waiting as a Sub Status; the generic waiting counters are
    # driven from Status, so the wait states are surfaced through
    # dyn_cat__sub_status instead of this set.
    waiting_activities=frozenset(),
    static_cat_cols=("Product", "Country"),
    dynamic_cat_cols=(
        "Sub Status",
        "Involved ST Function Div",
        "Involved Org line 3",
        "Owner Country",
        "Owner First Name",
    ),
    dynamic_num_cols=(),
    missing_tokens=(),
    k_max=12,
    label_kind="fixed_deadline",
    label_params={"deadline_hours": 240.0},
)

# `SR Latest Impact` is excluded deliberately.  It is 100% constant within every
# case, so the leakage screen cannot tell "fixed at creation" from "final value
# written back over the whole trace" -- and its name says the value is the
# latest one.  It would cost nothing to keep: impact medians run 178-205h
# against a global 181h, so it barely separates duration at all.  Given no
# upside and an unverifiable provenance, it stays out.
FORBIDDEN = frozenset({"SR Latest Impact", "_case_end_ts", "case_duration_h"})

_RAW_TO_CANON = {
    "Sub Status": "sub_status",
    "Involved ST Function Div": "involved_st_function_div",
    "Involved Org line 3": "involved_org_line3",
    "Owner Country": "owner_country",
    "Owner First Name": "owner_first_name",
    "Product": "product",
    "Country": "country",
}


def load_events(path: str, spec: LogSpec = SPEC) -> pd.DataFrame:
    """Read the VINST CSV export and return the canonical event frame."""
    df = pd.read_csv(path, sep=";", encoding="latin-1")
    df["ts"] = pd.to_datetime(df["Change Date+Time"], format="ISO8601", utc=True).dt.tz_localize(
        None
    )

    cid = spec.case_id_col
    df = df.sort_values([cid, "ts"], kind="stable").reset_index(drop=True)

    # 8 cases end in `Accepted` rather than `Completed`: still open when the log
    # was cut. These are the only genuinely right-censored cases in either log,
    # and a duration-based label is undefined for them.
    last_status = df.groupby(cid, sort=False)["Status"].last()
    censored = set(last_status[last_status != "Completed"].index)
    df = df[~df[cid].isin(censored)]

    g_full = df.groupby(cid, sort=False)
    book = pd.DataFrame(
        {
            "_case_start_ts": g_full["ts"].first(),
            "_case_end_ts": g_full["ts"].max(),
            "_n_events_full": g_full.size(),
        }
    )

    static_src = [c for c in spec.static_cat_cols if c in df.columns]
    statics = g_full[list(static_src)].first()
    statics.columns = [f"static__{_RAW_TO_CANON.get(c, c)}" for c in statics.columns]

    df = df[~df[spec.activity_col].isin(spec.terminal_activities)].copy()

    out = pd.DataFrame(index=df.index)
    out["case_id"] = df[cid].astype(str)
    out["event_ts"] = df["ts"]
    out["activity"] = df[spec.activity_col].astype(str)
    out["resource"] = df[spec.resource_col].where(df[spec.resource_col].notna())

    for c in spec.dynamic_cat_cols:
        if c in df.columns:
            out[f"dyn_cat__{_RAW_TO_CANON.get(c, c)}"] = df[c]

    out = out.sort_values(["case_id", "event_ts"], kind="stable")
    out["event_idx"] = out.groupby("case_id", sort=False).cumcount()

    book.index = book.index.astype(str)
    statics.index = statics.index.astype(str)
    out = out.merge(book, left_on="case_id", right_index=True, how="left")
    out = out.merge(statics, left_on="case_id", right_index=True, how="left")

    out.attrs["n_cases_full"] = int(book.shape[0])
    out.attrs["n_cases_censored_dropped"] = int(len(censored))
    out.attrs["n_cases_dropped_all_terminal"] = int(
        book.shape[0] - out["case_id"].nunique()
    )
    out.attrs["spec_name"] = spec.name
    return out.reset_index(drop=True)


def make_labels(events: pd.DataFrame, spec: LogSpec = SPEC, fit_cases=None) -> pd.DataFrame:
    """Construct the breach label: duration beyond a fixed policy deadline.

    Nothing is estimated from data, so `fit_cases` is unused and no part of the
    label can leak across the split.  See the module docstring for why a fitted
    percentile was abandoned.
    """
    starts = events.groupby("case_id")["_case_start_ts"].first()
    ends = events.groupby("case_id")["_case_end_ts"].first()
    dur = (ends - starts).dt.total_seconds() / 3600.0

    deadline = float(spec.label_params["deadline_hours"])
    out = (dur > deadline).astype(int).rename("y").to_frame()
    out["_case_duration_h"] = dur
    out.attrs["deadline_hours"] = deadline
    out.attrs["source"] = "fixed policy constant, not fitted"
    return out


def mark_still_at_risk(prefix_df: pd.DataFrame, deadline_hours: float) -> pd.Series:
    """True where the outcome is not yet determined by elapsed time alone.

    Once `num__elapsed_h` passes the deadline the case has already breached, so
    a model scoring that prefix is reading the answer off the clock rather than
    predicting anything.
    """
    return prefix_df["num__elapsed_h"] < deadline_hours
