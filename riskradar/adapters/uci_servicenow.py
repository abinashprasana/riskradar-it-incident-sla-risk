"""Adapter for the UCI 'Incident Management Process Enriched Event Log'.

Amaral, Fantinato & Peres (2018), DOI 10.24432/C57S4H.  ServiceNow export of
24,918 incidents / 141,712 events covering 2016-02-29 to 2017-02-16.

Two properties of this log drive every decision below.

1.  It is a *snapshot export of closed tickets*, not a live event stream.  The
    closure fields are back-filled onto every row of the trace: at the first
    event `closed_at` is 100% populated and already equal to its final value,
    `closed_code` 99.6%, `resolved_by` 99.6%, `resolved_at` 93.8%.  Knowing
    `closed_code` at event 1 splits the breach rate from 0.00 to 0.81.  None of
    these columns appear in any whitelist below, and `prefix_log.leakage_screen`
    exists to catch this class of column automatically.

2.  Every incident terminates in `Closed`; none are genuinely open.  The
    Resolved/Closed events are outcome-determining and are dropped as ROWS, not
    merely excluded as columns.  1,816 incidents (7.29%) consist solely of such
    events and therefore leave the study entirely.  They are not a random
    subset (breach rate 0.163 against 0.382 for those retained), which
    `splitting.build_case_index` reports as a selection-bias check.
"""

from __future__ import annotations

import pandas as pd

from ..logspec import LogSpec

SPEC = LogSpec(
    name="uci_servicenow",
    case_id_col="number",
    # sys_mod_count is the reliable order key: unique within 99.98% of cases,
    # non-monotonic in only 4.  sys_updated_at ties in 37.4% of cases and
    # cannot order alone, so it is the tie-break rather than the key.
    order_cols=("sys_mod_count", "sys_updated_at"),
    timestamp_col="sys_updated_at",
    activity_col="incident_state",
    resource_col="assignment_group",
    terminal_activities=frozenset({"Resolved", "Closed"}),
    waiting_activities=frozenset(
        {"Awaiting User Info", "Awaiting Vendor", "Awaiting Problem", "Awaiting Evidence"}
    ),
    static_cat_cols=(
        "category",
        "subcategory",
        "u_symptom",
        "contact_type",
        "location",
        "impact",
        "urgency",
        "caller_id",
        "opened_by",
        "sys_created_by",
        "knowledge",
        "u_priority_confirmation",
        "notify",
    ),
    # `assignment_group` is deliberately absent here: it is already the
    # `resource_col` above, and `prefix_log` surfaces that as
    # `dyn_cat__resource`. Listing it in both places emitted two byte-identical
    # feature columns (verified equal on 100.0% of prefix rows), which inflates
    # the apparent feature count and double-weights one attribute in any
    # distance- or split-based learner.
    dynamic_cat_cols=("priority",),
    dynamic_num_cols=("reassignment_count", "reopen_count", "sys_mod_count"),
    missing_tokens=("?",),
    # 5 events carry this sentinel in incident_state; dropped explicitly rather
    # than silently coerced into a category.
    sentinel_activities=frozenset({"-100"}),
    k_max=8,
    label_kind="flag",
    label_params={"flag_col": "made_sla", "breach_when": False},
)

_DATE_COLS = ("opened_at", "resolved_at", "closed_at", "sys_updated_at", "sys_created_at")
_NUM_COLS = ("reassignment_count", "reopen_count", "sys_mod_count")

# Present in the raw file and deliberately never promoted to a feature.
# Recorded so tests can assert none of them reaches the model matrix.
FORBIDDEN = frozenset(
    {
        "made_sla",       # the label
        "active",         # 1:1 with Closed; label proxy
        "closed_at",      # back-filled, 100% populated and final at event 1
        "closed_code",    # back-filled, 99.6% at event 1
        "resolved_by",    # back-filled, 99.6% at event 1
        "resolved_at",    # back-filled, 93.8% at event 1
        "problem_id",     # 98%+ missing and typically linked post-hoc
        "rfc",
        "vendor",
        "caused_by",
    }
)


def load_events(path: str, spec: LogSpec = SPEC) -> pd.DataFrame:
    """Read the raw CSV and return the canonical event frame.

    Case-level bookkeeping (`_case_start_ts`, `_case_end_ts`, `_made_sla_final`)
    is computed from the FULL trace before terminal events are removed, because
    the label and the true end time both live in the events about to be dropped.
    """
    df = pd.read_csv(path, na_values=list(spec.missing_tokens), low_memory=False)

    for c in _DATE_COLS:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", dayfirst=True)
    for c in _NUM_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "made_sla" in df.columns:
        s = df["made_sla"].astype(str).str.strip().str.lower()
        df["made_sla"] = s.map({"true": True, "false": False})

    cid = spec.case_id_col
    df = df.sort_values([cid, *spec.order_cols], kind="stable").reset_index(drop=True)

    # --- case-level bookkeeping, from the full trace ----------------------
    g_full = df.groupby(cid, sort=False)
    book = pd.DataFrame(
        {
            "_case_start_ts": g_full["opened_at"].first(),
            # closed_at has zero nulls and is constant within a case.
            "_case_end_ts": g_full["closed_at"].max(),
            "_made_sla_final": g_full["made_sla"].last(),
            "_n_events_full": g_full.size(),
        }
    )

    # --- static attributes, read at the first event of the full trace -----
    static_src = [c for c in spec.static_cat_cols if c in df.columns]
    statics = g_full[list(static_src)].first() if static_src else pd.DataFrame(index=book.index)
    statics = statics.add_prefix("static__")
    # cmdb_ci is 99.7% missing, so only its presence carries signal.
    if "cmdb_ci" in df.columns:
        statics["static__cmdb_ci_known"] = g_full["cmdb_ci"].first().notna().astype(int)

    # --- drop sentinel and terminal events --------------------------------
    n_before = len(df)
    if spec.sentinel_activities:
        df = df[~df[spec.activity_col].astype(str).isin(spec.sentinel_activities)]
    n_sentinel = n_before - len(df)

    df = df[~df[spec.activity_col].isin(spec.terminal_activities)].copy()

    # --- canonical frame ---------------------------------------------------
    out = pd.DataFrame(index=df.index)
    out["case_id"] = df[cid].astype(str)
    out["event_ts"] = df[spec.timestamp_col]
    out["activity"] = df[spec.activity_col].astype(str)
    out["resource"] = df[spec.resource_col].where(df[spec.resource_col].notna())

    for c in spec.dynamic_cat_cols:
        if c in df.columns:
            out[f"dyn_cat__{c}"] = df[c]
    for c in spec.dynamic_num_cols:
        if c in df.columns:
            out[f"dyn_num__{c}"] = df[c]
    if "assigned_to" in df.columns:
        out["dyn_cat__assigned_to_known"] = df["assigned_to"].notna().astype(int)

    out = out.sort_values(["case_id", "event_ts"], kind="stable")
    out["event_idx"] = out.groupby("case_id", sort=False).cumcount()

    book.index = book.index.astype(str)
    statics.index = statics.index.astype(str)
    out = out.merge(book, left_on="case_id", right_index=True, how="left")
    out = out.merge(statics, left_on="case_id", right_index=True, how="left")

    # --- selection-bias bookkeeping ---------------------------------------
    # Cases consisting only of terminal events vanish here. They are not a
    # random subset, so record the size and outcome rate of what was lost.
    y_full = book["_made_sla_final"].map(lambda v: 1 if v is False else (0 if v is True else None))
    kept_ids = set(out["case_id"].unique())
    lost = y_full[~y_full.index.isin(kept_ids)].dropna()

    out.attrs["n_sentinel_events_dropped"] = int(n_sentinel)
    out.attrs["n_cases_full"] = int(book.shape[0])
    out.attrs["n_cases_dropped_all_terminal"] = int(lost.shape[0])
    out.attrs["breach_rate_dropped"] = float(lost.mean()) if len(lost) else float("nan")
    out.attrs["breach_rate_kept"] = float(y_full[y_full.index.isin(kept_ids)].dropna().mean())
    out.attrs["spec_name"] = spec.name
    return out.reset_index(drop=True)


def make_labels(events: pd.DataFrame, spec: LogSpec = SPEC, fit_cases=None) -> pd.DataFrame:
    """Case-level label frame: index case_id, column `y`.

    `fit_cases` is ignored here.  The UCI log ships a native outcome flag, so
    nothing about the label is estimated from data and there is no fold
    dependence.  The parameter exists because the BPI adapter must fit its
    deadline on training cases only, and both adapters share one call site.
    """
    lab = (
        events.groupby("case_id")["_made_sla_final"]
        .first()
        .map(lambda v: 1 if v is False else (0 if v is True else None))
    )
    return lab.dropna().astype(int).rename("y").to_frame()
