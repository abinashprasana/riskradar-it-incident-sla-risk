"""Prefix-log construction and the leakage screen.

Outcome-oriented predictive process monitoring (Teinemaa, Dumas, La Rosa &
Maggi, ACM TKDD 13(2), 2019) classifies an ongoing case from a *prefix* of its
trace.  Instead of one feature vector per completed case, we emit one row per
(case, k) where k is the number of events observed, and every feature is
computed from events [0:k] alone.

The implementation exploits a convenient identity: because a prefix of length k
ends at event k-1, each event row of the canonical frame IS exactly one prefix
row.  So the whole prefix log is built with vectorised cumulative operations
over the event frame in one pass, rather than a Python loop over
(case, k) pairs.

`leakage_screen` is the safety net.  The UCI log back-fills its closure fields
onto every event of the trace, which is invisible to inspection and fatal to
evaluation.  Rather than trusting a hand-maintained blacklist, the screen
measures, for every candidate raw column, how much of the outcome is already
knowable at event 1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .logspec import FEATURE_PREFIXES, LogSpec, feature_columns

# A datetime column whose event-1 value points past the event-1 timestamp this
# often is describing something that has not happened yet.
FUTURE_REJECT = 0.50
# A column whose event-1 value alone separates the outcome this well warrants
# an explicit justification before use.
AUC_FLAG = 0.60


def build_prefix_log(
    events: pd.DataFrame,
    labels: pd.DataFrame,
    spec: LogSpec,
    k_max: int | None = None,
) -> pd.DataFrame:
    """Build the prefix log from a canonical event frame.

    One row per (case_id, k) for k = 1..min(n_nonterm, k_max).  All `num__`
    features are running values over events [0:k]; all `dyn_*` features are the
    value at event k-1; all `static__` features are case-invariant.
    """
    k_max = int(k_max if k_max is not None else spec.k_max)

    df = events.sort_values(["case_id", "event_idx"], kind="stable").reset_index(drop=True)
    g = df.groupby("case_id", sort=False)

    # Cases keep their true length even though prefixes are capped at k_max.
    n_nonterm = g["event_idx"].transform("size")

    # --- running structural counts ---------------------------------------
    first_act = ~df.duplicated(subset=["case_id", "activity"], keep="first")
    first_res = ~df.duplicated(subset=["case_id", "resource"], keep="first")
    prev_res = g["resource"].shift(1)
    is_wait = df["activity"].isin(spec.waiting_activities)

    out = pd.DataFrame(
        {
            "case_id": df["case_id"],
            "k": df["event_idx"] + 1,
            "event_ts": df["event_ts"],
            "_case_start_ts": df["_case_start_ts"],
            "_case_end_ts": df["_case_end_ts"],
            "_n_nonterm": n_nonterm,
        }
    )

    out["num__k"] = out["k"]
    out["num__n_distinct_activities"] = first_act.groupby(df["case_id"]).cumsum()
    out["num__n_distinct_resources"] = first_res.groupby(df["case_id"]).cumsum()
    out["num__n_resource_changes"] = (
        (df["resource"].ne(prev_res) & prev_res.notna()).groupby(df["case_id"]).cumsum()
    )
    out["num__n_activity_repeats"] = out["num__k"] - out["num__n_distinct_activities"]
    out["num__n_waiting_events"] = is_wait.groupby(df["case_id"]).cumsum()
    out["num__frac_waiting"] = out["num__n_waiting_events"] / out["num__k"]

    # --- timing -----------------------------------------------------------
    gap_h = g["event_ts"].diff().dt.total_seconds() / 3600.0
    out["num__last_gap_h"] = gap_h
    out["num__max_gap_h"] = gap_h.groupby(df["case_id"]).cummax()
    elapsed = (df["event_ts"] - df["_case_start_ts"]).dt.total_seconds() / 3600.0
    # 5 UCI events carry a first sys_updated_at up to 2.5h BEFORE opened_at --
    # a back-dating or timezone artifact. A duration cannot be negative, so
    # clip, but record the count rather than letting it disappear.
    n_negative_elapsed = int((elapsed < 0).sum())
    elapsed = elapsed.clip(lower=0)
    out["num__elapsed_h"] = elapsed
    out["num__log_elapsed_h"] = np.log1p(elapsed)
    # Mean gap is elapsed-since-first-EVENT spread over the gaps seen so far;
    # undefined at k=1, where there is no gap yet.
    since_first_event = (
        df["event_ts"] - g["event_ts"].transform("first")
    ).dt.total_seconds() / 3600.0
    out["num__mean_gap_h"] = np.where(
        out["num__k"] > 1, since_first_event / (out["num__k"] - 1).clip(lower=1), np.nan
    )
    out["num__hour_of_k"] = df["event_ts"].dt.hour
    out["num__dow_of_k"] = df["event_ts"].dt.dayofweek

    # --- system load, past-only -------------------------------------------
    # Counting concurrently-OPEN cases would require other cases' end times,
    # which are future information. Trailing arrival counts are past-only.
    starts = np.sort(df.groupby("case_id")["_case_start_ts"].first().values)
    ts = df["event_ts"].values
    for label, hours in (("arrivals_24h", 24), ("arrivals_7d", 24 * 7)):
        lo = ts - np.timedelta64(hours, "h")
        out[f"num__{label}"] = np.searchsorted(starts, ts, "right") - np.searchsorted(
            starts, lo, "left"
        )

    # --- static case attributes ------------------------------------------
    out["static__opened_hour"] = df["_case_start_ts"].dt.hour
    out["static__opened_dow"] = df["_case_start_ts"].dt.dayofweek
    out["static__opened_is_offhours"] = (
        (df["_case_start_ts"].dt.hour < 8)
        | (df["_case_start_ts"].dt.hour >= 18)
        | (df["_case_start_ts"].dt.dayofweek >= 5)
    ).astype(int)

    # --- carry through adapter-declared features --------------------------
    carried = [c for c in df.columns if c.startswith(FEATURE_PREFIXES)]
    for c in carried:
        out[c] = df[c]

    # `activity` is the current process state -- legitimate as-of-k information
    # and the single most process-aware categorical available.
    out["dyn_cat__activity"] = df["activity"]
    out["dyn_cat__activity_prev"] = g["activity"].shift(1)
    out["dyn_cat__resource"] = df["resource"]

    # --- cap and label ----------------------------------------------------
    out = out[out["k"] <= k_max].copy()
    out = out.merge(labels, left_on="case_id", right_index=True, how="inner")

    out.attrs["k_max"] = k_max
    out.attrs["n_negative_elapsed_clipped"] = n_negative_elapsed
    out.attrs["spec_name"] = spec.name
    return out.reset_index(drop=True)


def leakage_screen(
    raw: pd.DataFrame,
    case_id_col: str,
    order_cols: tuple[str, ...],
    y: pd.Series,
    timestamp_col: str,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Audit every raw column for information that should not exist at event 1.

    Returns one row per column with:

    ``constant_rate``   share of cases where the column never changes.  This is
                        NOT evidence of leakage on its own: `category` is
                        constant because a ticket's category is fixed at
                        creation, and `closed_at` is constant because the
                        export wrote the future backwards into every row.  Both
                        score 1.0, so the statistic cannot separate them.
    ``future_ref_rate`` for datetime columns only: share of cases whose event-1
                        value lies *after* the event-1 timestamp.  This is the
                        one mechanical proof of leakage available -- a column
                        cannot legitimately describe a moment that has not
                        arrived.
    ``auc_k1``          how well the event-1 value alone separates the label.
                        Categoricals are scored with in-sample target encoding,
                        which is deliberately optimistic: this is a screen, so
                        false alarms are cheaper than misses.
    ``verdict``         REJECT_FUTURE / FLAG_HIGH_AUC / ok

    The screen is a safety net, not the primary defence.  For non-temporal
    post-hoc columns such as `closed_code` or `resolved_by` no statistic
    distinguishes "fixed at creation" from "back-filled at closure" -- only
    knowing what the field means does.  That is why `logspec.feature_columns`
    whitelists by prefix and adapters must promote a column explicitly before a
    model can ever see it.  Run this against the RAW log: its job is to inform
    what may enter the whitelist in the first place.
    """
    df = raw.sort_values([case_id_col, *order_cols], kind="stable")
    g = df.groupby(case_id_col, sort=False)
    cols = columns or [c for c in df.columns if c not in (case_id_col, *order_cols)]

    y = y.copy()
    y.index = y.index.astype(str)
    ts_first = g[timestamp_col].first()
    ts_first.index = ts_first.index.astype(str)
    rows = []

    for c in cols:
        first = g[c].first()
        first.index = first.index.astype(str)
        nuniq = g[c].nunique(dropna=True)
        nuniq.index = nuniq.index.astype(str)
        constant = float((nuniq <= 1).mean())

        future = np.nan
        if pd.api.types.is_datetime64_any_dtype(df[c]) and c != timestamp_col:
            m = first.notna() & ts_first.notna()
            if m.any():
                future = float((first[m] > ts_first[m]).mean())

        yy = y.reindex(first.index).dropna()
        v = first.reindex(yy.index)
        try:
            if pd.api.types.is_numeric_dtype(v) and not pd.api.types.is_bool_dtype(v):
                auc = roc_auc_score(yy, v.fillna(v.median()))
            elif pd.api.types.is_datetime64_any_dtype(v):
                auc = roc_auc_score(yy, v.astype("int64"))
            else:
                enc = yy.groupby(v.astype("string").fillna("__NA__")).transform("mean")
                auc = roc_auc_score(yy, enc)
            auc = float(max(auc, 1.0 - auc))
        except (ValueError, TypeError):
            auc = np.nan

        if future == future and future > FUTURE_REJECT:
            verdict = "REJECT_FUTURE"
        elif auc == auc and auc > AUC_FLAG:
            verdict = "FLAG_HIGH_AUC"
        else:
            verdict = "ok"

        rows.append(
            {
                "column": c,
                "constant_rate": constant,
                "future_ref_rate": future,
                "auc_k1": auc,
                "verdict": verdict,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(["verdict", "auc_k1"], ascending=[True, False])
        .reset_index(drop=True)
    )


def assert_no_leakage(prefix_df: pd.DataFrame, forbidden: frozenset[str]) -> None:
    """Fail loudly if a forbidden raw column reached the feature matrix.

    Checks the bare name too, so `static__closed_code` is caught as readily as
    `closed_code`.
    """
    feats = feature_columns(prefix_df)
    bare = {f.split("__", 1)[-1] for f in feats}
    hit = sorted((bare & forbidden) | (set(feats) & forbidden))
    if hit:
        raise AssertionError(f"forbidden columns reached the feature matrix: {hit}")
