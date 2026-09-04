"""Log-agnostic specification of an event log.

`prefix_log.py` never sees a UCI or BPI column name.  Adapters translate their
own schema into the canonical event frame described below, and everything
downstream works off that frame plus the `LogSpec` that produced it.

Canonical event frame columns
-----------------------------
case_id        str        case identifier
event_idx      int        0-based position, assigned after ordering AND after
                          terminal events are removed
event_ts       datetime   timestamp of this event
_case_start_ts datetime   constant per case, from the FULL trace
_case_end_ts   datetime   constant per case, from the FULL trace (bookkeeping
                          only -- used for straddle detection, never a feature)
activity       str        the state/status of this event
resource       str        the team/group handling this event
static__*      any        case-invariant, read at event 0
dyn_cat__*     any        categorical, value as of this event
dyn_num__*     float      running numeric counter, value as of this event

Columns beginning with a single underscore are bookkeeping and are never
offered to a model.  `feature_columns()` is the single source of truth for
what counts as a feature, and it works by whitelisting prefixes -- not by
blacklisting known-bad names.  That distinction matters: in the UCI log the
closure fields are back-filled onto every event of the trace, so a blacklist
would need to anticipate every leaky column, while a whitelist admits only
columns an adapter has explicitly declared safe.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# `idx_cat__` / `idx_num__` are produced by index encoding, which gives every
# event position its own columns. They are whitelisted here for the same reason
# as the rest: a model may only see a column some component deliberately
# promoted, never one that arrived because nobody thought to exclude it.
FEATURE_PREFIXES = ("static__", "dyn_cat__", "dyn_num__", "num__", "idx_cat__", "idx_num__")

CANONICAL_BOOKKEEPING = (
    "case_id",
    "event_idx",
    "event_ts",
    "_case_start_ts",
    "_case_end_ts",
)


@dataclass(frozen=True)
class LogSpec:
    """Everything `prefix_log` needs to know about a log, in log-neutral terms."""

    name: str

    # --- raw-schema wiring, used only inside the adapter -------------------
    case_id_col: str
    order_cols: tuple[str, ...]
    timestamp_col: str
    activity_col: str
    resource_col: str

    # --- semantics --------------------------------------------------------
    terminal_activities: frozenset[str]
    waiting_activities: frozenset[str]

    # --- feature whitelists, in RAW column names --------------------------
    static_cat_cols: tuple[str, ...] = ()
    dynamic_cat_cols: tuple[str, ...] = ()
    dynamic_num_cols: tuple[str, ...] = ()

    # --- parsing ----------------------------------------------------------
    missing_tokens: tuple[str, ...] = ()
    sentinel_activities: frozenset[str] = frozenset()

    # --- prefix / label ---------------------------------------------------
    k_max: int = 8
    label_kind: Literal["flag", "duration_percentile"] = "flag"
    label_params: dict = field(default_factory=dict)

    def is_waiting(self, activity: str) -> bool:
        return activity in self.waiting_activities


def feature_columns(df) -> list[str]:
    """Columns of `df` that are legitimate model features.

    Whitelist by prefix.  Anything an adapter did not explicitly promote into
    a `static__` / `dyn_cat__` / `dyn_num__` column, and anything `prefix_log`
    did not derive into a `num__` column, is invisible to the model.
    """
    return [c for c in df.columns if c.startswith(FEATURE_PREFIXES)]


def split_feature_columns(df) -> tuple[list[str], list[str]]:
    """Return (numeric_features, categorical_features) among the feature columns."""
    import pandas as pd

    feats = feature_columns(df)
    num = [c for c in feats if pd.api.types.is_numeric_dtype(df[c])]
    cat = [c for c in feats if c not in num]
    return num, cat
