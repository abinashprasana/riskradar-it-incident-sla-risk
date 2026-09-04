"""Feature encoding for the prefix log.

Teinemaa et al. (ACM TKDD 2019) separate two axes: how a prefix is turned into
a feature vector (the encoding) and how prefixes are grouped into models (the
bucketing).  Both combinations used here are implemented in this module.

**Aggregation encoding, single bucket.**  The feature vector has the same width
at every prefix length, so one model serves all k.  `prefix_log` already emits
running aggregates per (case, k), so encoding is column-typing rather than a
reshape.

**Index encoding, prefix-length bucketing.**  Every event position gets its own
columns -- `activity` at position 1, at position 2, and so on -- which preserves
order that aggregation throws away.  The width then depends on k, so each
prefix length needs its own model.  That is the trade: higher fidelity, but the
training set for bucket k is only the cases long enough to reach k, and it
shrinks fast (UCI: 11.5k cases at k=1, 1.5k at k=8).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .logspec import split_feature_columns


def build_preprocess_pipeline(
    X: pd.DataFrame,
    min_frequency: int = 30,
    max_categories: int = 50,
) -> ColumnTransformer:
    """Median-impute numerics; constant-fill and one-hot encode categoricals.

    Two deliberate departures from the original `feature_engineering.py`:

    Categoricals are filled with an explicit `__MISSING__` level rather than the
    mode.  Missingness here is informative and often near-total -- `cmdb_ci` is
    99.7% absent -- so imputing the mode would invent a dominant category that
    does not exist and discard the only signal the column carries.

    One-hot encoding is bounded by `min_frequency` and `max_categories`.  The
    log has thousands of distinct `caller_id` values; unbounded encoding was
    what inflated the original artifact to 56 MB, and rare levels are noise the
    model would otherwise memorise.
    """
    num_cols, cat_cols = split_feature_columns(X)

    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        # Numeric features here span wildly different scales --
                        # elapsed hours reach the thousands while counts sit in
                        # single digits. Without scaling, lbfgs fails to
                        # converge and the logistic-regression arm is compared
                        # while handicapped, which would make the encoding
                        # comparison a comparison of convergence instead. Trees
                        # are unaffected by a monotonic rescaling.
                        ("scale", StandardScaler()),
                    ]
                ),
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
                        (
                            "encode",
                            OneHotEncoder(
                                handle_unknown="infrequent_if_exist",
                                min_frequency=min_frequency,
                                max_categories=max_categories,
                            ),
                        ),
                    ]
                ),
                cat_cols,
            ),
        ],
        remainder="drop",
    )


def aggregation_encode(prefix_df: pd.DataFrame) -> pd.DataFrame:
    """Select the model matrix from a prefix log.

    Whitelist-driven: only `static__` / `dyn_cat__` / `dyn_num__` / `num__`
    columns survive, so bookkeeping and the label cannot reach a model by
    accident.  Categoricals are cast to string because a one-hot encoder must
    not treat a category code as an ordered quantity.

    The cast targets plain `object` with `np.nan` rather than pandas' nullable
    `string` dtype: sklearn's imputer detects missing values with `X != X`,
    which raises on `pd.NA` instead of returning a mask.
    """
    num_cols, cat_cols = split_feature_columns(prefix_df)
    X = prefix_df[num_cols + cat_cols].copy()
    for c in cat_cols:
        s = X[c]
        X[c] = np.where(s.notna(), s.astype(str), np.nan)
    return X


def index_encode(prefix_df: pd.DataFrame, events: pd.DataFrame, k: int) -> pd.DataFrame:
    """Wide encoding for one prefix-length bucket: a column per (attribute, position).

    Returns a frame indexed by `case_id`, covering exactly the cases that have a
    prefix of length `k`, with:

    * `static__*` -- case-invariant, carried through unchanged;
    * `idx_cat__<attr>_<p>` -- categorical `attr` as observed at position `p`;
    * `idx_num__<attr>_<p>` -- numeric `attr` at position `p`, plus elapsed time.

    Position columns run 1..k, so ordering survives: "reassigned, then waited"
    and "waited, then reassigned" are different vectors here and identical under
    aggregation encoding.
    """
    bucket = prefix_df[prefix_df["k"] == k]
    cases = set(bucket["case_id"])
    if not cases:
        return pd.DataFrame()

    ev = events[events["case_id"].isin(cases) & (events["event_idx"] < k)].copy()
    ev["elapsed_h"] = (
        (ev["event_ts"] - ev["_case_start_ts"]).dt.total_seconds() / 3600.0
    ).clip(lower=0)

    cat_src = ["activity", "resource"] + [c for c in ev.columns if c.startswith("dyn_cat__")]
    num_src = ["elapsed_h"] + [c for c in ev.columns if c.startswith("dyn_num__")]

    idx = ev.set_index(["case_id", "event_idx"]).sort_index()
    parts = []
    for c in cat_src:
        w = idx[c].unstack("event_idx")
        w.columns = [f"idx_cat__{c.replace('dyn_cat__', '')}_{int(p) + 1}" for p in w.columns]
        parts.append(w.astype(object))
    for c in num_src:
        w = idx[c].unstack("event_idx")
        w.columns = [f"idx_num__{c.replace('dyn_num__', '')}_{int(p) + 1}" for p in w.columns]
        parts.append(w)

    X = pd.concat(parts, axis=1)

    static_cols = [c for c in bucket.columns if c.startswith("static__")]
    statics = bucket.set_index("case_id")[static_cols]
    X = X.join(statics, how="left")

    for c in X.columns:
        if c.startswith("idx_cat__"):
            s = X[c]
            X[c] = np.where(s.notna(), s.astype(str), np.nan)

    return X.reindex(bucket["case_id"].to_numpy())


def make_buckets(prefix_df: pd.DataFrame, k_max: int) -> dict[int, pd.DataFrame]:
    """Split a prefix log into one frame per prefix length."""
    return {k: prefix_df[prefix_df["k"] == k] for k in range(1, k_max + 1)}
