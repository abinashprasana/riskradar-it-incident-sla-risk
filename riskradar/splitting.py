"""Out-of-time splitting with straddle removal.

A random split over a prefix log is catastrophic: prefixes of the same case land
on both sides, so the model sees a case's own future.  Splitting at the case
level fixes that but is still not enough.  Weytjens & De Weerdt ("Creating
Unbiased Public Benchmark Datasets with Data Leakage Prevention for Predictive
Process Monitoring", BPM 2021 Workshops, LNBIP 436, pp. 18-29) show that a plain
temporal split still leaks, because a case that starts before the boundary and
finishes after it was executing under conditions the test period also
experienced.  Their correction, implemented here, is to drop cases that straddle
a boundary.

This matters more than usual for the UCI log: the leakage screen finds that
`opened_at` alone separates the outcome at AUC 0.708, which is not the ticket
telling us anything -- it is the period it belongs to.  Under a random split
that period signal is free accuracy.  Under this split it is not.

Folds are assigned from a single case-level scalar (`case_start_ts`), so a case
cannot appear in two folds by construction, and every prefix inherits its
case's fold.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class TemporalSplit:
    """Four folds, each with exactly one job.

    ``train``  fit model parameters.
    ``val``    hyperparameter search, early stopping, architecture choice.
    ``calib``  the calibrator and the conformal quantiles. Never trained on,
               never searched on.
    ``test``   scored once, at the very end.

    Keeping ``val`` and ``calib`` separate matters more than it looks. Selecting
    a model and then calibrating it on the same held-out rows makes the reported
    calibration optimistic for the same reason selecting on test does: the fold
    has already been used to make a choice.
    """

    train: pd.Index
    calib: pd.Index
    test: pd.Index
    test_boundary: pd.Timestamp
    val: pd.Index = field(default_factory=lambda: pd.Index([]))
    calib_mode: str = "random"
    n_dropped_straddling: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "n_train_cases": int(len(self.train)),
            "n_val_cases": int(len(self.val)),
            "n_calib_cases": int(len(self.calib)),
            "n_test_cases": int(len(self.test)),
            # The only temporal boundary that exists. Under calib_mode="random"
            # train, val and calib are one pool split at random, so quoting a
            # cut date between them would imply an ordering that is not there.
            "test_boundary": str(self.test_boundary),
            "calib_mode": self.calib_mode,
            "n_dropped_straddling": dict(self.n_dropped_straddling),
        }


def build_case_index(events: pd.DataFrame, labels: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per case: start, end, observable trace length, and optionally y.

    `labels` is optional because the split must be computable without them.  In
    BPI 2013 the label is a duration threshold fitted on the training cases, so
    the label depends on the split; requiring labels up front would force that
    threshold to be fitted on the whole log, which is exactly the leak the
    fitted-on-train rule exists to prevent.
    """
    g = events.groupby("case_id")
    idx = pd.DataFrame(
        {
            "case_start_ts": g["_case_start_ts"].first(),
            "case_end_ts": g["_case_end_ts"].first(),
            "n_nonterm": g.size(),
        }
    )
    if labels is not None:
        idx = idx.join(labels["y"], how="inner")
    return idx.sort_values("case_start_ts")


def temporal_case_split(
    case_index: pd.DataFrame,
    train_frac: float = 0.50,
    calib_frac: float = 0.10,
    val_frac: float = 0.15,
    drop_straddling: bool = True,
    calib_mode: str = "random",
    seed: int = 42,
) -> TemporalSplit:
    """Chronological train / val / calib / test split over cases.

    `val` selects models and hyperparameters, `calib` fits calibrators and
    conformal quantiles, and `test` is touched exactly once at the very end.
    Test cases are never dropped -- both logs are fully closed by the end of
    their observation window, so nothing straddles the right-hand edge.

    Only the sum `train_frac + val_frac + calib_frac` sets the test boundary, so
    the defaults here (0.50 / 0.15 / 0.10 = 0.75) place it in exactly the same
    position as the previous three-fold defaults (0.60 / 0.15 = 0.75). That is
    deliberate: adding the fourth fold re-partitions the modelling pool without
    moving a single case in or out of test, so no previously published test
    number is invalidated by the change.
    """
    if not 0 < train_frac < 1 or not 0 <= calib_frac < 1 or not 0 <= val_frac < 1:
        raise ValueError("fractions must be in [0, 1)")
    if train_frac + calib_frac + val_frac >= 1:
        raise ValueError("train_frac + val_frac + calib_frac must sum to < 1")

    ci = case_index.sort_values("case_start_ts")
    starts = ci["case_start_ts"]
    modelling_frac = train_frac + val_frac + calib_frac
    b_test = starts.quantile(modelling_frac)

    in_test = starts >= b_test
    pre_test = ~in_test

    # Straddle removal is applied ONCE, against the test boundary, and only to
    # the modelling period.  The leakage it exists to prevent is a case being
    # trained on while it is still executing during the test window; a case
    # crossing the internal train/calib line is not that.
    #
    # Applying the rule to each fold's own right-hand edge (the naive reading)
    # is actively harmful here: the calib window is ~2 weeks while breaching
    # cases run a median of 402h, so almost every breach straddles out and the
    # fold's base rate collapses to 0.03 against a population 0.38.  Selecting a
    # threshold on such a fold would be worse than not tuning at all.
    dropped: dict[str, int] = {"modelling_period": 0}
    if drop_straddling:
        keep = pre_test & (ci["case_end_ts"] < b_test)
        dropped["modelling_period"] = int((pre_test & ~keep).sum())
        pre_test = keep

    pre_idx = ci.index[pre_test]
    n_calib = int(round(len(pre_idx) * calib_frac / modelling_frac))
    n_val = int(round(len(pre_idx) * val_frac / modelling_frac))

    if calib_mode == "random":
        # Calib exists to choose a model and a threshold without touching test.
        # Every case in the pool already predates the test window, so that
        # requirement is met however calib is drawn from it -- and drawing at
        # random keeps calib distributionally identical to train.
        #
        # The temporal-tail alternative does not: survival to the test boundary
        # requires a short case, so the late portion of the modelling period is
        # systematically short and therefore systematically non-breaching (base
        # rate 0.12 against a train 0.48).  Selecting on that is selecting on an
        # artifact of the straddle rule.
        #
        # The cost, which belongs in the write-up rather than being papered
        # over: calib resembles train, not test, so anything tuned on it
        # transfers imperfectly across the base-rate drift this log exhibits
        # (0.64 in March against 0.23 in May).  Tuning to an unseen future is
        # not available at any price.
        drawn = ci.loc[pre_idx].sample(n=n_calib + n_val, random_state=seed).index
        calib_idx, val_idx = drawn[:n_calib], drawn[n_calib:]
        train_idx = pre_idx.difference(drawn)
    elif calib_mode == "temporal_tail":
        # Both held-out folds come off the late edge, calib last of all.
        cut_val = len(pre_idx) - n_calib - n_val
        train_idx = pre_idx[:cut_val]
        val_idx = pre_idx[cut_val : cut_val + n_val]
        calib_idx = pre_idx[cut_val + n_val :]
    else:
        raise ValueError(f"unknown calib_mode {calib_mode!r}")

    return TemporalSplit(
        train=train_idx,
        val=val_idx,
        calib=calib_idx,
        test=ci.index[in_test],
        test_boundary=b_test,
        calib_mode=calib_mode,
        n_dropped_straddling=dropped,
    )


def random_case_split(
    case_index: pd.DataFrame,
    train_frac: float = 0.50,
    calib_frac: float = 0.10,
    val_frac: float = 0.15,
    seed: int = 42,
) -> TemporalSplit:
    """Case-level random split, for logs that cannot support an out-of-time one.

    Still splits at the CASE level, so a case's prefixes never span folds -- the
    error that makes a naive prefix-log split worthless.  It makes no
    out-of-time claim, and results from it are not comparable to a temporally
    split log without saying so.

    This exists for BPI 2013, whose cases all arrive inside a ten-day window
    while running a median 7.5 days, so any chronological boundary is straddled
    by almost every case.  Forcing a temporal split there does not produce a
    conservative estimate; it produces a training set of sub-hour incidents and
    a meaningless one.
    """
    ci = case_index
    shuffled = ci.sample(frac=1.0, random_state=seed).index
    n_train = int(len(shuffled) * train_frac)
    n_val = int(len(shuffled) * val_frac)
    n_calib = int(len(shuffled) * calib_frac)
    a, b, c = n_train, n_train + n_val, n_train + n_val + n_calib

    return TemporalSplit(
        train=shuffled[:a],
        val=shuffled[a:b],
        calib=shuffled[b:c],
        test=shuffled[c:],
        test_boundary=pd.NaT,
        calib_mode=f"random_case_split(seed={seed})",
        n_dropped_straddling={"modelling_period": 0},
    )


def slice_prefixes(prefix_df: pd.DataFrame, cases: pd.Index) -> pd.DataFrame:
    """All prefix rows belonging to `cases`."""
    return prefix_df[prefix_df["case_id"].isin(set(cases))]
