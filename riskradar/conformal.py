"""Split conformal prediction, conditioned on prefix length.

Turns a score into a *set* of labels that contains the truth with a stated
frequency. A singleton {breach} is a confident call; {breach, no-breach} means
the model declines to commit at that confidence level. That is what closes the
loop on the project's own question -- the earliness curve says accuracy rises
with k, and this says at which k a decision meets a stated reliability bar.

Two design points that are specific to this data rather than to conformal
prediction in general:

**Binning.** Coverage should hold at each prefix length, not merely on average,
so the calibration is Mondrian-conditioned on k. Label-conditional bins (k x
class) would be better still, but the calibration fold has 73 negatives at k=7
and 46 at k=8 -- a 90% quantile estimated from ~5 order statistics is
calibration noise, not a guarantee. So k in {6, 7, 8} is pooled into one bin
(278 negatives), which is defensible because the base rate is nearly flat
across those three lengths (0.70 / 0.77 / 0.79).

**Validity.** Split conformal guarantees marginal coverage under
exchangeability. A deliberately temporal split breaks exchangeability, and this
log drifts hard (train prior 0.56 against test 0.40). Under-coverage is
therefore an expected *measurement* of that drift, not a bug in the method, and
it is reported rather than explained away.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Prefix lengths sharing a calibration bin, because individually they are too
# small to estimate a quantile from. See module docstring.
DEFAULT_POOL = (6, 7, 8)


def bin_of(k: np.ndarray, pool: tuple[int, ...] = DEFAULT_POOL) -> np.ndarray:
    """Map prefix length to Mondrian bin id; pooled lengths share the lowest id."""
    k = np.asarray(k)
    if not pool:
        return k.astype(int)
    lo = min(pool)
    return np.where(np.isin(k, pool), lo, k).astype(int)


@dataclass
class ConformalCalibration:
    """Per-bin nonconformity quantiles at a fixed miscoverage level."""

    epsilon: float
    quantiles: dict[int, float] = field(default_factory=dict)
    n_per_bin: dict[int, int] = field(default_factory=dict)
    pool: tuple[int, ...] = DEFAULT_POOL

    def as_dict(self) -> dict:
        return {
            "epsilon": self.epsilon,
            "nominal_coverage": 1 - self.epsilon,
            "pooled_bins": list(self.pool),
            "quantile_by_bin": {str(k): round(v, 5) for k, v in sorted(self.quantiles.items())},
            "n_by_bin": {str(k): v for k, v in sorted(self.n_per_bin.items())},
        }


def _nonconformity(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    """1 - p(true label). Large means the model was confidently wrong."""
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    y = np.asarray(y)
    return np.where(y == 1, 1.0 - p, p)


def calibrate(
    p: np.ndarray,
    y: np.ndarray,
    k: np.ndarray,
    epsilon: float = 0.1,
    pool: tuple[int, ...] = DEFAULT_POOL,
) -> ConformalCalibration:
    """Per-bin conformal quantile at the ceil((n+1)(1-eps)) order statistic.

    The finite-sample correction (n+1 rather than n) is what makes the coverage
    guarantee exact rather than asymptotic, and it matters at these fold sizes.
    """
    alpha = _nonconformity(p, y)
    bins = bin_of(k, pool)
    cal = ConformalCalibration(epsilon=epsilon, pool=pool)

    for b in np.unique(bins):
        a = np.sort(alpha[bins == b])
        n = len(a)
        if n == 0:
            continue
        rank = int(np.ceil((n + 1) * (1 - epsilon))) - 1
        # Too few points to reach the requested quantile: widen to the maximum
        # observed rather than silently returning an over-confident threshold.
        cal.quantiles[int(b)] = float(a[min(rank, n - 1)])
        cal.n_per_bin[int(b)] = int(n)
    return cal


def predict_sets(p: np.ndarray, k: np.ndarray, cal: ConformalCalibration) -> np.ndarray:
    """(n, 2) boolean array: does the set contain label 0, label 1?

    A label is admitted when its nonconformity does not exceed the bin's
    threshold, so both, one, or neither may be present. The empty set is not an
    error -- it means the model is confidently wrong about both options at this
    level, which is itself informative.
    """
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    bins = bin_of(k, cal.pool)
    q = np.array([cal.quantiles.get(int(b), 1.0) for b in bins])
    return np.stack([(p <= q), ((1.0 - p) <= q)], axis=1)


def coverage_report(
    sets: np.ndarray,
    y: np.ndarray,
    k: np.ndarray,
    cal: ConformalCalibration,
) -> pd.DataFrame:
    """Empirical coverage and set size per prefix length.

    `covered` is the number that matters: the share of rows whose set contains
    the true label. It should sit near 1 - epsilon, and where it does not, the
    exchangeability assumption is what failed.
    """
    y = np.asarray(y).astype(int)
    contains = sets[np.arange(len(y)), y]
    size = sets.sum(axis=1)

    df = pd.DataFrame({"k": np.asarray(k), "covered": contains, "size": size, "y": y})
    out = (
        df.groupby("k")
        .agg(
            n=("covered", "size"),
            coverage=("covered", "mean"),
            mean_set_size=("size", "mean"),
            singleton_rate=("size", lambda s: float((s == 1).mean())),
            abstain_rate=("size", lambda s: float((s == 2).mean())),
            empty_rate=("size", lambda s: float((s == 0).mean())),
        )
        .reset_index()
    )
    out["nominal"] = 1 - cal.epsilon
    out["gap"] = out["coverage"] - out["nominal"]
    out["bin"] = bin_of(out["k"].to_numpy(), cal.pool)
    return out
