"""Probability calibration, and the label shift that defeats it here.

The model's scores rank well and mean little. On the variable cohort it
over-predicts breach at every prefix length; on the fixed cohort it
*under*-predicts by 37 points at k=1. Both are real, they have different
causes, and only one of them is fixable by a calibrator.

**The fixable part.** Fold base rates run train 0.563, val 0.576, calib 0.583,
test 0.396. That is label shift: P(y) moves between fitting and deployment
while P(x|y) largely does not. A calibrator fitted on calib targets 0.58 and
therefore cannot, even in principle, remove a bias that exists because test is
0.40 -- so Platt and isotonic are expected to leave ECE roughly where they
found it. `prior_shift_offset` addresses the actual cause by re-anchoring the
prior, with the new prior estimated by EM from the *unlabelled* test scores
(Saerens, Latinne & Decaestecker, Neural Computation 14(1), 2002). No test
label is touched, so this stays legitimate.

**The unfixable part.** The fixed cohort holds cases that ran to eight or more
events; those breach at 0.73 against a population 0.40. At k=1 nothing in the
feature vector says this case will turn out long, so the model predicts near
the population rate and is "miscalibrated" against a cohort defined by its own
future. Conditioning on trace length would be conditioning on what has not
happened yet. That gap is a floor, not a defect, and it closes on its own as k
rises and length becomes observable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


@dataclass
class Calibrator:
    """A fitted probability map, plus the name of what it is."""

    kind: str
    transform: object

    def __call__(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        if self.kind == "identity":
            return p
        if self.kind == "platt":
            return self.transform.predict_proba(_logit(p).reshape(-1, 1))[:, 1]
        if self.kind == "isotonic":
            return np.clip(self.transform.predict(p), 0.0, 1.0)
        if self.kind == "beta":
            a, b, c = self.transform
            lp = np.log(np.clip(p, EPS, 1 - EPS))
            lq = np.log(np.clip(1 - p, EPS, 1 - EPS))
            return _sigmoid(a * lp - b * lq + c)
        if self.kind == "prior_shift":
            return _sigmoid(_logit(p) + float(self.transform))
        raise ValueError(f"unknown calibrator {self.kind!r}")


def fit_platt(p: np.ndarray, y: np.ndarray) -> Calibrator:
    """Logistic regression on the score's logit. One slope, one intercept."""
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
    lr.fit(_logit(p).reshape(-1, 1), np.asarray(y))
    return Calibrator("platt", lr)


def fit_isotonic(p: np.ndarray, y: np.ndarray) -> Calibrator:
    """Non-parametric and monotone. Flexible, and prone to overfit a small fold."""
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(np.asarray(p, dtype=float), np.asarray(y, dtype=float))
    return Calibrator("isotonic", iso)


def fit_beta(p: np.ndarray, y: np.ndarray) -> Calibrator:
    """Beta calibration (Kull, Silva Filho & Flach, AISTATS 2017).

    Two-parameter family that, unlike Platt, can represent a map which is not
    symmetric about 0.5 -- the shape actually needed when a model is wrong in
    the middle of its range but right at the extremes, which is this one.
    """
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    X = np.column_stack([np.log(p), -np.log(1 - p)])
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
    lr.fit(X, np.asarray(y))
    a, b = lr.coef_[0]
    return Calibrator("beta", (float(a), float(b), float(lr.intercept_[0])))


def estimate_prior_em(
    p_source: np.ndarray,
    prior_source: float,
    max_iter: int = 200,
    tol: float = 1e-8,
) -> float:
    """EM estimate of the target prior from unlabelled target scores.

    Saerens-Latinne-Decaestecker. Takes scores the *source*-calibrated model
    assigns to target inputs and iterates: re-weight by the current prior
    guess, re-estimate the prior as the mean posterior, repeat. Uses no target
    labels, which is what makes it usable before the outcomes are known.
    """
    p = np.clip(np.asarray(p_source, dtype=float), EPS, 1 - EPS)
    prior_source = float(np.clip(prior_source, EPS, 1 - EPS))
    prior = prior_source

    for _ in range(max_iter):
        r1 = (prior / prior_source) * p
        r0 = ((1 - prior) / (1 - prior_source)) * (1 - p)
        post = r1 / (r1 + r0)
        new = float(post.mean())
        if abs(new - prior) < tol:
            prior = new
            break
        prior = new
    return float(np.clip(prior, EPS, 1 - EPS))


def prior_shift_offset(prior_target: float, prior_source: float) -> Calibrator:
    """Additive logit shift that moves a model's prior from source to target.

    log(pi_t / (1 - pi_t)) - log(pi_s / (1 - pi_s)). Rank-preserving, so AUC is
    untouched by construction and only calibration can move.
    """
    pt = float(np.clip(prior_target, EPS, 1 - EPS))
    ps = float(np.clip(prior_source, EPS, 1 - EPS))
    return Calibrator("prior_shift", np.log(pt / (1 - pt)) - np.log(ps / (1 - ps)))


def fit_all(
    p_calib: np.ndarray,
    y_calib: np.ndarray,
    p_target: np.ndarray | None = None,
) -> dict[str, Calibrator]:
    """Every calibrator this project compares, fitted on the calibration fold.

    `p_target` is optional and unlabelled; supplying it enables the EM
    prior-shift arm, which is the only one of these that can respond to a base
    rate the calibration fold never saw.
    """
    out: dict[str, Calibrator] = {
        "uncalibrated": Calibrator("identity", None),
        "platt": fit_platt(p_calib, y_calib),
        "isotonic": fit_isotonic(p_calib, y_calib),
        "beta": fit_beta(p_calib, y_calib),
    }
    if p_target is not None:
        prior_src = float(np.mean(y_calib))
        prior_tgt = estimate_prior_em(np.asarray(p_target), prior_src)
        out["prior_shift_em"] = prior_shift_offset(prior_tgt, prior_src)
        out["prior_shift_em"].estimated_prior = prior_tgt  # type: ignore[attr-defined]
        out["prior_shift_em"].source_prior = prior_src  # type: ignore[attr-defined]

        # Platt first to fix the shape, then the offset to fix the level. The
        # two failures are separable and neither map alone addresses both.
        platt = out["platt"]
        shift = out["prior_shift_em"]
        composed = Calibrator("composed", None)
        composed.__dict__["_fns"] = (platt, shift)
        composed.__class__ = _Composed
        out["platt_then_prior_shift"] = composed
    return out


class _Composed(Calibrator):
    """Two calibrators applied in order."""

    def __call__(self, p: np.ndarray) -> np.ndarray:
        first, second = self.__dict__["_fns"]
        return second(first(p))
