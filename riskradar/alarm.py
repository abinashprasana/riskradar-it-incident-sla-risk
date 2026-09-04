"""Cost-sensitive alarms: when to escalate, given what escalation costs.

An AUC does not tell a team lead what to do. A threshold does, but only once
someone says what a false alarm costs relative to a missed breach. This turns
the probability into a decision under an explicit, stated cost model, following
the alarm-based framing of Fahrenkrog-Petersen, Tax, Teinemaa, Dumas, de Leoni,
Maggi & Weidlich, "Fire Now, Fire Later", Knowledge and Information Systems
64(2), 559-587, 2022.

The alarm fires **once per case**, at the first prefix where the score crosses
the threshold, and monitoring then stops. That matters: a per-prefix threshold
would let one incident raise eight alarms, which is not how a queue works.

    never fires, no breach   0
    never fires, breach      c_breach
    fires, no breach         c_intervene
    fires, breach            c_intervene + (1 - effectiveness) * c_breach

`effectiveness` is the share of breaches an early escalation actually averts.
It is a stated assumption, not a measured quantity -- this log records no
interventions, so nothing here can estimate it. The sweep exists precisely
because that number is unknown.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CostModel:
    c_intervene: float = 1.0
    c_breach: float = 10.0
    effectiveness: float = 0.5

    @property
    def ratio(self) -> float:
        return self.c_intervene / self.c_breach

    def break_even_threshold(self) -> float:
        """Closed-form p* for a single decision: (c_i / c_b) / effectiveness.

        Escalating is worth it when c_i + (1-e)*c_b*p < c_b*p. This ignores the
        sequential structure -- the real alarm may wait for a later, better
        informed prefix -- so it is a sanity anchor for the grid search rather
        than a replacement for it.
        """
        if self.effectiveness <= 0:
            return float("inf")
        return self.ratio / self.effectiveness


def first_alarm(scored: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """One row per case: whether an alarm fired, and at which prefix.

    `scored` needs `case_id`, `k`, `y`, `p`.
    """
    df = scored.sort_values(["case_id", "k"])
    fired = df[df["p"] >= threshold].groupby("case_id").first()
    out = df.groupby("case_id").agg(y=("y", "first"), n_k=("k", "size"))
    out["fired_k"] = fired["k"].reindex(out.index)
    out["fired"] = out["fired_k"].notna()
    return out


def evaluate_alarm(scored: pd.DataFrame, threshold: float, cost: CostModel) -> dict:
    a = first_alarm(scored, threshold)
    y = a["y"].to_numpy().astype(bool)
    fired = a["fired"].to_numpy()

    per_case = np.where(
        fired,
        cost.c_intervene + np.where(y, (1 - cost.effectiveness) * cost.c_breach, 0.0),
        np.where(y, cost.c_breach, 0.0),
    )
    return {
        "threshold": float(threshold),
        "n_cases": int(len(a)),
        "alarm_rate": float(fired.mean()),
        "mean_cost": float(per_case.mean()),
        "mean_alarm_k": float(a.loc[a["fired"], "fired_k"].mean()) if fired.any() else float("nan"),
        "recall": float(fired[y].mean()) if y.any() else float("nan"),
        "precision": float(y[fired].mean()) if fired.any() else float("nan"),
    }


def baselines(scored: pd.DataFrame, cost: CostModel) -> dict:
    """The two trivial policies any threshold must beat to be worth having."""
    a = first_alarm(scored, threshold=2.0)  # unreachable, so nothing fires
    y = a["y"].to_numpy().astype(bool)
    prevalence = float(y.mean())
    return {
        "never_alarm": prevalence * cost.c_breach,
        "always_alarm_at_k1": cost.c_intervene
        + (1 - cost.effectiveness) * cost.c_breach * prevalence,
        "prevalence": prevalence,
    }


def optimise_threshold(
    scored: pd.DataFrame,
    cost: CostModel,
    grid: np.ndarray | None = None,
) -> tuple[float, pd.DataFrame]:
    """Grid-search the firing threshold that minimises expected cost per case.

    Must be run on the validation fold. Choosing it on test would make the
    reported saving a best-of-N, which is the error this project exists to
    avoid repeating.
    """
    grid = np.arange(0.02, 1.0, 0.02) if grid is None else grid
    rows = [evaluate_alarm(scored, t, cost) for t in grid]
    df = pd.DataFrame(rows)
    return float(df.loc[df["mean_cost"].idxmin(), "threshold"]), df


def sweep(
    scored_val: pd.DataFrame,
    scored_test: pd.DataFrame,
    ratios=(0.02, 0.05, 0.10, 0.20, 0.50, 1.00),
    effectivenesses=(0.2, 0.5, 0.8, 1.0),
) -> pd.DataFrame:
    """Tune on val and apply to test, across the cost assumptions.

    The point of the grid is honesty about what is assumed. If the recommended
    threshold swings wildly across plausible cost ratios, the decision layer is
    driven by the assumption rather than by the model, and that should be
    visible rather than hidden behind one chosen operating point.
    """
    rows = []
    for r in ratios:
        for e in effectivenesses:
            cost = CostModel(c_intervene=r * 10.0, c_breach=10.0, effectiveness=e)
            tau, _ = optimise_threshold(scored_val, cost)
            test_at_tau = evaluate_alarm(scored_test, tau, cost)
            base = baselines(scored_test, cost)
            best_trivial = min(base["never_alarm"], base["always_alarm_at_k1"])
            rows.append({
                "cost_ratio": r,
                "effectiveness": e,
                "tau_star": tau,
                "break_even_p": cost.break_even_threshold(),
                "test_mean_cost": test_at_tau["mean_cost"],
                "best_trivial_cost": best_trivial,
                "saving_vs_trivial": 1 - test_at_tau["mean_cost"] / best_trivial
                if best_trivial > 0 else float("nan"),
                "alarm_rate": test_at_tau["alarm_rate"],
                "mean_alarm_k": test_at_tau["mean_alarm_k"],
                "recall": test_at_tau["recall"],
                "precision": test_at_tau["precision"],
            })
    return pd.DataFrame(rows)
