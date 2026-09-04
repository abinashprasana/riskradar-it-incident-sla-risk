# Model card — KAIROS

Prefix-based SLA-breach risk scoring for IT incident management, evaluated as
outcome-oriented predictive process monitoring.

Generated from `artifacts/`. Regenerate with:

```bash
python scripts/run_all.py     # every arm below, in order
```

or individually:

```bash
python scripts/run_leaky_baseline.py
python scripts/run_experiment.py --log uci_servicenow
python scripts/run_experiment.py --log uci_servicenow --split-mode random_case --out artifacts/uci_servicenow/agg_randomsplit
python scripts/run_experiment.py --log uci_servicenow --learner logreg --out artifacts/uci_servicenow/agg_logreg
python scripts/run_experiment.py --log uci_servicenow --encoding index
python scripts/run_experiment.py --log bpi2013
python scripts/run_experiment.py --log bpi2013 --learner logreg --out artifacts/bpi2013/agg_logreg
python scripts/run_experiment.py --log bpi2013 --encoding index
python scripts/run_experiment.py --log uci_servicenow --imbalance none --out artifacts/uci_servicenow/agg_noweight
python scripts/make_figures.py
```

---

## What the model does

Given the first `k` events of an incident, it estimates the probability that
the incident will breach its SLA. One row per `(case, k)`; every feature is
computed from events `[0:k]` alone.

**Intended use.** Research and method demonstration. Ranking an open queue by
predicted breach risk to prioritise human review.

**Not for.** Automated action without a human in the loop; individual
performance assessment of the staff named in `resource` / owner fields; any
deployment outside the organisation whose log it was fitted on.

---

## Headline results

| Log | Arm | ROC-AUC |
|:--|:--|--:|
| UCI | **As shipped**: whole-trace features, random split, model selected on test | **0.9674** |
| UCI | Prefix-based, out-of-time, one decision per ticket at k=1 | 0.7482 |
| UCI | Prefix-based, out-of-time, fixed cohort at k=1 | 0.6185 |
| UCI | Prefix-based, out-of-time, fixed cohort at k=8 | 0.8874 |
| UCI | Same as above but random case split (ablation) | 0.7437 |
| BPI 2013 | Prefix-based, random split, fixed cohort at k=1 | 0.9124 |
| BPI 2013 | Prefix-based, random split, fixed cohort at k=12 | 0.8983 |
| UCI | Encoding control: aggregation + logreg, k=8 | 0.8703 |
| UCI | Encoding control: index + logreg, k=8 | 0.8282 |
| BPI 2013 | Encoding control: aggregation + logreg, k=12 | 0.8599 |
| BPI 2013 | Encoding control: index + logreg, k=12 | 0.7291 |

The 0.9674 is reproduced exactly from the untouched original modules by
`scripts/run_leaky_baseline.py`, so the comparison is against the real prior
result and not a strawman. **The gap between row 1 and row 3 is the finding.**

Two ways of reading the earliness curve are reported, and only one answers the
question:

- **Variable cohort** — all prefixes available at each `k`. In the UCI log the
  base rate climbs 0.232 → 0.730 across k=1..8 as short cases exhaust and the
  survivors are the breach-prone ones, so this curve rises even for a model
  that learned nothing.
- **Fixed cohort** — only cases long enough to reach `k_max`, so the population
  is identical at every `k` and a change in AUC is a change in what the model
  knows. **This is the earliness curve.**

---

## Data

| | UCI ServiceNow | BPI Challenge 2013 (Volvo IT) |
|:--|:--|:--|
| Source | UCI ML Repository, DOI 10.24432/C57S4H | 4TU.ResearchData, DOI 10.4121/500573e6-accc-4b0c-9576-aa5468b10cee |
| Cases / events | 24,918 / 141,712 | 7,554 / 65,533 |
| Cases used | 23,102 | 7,545 |
| Label | native `made_sla` flag | constructed: duration > 240h |
| Base rate | 0.382 | 0.310 |
| Split | out-of-time, straddlers dropped | random at case level (see below) |
| k_max | 8 | 12 |

### The UCI log is a snapshot export, not an event stream

This is the most important property of the data and it is not documented
upstream. Closure fields are back-filled onto every row of a trace. Measured at
the **first** event of each case:

| Column | Populated at k=1 | Equals the case-final value |
|:--|--:|--:|
| `closed_at` | 100.0% | 100% |
| `closed_code` | 99.6% | 100% |
| `resolved_by` | 99.6% | 100% |
| `resolved_at` | 93.8% | 100% |

Knowing `closed_code` at event 1 splits the breach rate from 0.000 (code 12) to
0.814 (code 15). Any prefix builder that reads "the row at event k" imports the
case's ending at k=1.

**This is why features are whitelisted, not blacklisted.** A model may only see
columns an adapter explicitly promoted (`riskradar/logspec.py:feature_columns`).
`leakage_screen` is the secondary net, and its limits are stated honestly: for
datetime columns it proves leakage mechanically (`closed_at` refers to a moment
after the observation point in 99.96% of cases), but for non-temporal fields
like `closed_code` no statistic distinguishes "fixed at creation" from
"back-filled at closure" — `category` and `closed_at` are both 100% constant
within a case. Only knowing what the field means separates them.

### Selection bias from terminal-event removal

`Resolved` and `Closed` events determine the outcome and are dropped as rows.
1,816 UCI cases (7.29%) consist solely of such events and leave the study.
They are **not** a random subset: breach rate 0.163 against 0.382 for those
retained. Results therefore describe incidents with at least one non-terminal
event.

### Concept drift in the UCI log

Monthly breach rate: **March 2016 0.642, April 0.243, May 0.234.** A random
split lets a model exploit this; `opened_at` alone separates the outcome at AUC
0.708 under in-sample target encoding. The out-of-time split removes that free
accuracy, which is most of why the honest number is lower.

The dataset is effectively three months, not the year its date range suggests:
99.3% of cases start between March and May 2016, and the remaining 274 spread
across the following nine months. Experiments restrict to that dense window.

### BPI 2013 cannot support an out-of-time split

7,483 of 7,545 cases *start* inside a ten-day window (middle 50% between
2012-04-27 and 2012-05-03) while running a median 7.5 days. Every chronological
boundary is straddled by almost every case: straddle removal discards 4,081 of
5,658 modelling cases and leaves a training set of sub-hour incidents, on which
a fitted duration percentile collapses to 0.6h and labels 82% of the test set
as breached.

BPI 2013 therefore uses a **random case-level split**, which still prevents a
case's prefixes from spanning folds but makes no out-of-time claim. Its numbers
are compared against UCI *under the same random protocol*, never against UCI's
temporal result. This is a property of the dataset, and the ablation quantifies
what the protocol difference is worth: on UCI, random splitting inflates
fixed-cohort AUC by **+0.125 at k=1** and **+0.049 at k=8**.

### The BPI label is a modelling choice

There is no SLA field. A case breaches if its duration exceeds **240 hours**, a
stated constant chosen because it yields a base rate (0.310) near UCI's (0.366)
so a cross-log comparison is not mostly a comparison of prevalence. It is not a
fitted quantile — see above for why that fails. **Do not present this label as
if the dataset shipped one.**

Prefixes whose elapsed time already exceeds the deadline have a determined
outcome and are excluded from the primary curve (3.7% of test prefixes);
including them would inflate AUC toward 1.0 at high `k`.

---

## Features

Whitelisted by prefix: `static__` (case-invariant, read at event 0),
`dyn_cat__` / `dyn_num__` (value as of event k-1), `num__` (running aggregates
over events `[0:k]`). 26 numeric and 16 categorical columns for UCI.

Process-generic features work on both logs: distinct activities and resources
so far, resource changes (the handover proxy, and the cross-log analogue of
`reassignment_count`), activity repeats, waiting events, inter-event gaps,
elapsed time, and trailing arrival counts.

**Excluded and why:**

- `made_sla` (the label), `active` (1:1 with `Closed`)
- `closed_at`, `closed_code`, `resolved_by`, `resolved_at` (back-filled)
- `resolution_hours`, `total_events`, all `*_max` / `*_mean` whole-trace
  aggregates — only as-of-k versions are legal
- `problem_id`, `rfc`, `vendor`, `caused_by` — 98%+ missing and typically
  linked post-hoc
- BPI `SR Latest Impact` — 100% constant within a case, so provenance is
  unverifiable, and impact medians run 178–205h against a global 181h, so it
  carries almost no signal. No upside, unverifiable risk.
- **Open-workload counts** were considered and rejected: counting concurrently
  open cases requires other cases' end times, which are future information.
  Trailing arrival counts (`arrivals_24h`, `arrivals_7d`) are past-only.

### One feature that needs interrogation before it is trusted

`num__elapsed_h` at k=1 has a median of 0.033 h yet scores AUC 0.636 alone. A
two-minute latency predicting a breach days away is almost certainly encoding
creation channel or automation rather than urgency — auto-created tickets
update instantly. It is legally as-of-k so it stays, but it will dominate the
k=1 model, and it should be cross-tabulated against `contact_type` and
`sys_created_by` before anyone reads it as a causal signal.

---

## Evaluation protocol: four folds

| Fold | Share | Cases | Job |
|:--|--:|--:|:--|
| train | 0.50 | 9,602 | fit model parameters |
| val | 0.15 | 2,881 | model selection, hyperparameters, early stopping |
| calib | 0.10 | 1,920 | calibrators and conformal quantiles only |
| test | 0.25 | 5,707 | scored once, at the end |

`val` and `calib` are deliberately separate. Selecting a model and then
calibrating it on the same held-out rows makes the reported calibration
optimistic for the same reason selecting on test does — the fold has already
been used to make a choice.

The three modelling fractions sum to 0.75, exactly as the previous three-fold
default did, so the test boundary did not move: the test fold is byte-identical
at 5,707 cases and 17,118 prefix rows, and the straddler count is unchanged at
2,718. Adding the fourth fold re-partitioned the modelling pool and invalidated
no previously reported test number. `tests/test_modelling_infra.py` asserts this.

Every per-k AUC now carries a 95% percentile bootstrap interval (1,000
resamples). This matters more than it sounds: the fixed cohort is 478 test
cases, and at k=8 the interval is **0.887 [0.858, 0.915]**. Several model
comparisons in this card are smaller than that width and should be read as
ties.

### Class weighting: no effect on ranking, a real cost to calibration

Control arm, XGBoost with `scale_pos_weight = n_neg/n_pos` against no weighting,
UCI fixed cohort:

| | mean \|ΔAUC\| | mean ΔBrier | mean ΔECE |
|:--|--:|--:|--:|
| weighted − unweighted | 0.0016 | **+0.0144** | **+0.0198** |

AUC confidence intervals overlap at every k. Weighting moves discrimination
essentially not at all — AUC is invariant to a monotone rescaling of scores —
while making calibration measurably **worse**, because it pushes predictions up
on a cohort the model already over-predicts relative to its own base rate.

Weighting is retained as the default for continuity with the published curve;
the evidence says a calibration-focused configuration should turn it off, and
that decision belongs with the calibration work rather than being made here.

Note the first version of this arm reported a difference of exactly zero at
every k. That was a defect, not a result: `class_weight` is a scikit-learn
parameter that XGBoost ignores entirely, so toggling it changed nothing for the
learner that actually gets selected. `tests/test_modelling_infra.py` now guards
against it recurring.

---

## Method

- **Prefix log**: one row per `(case, k)`, `k = 1..min(n_nonterm, k_max)`.
- **Split**: cases ordered by start time; a case may never appear in two folds;
  no case in the modelling period may still be running when the test period
  begins (Weytjens & De Weerdt, BPM 2021 Workshops, LNBIP 436).
- **Calibration fold**: drawn at random from the modelling period, not as its
  temporal tail. Surviving to the test boundary requires being short, and short
  cases are non-breaching, so a temporal-tail calib had base rate 0.124 against
  a train 0.483. Model choice on that fold would be choice on an artifact.
  The cost, stated plainly: calib resembles train rather than test, so anything
  tuned on it transfers imperfectly across the observed drift.
- **Encoding**: aggregation encoding, single bucket — one model serves all `k`.
  Index encoding with prefix-length bucketing was also built and compared; see
  below.
- **Models**: logistic regression, random forest, XGBoost. Selected on the
  calibration fold. Test is scored exactly once.
- **Bounded one-hot** (`min_frequency=30`, `max_categories=50`): the log has
  thousands of `caller_id` values, and unbounded encoding is what inflated the
  original artifact to 56 MB.

### Encoding comparison

Both Teinemaa et al. combinations were built: aggregation encoding with a single
bucket, and index encoding with prefix-length bucketing (one model per `k`,
every event position given its own columns, so ordering survives).

Learner held fixed at logistic regression, fixed-cohort AUC:

| k | UCI agg | UCI index | | k | BPI agg | BPI index |
|--:|--:|--:|---|--:|--:|--:|
| 1 | 0.589 | 0.592 | | 1 | 0.860 | 0.594 |
| 4 | 0.710 | 0.682 | | 6 | 0.877 | 0.690 |
| 8 | 0.870 | 0.828 | | 12 | 0.860 | 0.729 |

**On UCI the two encodings track each other closely until k=8, where
aggregation pulls ahead by 0.038; on BPI index encoding loses heavily at every
k.** The asymmetry has a concrete cause rather than being
a mystery: this is *pure* index encoding — static attributes plus per-position
values — so it carries none of the running aggregates (`n_distinct_resources`,
gap statistics, elapsed time, arrival counts). UCI has 13 static case
attributes that both arms share, so dropping the aggregates costs little. BPI
has two, so the aggregates are most of its signal and losing them is fatal.

Aggregation wins at 11 of the 12 prefix lengths measured across both logs; the
one exception is UCI k=6, by 0.002. The size of the win is what differs. On UCI
the encoding margin (0.001-0.038) is the same order as switching learner
(0.013-0.042 for XGBoost over logreg), so neither choice dominates and both are
small. On BPI the encoding margin is 0.15-0.30 against a learner effect of
0.02-0.05, so the encoding decides the outcome.

This replicates the benchmark finding that aggregation encoding with a single
bucket is a strong baseline that more elaborate schemes do not reliably beat,
and it is the reason the headline arm uses it.

Index buckets are fitted with L2 logistic regression only, not model-selected
per bucket: width grows with `k` while the bucket's training set shrinks (UCI
k=8 has 1,541 training cases against 89 raw columns before one-hot), so a
per-bucket search would mostly select calibration-fold noise.

### Known data-quality handling

| Issue | Handling |
|:--|:--|
| `?` as missing token | parsed as NaN at load; previously survived as a real one-hot level |
| 5 events with `incident_state == '-100'` | dropped, counted in `run_config.json` |
| 5 events with first `sys_updated_at` up to 2.5h before `opened_at` | elapsed clipped to 0, count recorded |
| 8 BPI cases still open at log end | dropped (the only genuine censoring in either log) |
| Missing categoricals | filled with an explicit `__MISSING__` level, not the mode |

---

## Limitations

1. **Single organisation per log.** Nothing here transfers to another service
   desk without refitting.
2. **The fixed cohort is small.** 478 UCI test cases reach k=8, so the
   right-hand end of the earliness curve is noisier than the left.
3. **Calibration is poor at low k** (Brier 0.34 at fixed-cohort k=1). The
   scores rank better than they calibrate; use them for ordering a queue, not
   as literal probabilities, unless recalibrated.
4. **No cost model.** A threshold that is right for a team with spare capacity
   is wrong for one without. The alarm/cost layer is deliberately out of scope.
5. **Both logs are historical and fully closed.** Nothing here has been tested
   against a live queue, and the drift observed within three months suggests
   monitoring would be required in any real deployment.
6. **Survival analysis was considered and rejected**, not omitted. Every UCI
   incident reaches `Closed` and `closed_at` has zero nulls; the 1,556 missing
   `resolved_at` values are a data-quality artefact of a back-filled field, not
   right-censoring. Censoring requires cases in flight at the observation
   cutoff, and there are none.

## What these numbers may not be used to claim

- Not a benchmark win. No comparison against a published result on either log
  is made, and the honest numbers are **lower** than the original 0.9674 by
  design.
- BPI 2013 results are **not** out-of-time and are not comparable to UCI's
  temporal result.
- The BPI label is constructed, not observed.

---

## Governance

The intended use — ranking an internal IT queue for human review — would not
normally fall within EU AI Act Annex III high-risk categories, which for
critical infrastructure require the system to be a safety component in the
management or operation of that infrastructure. That assessment is stated here
so it can be checked, not asserted as a clearance, and it has not been reviewed
by anyone qualified to give a legal opinion.

Practices adopted regardless: this card; whitelist-based feature governance with
an automated leakage screen; recorded provenance (`run_config.json` captures
interpreter and library versions, dropped-row counts, and split configuration);
a test suite asserting fold disjointness and the absence of forbidden columns;
and human-in-the-loop framing throughout.

**Not yet done:** decision logging, drift monitoring, and any review of the
fairness implications of features naming individual staff (`opened_by`,
`resource`, owner fields). Those fields improve accuracy and carry obvious
potential for misuse as performance measures; the "not for" line above is a
statement of intent, not an enforced control.

## Reproducibility

Deterministic given the seed (default 42). Dependencies pinned in
`pyproject.toml`. Results generated on Python 3.13.3 with scikit-learn 1.7.1,
pandas 2.3.2, numpy 2.3.2, xgboost 3.1.1. `runtime.txt` pins 3.11 for the
deferred Streamlit deployment; no pickled model is shipped, so results move
between interpreters as CSV and JSON and no cross-version unpickling arises.

## Status of the Streamlit app

`app.py` is **untouched by this work** and still reflects the old pipeline: it
retrains a leaky model at startup and Tab 4 reports in-sample metrics over data
the model was fitted on. The numbers in this card do not describe what that app
displays. Bringing the UI onto these artifacts is deferred.
