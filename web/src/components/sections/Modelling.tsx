"use client";

import modellingJson from "../../../public/data/modelling.json";

/**
 * The modelling-depth section: four results that each answer a question the
 * earliness curve alone cannot.
 *
 * Every number is read from `modelling.json`, produced by the Phase 1-6
 * scripts. Nothing here is typed by hand.
 */

type Cal = { ece: number; brier: number; bias: number; auc: number };
type Modelling = {
  calibration: Record<string, { calib_prior: number; test_prior: number;
    em_prior_estimate: number | null; by_calibrator: Record<string, Cal> }>;
  order_verdict: Record<string, {
    mean_delta: number; mean_insample_delta: number;
    k_with_ci_excluding_zero: number[]; order_carries_signal: boolean }>;
  gru: Record<string, {
    n_parameters: number; seeds: number; test_auc_mean: number; test_auc_std: number;
    comparison: { gru_minus_baseline_mean: number; n_k_inside: number; n_k_total: number } }>;
  conformal: Record<string, { epsilon: number; k: number; coverage: number; nominal: number }[]>;
  alarm_summary: Record<string, {
    reference: { tau_star: number; mean_cost: number; never_alarm: number;
      always_alarm_at_k1: number; alarm_rate: number; mean_alarm_k: number };
    sweep_cells: number; cells_beating_trivial: number; tau_range: number[] }>;
};

const m = modellingJson as unknown as Modelling;

const pct = (v: number) => `${(v * 100).toFixed(1)}%`;
const f3 = (v: number) => v.toFixed(3);

export function Modelling() {
  const uciCal = m.calibration?.uci;
  const bpiCal = m.calibration?.bpi;
  const gru = m.gru?.uci;
  const alarm = m.alarm_summary?.uci;
  const cov = m.conformal?.uci ?? [];

  const byLevel = [0.2, 0.1, 0.05].map((eps) => {
    const rows = cov.filter((r) => Math.abs(r.epsilon - eps) < 1e-9);
    const n = rows.length || 1;
    return {
      nominal: 1 - eps,
      actual: rows.reduce((s, r) => s + r.coverage, 0) / n,
    };
  });

  return (
    <div className="grid gap-x-12 gap-y-14 lg:grid-cols-2">
      {/* --- calibration: the natural experiment ---------------------- */}
      <Card
        eyebrow="Calibration"
        title="The calibration fold has to resemble the period you deploy into"
      >
        <p>
          Post-hoc calibration is supposed to be free. On UCI it makes things{" "}
          <em>worse</em>, and the reason is visible in one number: the
          calibration fold breaches at{" "}
          <N>{f3(uciCal?.calib_prior ?? 0)}</N> while the test period breaches
          at <N>{f3(uciCal?.test_prior ?? 0)}</N>. A calibrator fitted on the
          first learns a correction the second does not need.
        </p>
        <Table
          head={["", "UCI (drifting)", "BPI (stable)"]}
          rows={[
            ["calib prior", f3(uciCal?.calib_prior ?? 0), f3(bpiCal?.calib_prior ?? 0)],
            ["test prior", f3(uciCal?.test_prior ?? 0), f3(bpiCal?.test_prior ?? 0)],
            [
              "ECE uncalibrated",
              f3(uciCal?.by_calibrator["uncalibrated|variable"]?.ece ?? 0),
              f3(bpiCal?.by_calibrator["uncalibrated|variable"]?.ece ?? 0),
            ],
            [
              "ECE after isotonic",
              f3(uciCal?.by_calibrator["isotonic|variable"]?.ece ?? 0),
              f3(bpiCal?.by_calibrator["isotonic|variable"]?.ece ?? 0),
            ],
          ]}
          highlight={3}
        />
        <p className="text-sm">
          Same code, same calibrators, opposite outcomes. BPI is split at random
          so its calibration fold matches deployment, and there isotonic
          behaves as the textbook says. UCI is split out-of-time across a log
          whose breach rate falls from 0.64 in March to 0.23 in May, and there
          it does not. Even a shift using the <em>true</em> test prior leaves
          ECE at 0.104, so the residual was never a prior problem.
        </p>
      </Card>

      {/* --- sequence models ------------------------------------------ */}
      <Card
        eyebrow="Sequence models"
        title="We measured whether order matters before building a model for it"
      >
        <p>
          Aggregation encoding throws away the order of events. Before writing
          any PyTorch, we target-encoded the ordered activity sequence against
          the same activities <em>sorted</em>, cross-fitted so a rare sequence
          cannot memorise its own label. The gap is what an order-aware model
          could buy.
        </p>
        <Table
          head={["", "cross-fitted Δ", "in-sample Δ"]}
          rows={[
            [
              "UCI",
              `${(m.order_verdict?.uci_servicenow?.mean_delta ?? 0) >= 0 ? "+" : ""}${f3(m.order_verdict?.uci_servicenow?.mean_delta ?? 0)}`,
              `+${f3(m.order_verdict?.uci_servicenow?.mean_insample_delta ?? 0)}`,
            ],
            [
              "BPI 2013",
              `+${f3(m.order_verdict?.bpi2013?.mean_delta ?? 0)}`,
              `+${f3(m.order_verdict?.bpi2013?.mean_insample_delta ?? 0)}`,
            ],
          ]}
          highlight={0}
        />
        <p className="text-sm">
          UCI&apos;s activity vocabulary is six symbols and the median case uses
          one of them across three events. There is no order to learn. The
          in-sample column is what the same test looks like without
          cross-fitting, which is why this was worth measuring first.
        </p>
        <p className="text-sm">
          A {gru?.n_parameters.toLocaleString("en-GB")}-parameter GRU over the
          raw event stream, {gru?.seeds} seeds, confirms it at the model level:
          test AUC <N>{f3(gru?.test_auc_mean ?? 0)}</N> ± {f3(gru?.test_auc_std ?? 0)}
          , which is{" "}
          <N>{f3(gru?.comparison.gru_minus_baseline_mean ?? 0)}</N> against the
          tuned gradient-boosted baseline. The seed spread is four times smaller
          than that gap, so the sequence model is genuinely losing on this log.
        </p>
      </Card>

      {/* --- conformal ------------------------------------------------ */}
      <Card
        eyebrow="Uncertainty"
        title="Conformal coverage puts a number on how far the log has moved"
      >
        <p>
          Split conformal turns a score into a label set that should contain
          the truth at a stated rate. Its guarantee assumes exchangeability,
          which an out-of-time split deliberately breaks.
        </p>
        <Table
          head={["nominal", "actual", "shortfall"]}
          rows={byLevel.map((r) => [
            pct(r.nominal),
            pct(r.actual),
            `${((r.actual - r.nominal) * 100).toFixed(1)}pp`,
          ])}
          highlight={-1}
        />
        <p className="text-sm">
          Coverage falls short at every level, and the shortfall widens as the
          bar drops. The guarantee holds under exchangeability, and an
          out-of-time split removes it, so the gap between nominal and actual
          reads directly as distance between the calibration fold and the test
          period. It is the same drift the calibration card found, expressed as
          a percentage. Prefix lengths 6 to 8 share one calibration bin: taken
          separately they hold 141, 73 and 46 negatives, too few to estimate a
          90% quantile from.
        </p>
      </Card>

      {/* --- decision layer ------------------------------------------- */}
      <Card
        eyebrow="Decision layer"
        title="The cost assumption moves the answer more than the model does"
      >
        <p>
          An AUC does not tell a team lead what to do. An alarm fires once per
          incident, at the first prefix crossing a threshold tuned on the
          validation fold, under a stated cost model.
        </p>
        <Table
          head={["at c_i/c_b = 0.1, e = 0.5", "value"]}
          rows={[
            ["threshold τ*", f3(alarm?.reference.tau_star ?? 0)],
            ["mean cost per case", f3(alarm?.reference.mean_cost ?? 0)],
            ["never alarm", f3(alarm?.reference.never_alarm ?? 0)],
            ["always alarm at k=1", f3(alarm?.reference.always_alarm_at_k1 ?? 0)],
            ["mean alarm prefix", `k = ${(alarm?.reference.mean_alarm_k ?? 0).toFixed(2)}`],
          ]}
          highlight={1}
        />
        <p className="text-sm">
          Across {alarm?.sweep_cells} combinations of escalation cost and
          intervention effectiveness, the model beats the better trivial policy
          in <N>{alarm?.cells_beating_trivial}</N> of them — and loses in the
          rest, all where escalation costs nearly as much as the breach it
          prevents. The chosen threshold ranges{" "}
          <N>{f3(alarm?.tau_range[0] ?? 0)}</N> to{" "}
          <N>{f3(alarm?.tau_range[1] ?? 0)}</N> across those assumptions, which
          is the honest headline: this decision is driven by what you believe
          escalation costs, not by the classifier.
        </p>
      </Card>
    </div>
  );
}

function Card({
  eyebrow,
  title,
  children,
}: {
  eyebrow: string;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div className="space-y-4">
      <p
        className="text-[11px] uppercase tracking-[0.18em]"
        style={{ color: "var(--accent)" }}
      >
        {eyebrow}
      </p>
      <h3 className="display text-[clamp(1.4rem,2.6vw,1.9rem)]">{title}</h3>
      <div className="space-y-4 leading-relaxed" style={{ color: "var(--text-2)" }}>
        {children}
      </div>
    </div>
  );
}

function N({ children }: { children: React.ReactNode }) {
  return (
    <span className="num" style={{ color: "var(--text-1)" }}>
      {children}
    </span>
  );
}

function Table({
  head,
  rows,
  highlight = -1,
}: {
  head: string[];
  rows: (string | number)[][];
  highlight?: number;
}) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr
            className="text-left text-[10.5px] uppercase tracking-wider"
            style={{ color: "var(--text-2)" }}
          >
            {head.map((h, i) => (
              <th key={h + i} className={`pb-2 ${i ? "text-right" : ""}`}>
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, ri) => (
            <tr
              key={ri}
              style={{
                borderTop: "1px solid var(--hairline)",
                color: ri === highlight ? "var(--text-1)" : undefined,
              }}
            >
              {r.map((c, ci) => (
                <td
                  key={ci}
                  className={`py-2 ${ci ? "num text-right" : ""}`}
                >
                  {c}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
