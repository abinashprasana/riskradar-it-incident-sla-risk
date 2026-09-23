"use client";

import summaryJson from "../../../public/data/summary.json";
import type { Summary } from "@/lib/data";

/**
 * Closing section: how the thing was engineered, and what it does not claim.
 *
 * This carries the evaluation rigour that earlier drafts spent two whole
 * sections on. Presented as standard practice, which is what it is.
 */

const s = summaryJson as unknown as Summary;

const CRAFT = [
  {
    h: "Features are whitelisted",
    p: "A model sees a column only when an adapter has handed it over on purpose. The leaks in this log are not the obvious ones: the close code is already filled in at the first event, and on its own it moves the breach rate from 0.25 to 0.64. Closure fields, resolution times and whole-trace totals never reach it, because none of them exist at the moment you are asking.",
  },
  {
    h: "Evaluation runs forward in time",
    p: "Training cases all close before the test period opens. Incidents still running when it starts are dropped, so nothing is scored on a week the model already lived through.",
  },
  {
    h: "Four folds, one job each",
    p: "Parameters fit on train, choices made on validation, calibrators fit on their own held-out slice, and test read once at the very end.",
  },
  {
    h: "Every number has an interval",
    p: "A thousand-resample bootstrap on each prefix length. At k=8 the cohort is 478 incidents, and several comparisons on this page are narrower than their own error bars. Those are reported as ties.",
  },
  {
    h: "Checked against a second log",
    p: "The method is checked against BPI Challenge 2013 from Volvo IT, where the arrival pattern is different enough to break parts of it. Where it breaks, that is written down.",
  },
  {
    h: "It reproduces",
    p: "77 tests, and a single command regenerates every figure here from the raw event logs.",
  },
];

export function HowItsBuilt() {
  return (
    <div className="grid gap-12 lg:grid-cols-[1fr_1.15fr]">
      <div>
        <p
          className="mb-4 text-[11px] uppercase tracking-[0.18em]"
          style={{ color: "var(--accent)" }}
        >
          How it is built
        </p>
        <h2 className="display text-[clamp(1.9rem,4.4vw,3rem)]">
          The protocol is the product
        </h2>
        <p className="mt-5 leading-relaxed" style={{ color: "var(--text-2)" }}>
          A risk score is worth exactly as much as the evaluation behind it.
          An incident log is an unforgiving place to get that right: its columns
          are written and rewritten as a case moves, and its arrival pattern
          shifts week to week. The protocol below was built around both facts
          before any model was fitted.
        </p>

        <dl className="mt-9 grid grid-cols-2 gap-x-6 gap-y-6">
          <Stat label="Incidents" value={s.uci.n_cases_total.toLocaleString("en-GB")} />
          <Stat label="Held-out test" value={s.uci.split.n_test_cases.toLocaleString("en-GB")} />
          <Stat label="Tests" value="77" />
          <Stat label="Logs" value="2" />
        </dl>

        <div className="mt-9 flex flex-wrap gap-3">
          <a
            href="https://github.com/abinashprasana/riskradar-it-incident-sla-risk"
            className="rounded-lg px-5 py-3 text-sm font-medium transition-transform duration-150 hover:-translate-y-0.5"
            style={{ background: "var(--accent)", color: "var(--accent-ink)" }}
          >
            Read the code
          </a>
          <a
            href="https://github.com/abinashprasana/riskradar-it-incident-sla-risk/blob/main/docs/MODEL_CARD.md"
            className="rounded-lg px-5 py-3 text-sm font-medium transition-colors duration-150"
            style={{ border: "1px solid var(--hairline)", color: "var(--text-1)" }}
          >
            Model card
          </a>
        </div>
      </div>

      <div className="space-y-7">
        {CRAFT.map((c, i) => (
          <div
            key={c.h}
            className="grid grid-cols-[auto_1fr] gap-4"
            style={{ borderTop: i ? "1px solid var(--hairline)" : undefined, paddingTop: i ? 24 : 0 }}
          >
            <span
              className="num mt-[3px] text-xs"
              style={{ color: "var(--accent)" }}
            >
              {String(i + 1).padStart(2, "0")}
            </span>
            <div>
              <h3 className="text-[0.95rem] font-medium">{c.h}</h3>
              <p className="mt-1.5 text-sm leading-relaxed" style={{ color: "var(--text-2)" }}>
                {c.p}
              </p>
            </div>
          </div>
        ))}

        <div
          className="rounded-lg p-4 text-sm leading-relaxed"
          style={{
            background: "var(--surface-2)",
            borderLeft: "3px solid var(--hairline)",
            color: "var(--text-2)",
          }}
        >
          <strong style={{ color: "var(--text-1)" }}>What this does not claim.</strong>{" "}
          One organisation per log, so nothing transfers to another service desk
          without refitting. Scores below k=4 rank better than they calibrate.
          The BPI figures use a random split and sit on a constructed deadline,
          so they answer a narrower question than the UCI ones.
        </div>
      </div>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="text-[11px] uppercase tracking-wider" style={{ color: "var(--text-2)" }}>
        {label}
      </dt>
      <dd className="num mt-1 text-2xl">{value}</dd>
    </div>
  );
}
