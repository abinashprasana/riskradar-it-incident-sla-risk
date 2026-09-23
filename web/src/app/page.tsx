"use client";

import dynamic from "next/dynamic";
import { KairosLockup, KairosMark } from "@/components/brand/KairosMark";
import { EarlinessCurve } from "@/components/charts/EarlinessCurve";
import { Explorer } from "@/components/explorer/Explorer";
import { HowItsBuilt } from "@/components/sections/HowItsBuilt";
import { Modelling } from "@/components/sections/Modelling";
import { useInView, useScrollProgress } from "@/lib/useScrollProgress";
import { SERIES, SERIES_DASH, type ExplorerSample, type PerKData, type Summary } from "@/lib/data";

import summaryJson from "../../public/data/summary.json";
import perKJson from "../../public/data/per_k.json";
import sampleJson from "../../public/data/explorer_sample.json";

// The heaviest thing on the page, and it decorates rather than carries the
// argument, so it loads after everything that does.
const TraceField = dynamic(
  () => import("@/components/three/TraceField").then((m) => m.TraceField),
  { ssr: false },
);

const summary = summaryJson as unknown as Summary;
const perK = perKJson as unknown as PerKData;
const sample = sampleJson as unknown as ExplorerSample;

const fixed = (arm: keyof PerKData) => perK[arm].filter((p) => p.cohort === "fixed");
const variable = (arm: keyof PerKData) => perK[arm].filter((p) => p.cohort === "variable");

export default function Page() {
  const { ref: curveRef, progress: curveProgress } = useScrollProgress<HTMLDivElement>({
    start: 0.85,
    end: 0.35,
  });
  const { ref: fieldRef, progress: fieldProgress } = useScrollProgress<HTMLDivElement>({
    start: 0.9,
    end: 0.2,
  });

  return (
    <main>
      <Nav />

      {/* ── 1 · What it does ─────────────────────────────────────────── */}
      <section className="depth relative flex min-h-[92vh] items-center overflow-hidden">
        {/* bias=1 clears the field off the left, so the headline sits on a
            clean ground rather than on top of moving geometry. */}
        <div className="absolute inset-0">
          <TraceField resolve={0.72} bias={1} />
        </div>
        <div
          className="pointer-events-none absolute inset-0"
          style={{
            background:
              "linear-gradient(100deg, var(--ground) 24%, color-mix(in srgb, var(--ground) 58%, transparent) 54%, transparent 80%)",
          }}
        />

        <div className="relative mx-auto w-full max-w-6xl px-6">
          <Opening />
        </div>
      </section>

      {/* ── 2 · The question ─────────────────────────────────────────── */}
      <section id="curve" className="scroll-mt-20">
        <Section tint>
          <Eyebrow>The question</Eyebrow>
          <h2 className="display max-w-3xl text-[clamp(1.9rem,4.4vw,3rem)]">
            How early can you actually call it?
          </h2>
          <p className="mt-5 max-w-2xl leading-relaxed" style={{ color: "var(--text-2)" }}>
            Accuracy changes with how much of an incident you have seen. One
            event in, the model has who opened the ticket and what they said it
            was about. Eight events in, it has watched the thing struggle. The
            curve between those two points is the answer.
          </p>

          <div ref={curveRef} className="mt-10 grid gap-10 lg:grid-cols-[1.5fr_1fr] lg:items-start">
            <div
              className="rounded-xl p-6"
              style={{ background: "var(--surface-1)", border: "1px solid var(--hairline)" }}
            >
              <EarlinessCurve
                progress={curveProgress}
                yMin={0.5}
                yMax={0.95}
                showBand
                readHead
                series={[
                  {
                    id: "fixed",
                    label: "same incidents throughout",
                    color: SERIES.fixedCohort,
                    dash: SERIES_DASH.fixedCohort,
                    points: fixed("uci_headline"),
                  },
                  {
                    id: "variable",
                    label: "whole live queue",
                    color: SERIES.variableCohort,
                    dash: SERIES_DASH.variableCohort,
                    points: variable("uci_headline"),
                  },
                ]}
                baseRates={variable("uci_headline").map((p) => ({
                  k: p.k,
                  rate: p.base_rate_k,
                  n: p.n_k,
                }))}
                caption="Area under the ROC curve against how many events have been observed, on a held-out test period."
              />
            </div>

            <div className="space-y-5">
              <p className="leading-relaxed">
                Following one group of incidents throughout, the model climbs
                from{" "}
                <span className="num" style={{ color: "var(--series-1)" }}>
                  {summary.uci.fixed_k1_auc.toFixed(3)}
                </span>{" "}
                at one event to{" "}
                <span className="num" style={{ color: "var(--series-1)" }}>
                  {summary.uci.fixed_kmax_auc.toFixed(3)}
                </span>{" "}
                at eight. The shaded bands are 95% bootstrap intervals, so you
                can see where a difference stops meaning anything.
              </p>
              <p style={{ color: "var(--text-2)" }}>
                The second line is the whole queue as it would really arrive. It
                sits higher, and most of that is composition: short incidents
                close and leave, so the survivors skew long and breach-prone. Its
                breach rate climbs from <span className="num">0.23</span> to{" "}
                <span className="num">0.73</span> along the same axis. Holding
                one group of {Object.keys(sample.cases).length} incidents fixed
                separates the model improving from the population changing under
                it.
              </p>
              <Callout>
                Around five or six events the curve crosses into territory where
                acting on it beats guessing. That crossing point, more than any
                single score, is what a triage lead would want from this.
              </Callout>
            </div>
          </div>
        </Section>
      </section>

      {/* ── 3 · How it works ─────────────────────────────────────────── */}
      <Section>
        <div ref={fieldRef} className="grid gap-12 lg:grid-cols-[1fr_1fr] lg:items-center">
          <div>
            <Eyebrow>How it works</Eyebrow>
            <h2 className="display text-[clamp(1.9rem,4.4vw,3rem)]">
              Every incident is scored at each moment of its life
            </h2>
            <p className="mt-5 leading-relaxed" style={{ color: "var(--text-2)" }}>
              Most models treat a ticket as one finished record. The first
              version of this project did that and scored{" "}
              <span className="num">{summary.leaky.auc.toFixed(3)}</span>, on
              columns that only exist after a ticket closes. This one treats
              it as a sequence and produces a row for every point along the way:
              after the first event, after the first two, and so on. Each row
              carries what the service desk could see at that point. Counters
              hold their running values. Time is time elapsed so far.
            </p>
            <p className="mt-4 leading-relaxed" style={{ color: "var(--text-2)" }}>
              The score then answers the question a team lead actually has
              at nine in the morning: how worried should I be about this one,
              right now, with the queue still open.
            </p>

            <dl className="mt-8 grid grid-cols-2 gap-x-6 gap-y-5">
              <Fact
                label="Incidents"
                value={summary.uci.n_cases_total.toLocaleString("en-GB")}
                note="ServiceNow, March to May 2016"
              />
              <Fact
                label="Scored moments"
                value="81,449"
                note="one row per incident per event"
              />
              <Fact
                label="Held-out incidents"
                value={summary.uci.split.n_test_cases.toLocaleString("en-GB")}
                note={`from ${summary.uci.split.test_boundary.slice(0, 10)}`}
              />
              <Fact
                label="Breach rate"
                value={summary.uci.breach_rate_kept_cases.toFixed(3)}
                note="across the study population"
              />
            </dl>
          </div>

          <div
            className="depth relative min-h-[420px] overflow-hidden rounded-xl"
            style={{ border: "1px solid var(--hairline)", background: "var(--surface-1)" }}
          >
            <TraceField resolve={fieldProgress} density={0.5} />
            {/* Edge falloff. The field is a rectangle inside a rounded card, so
                without this the traces stop dead against the border instead of
                receding into it. */}
            <div
              className="pointer-events-none absolute inset-0"
              style={{
                background:
                  "radial-gradient(115% 105% at 50% 45%, transparent 34%, var(--surface-1) 96%)",
              }}
            />
            {/* Scrim, so the caption stays readable whatever the field is doing
                underneath it. */}
            <div
              className="pointer-events-none absolute inset-x-0 bottom-0 h-28"
              style={{
                background:
                  "linear-gradient(to top, var(--surface-1) 18%, transparent 100%)",
              }}
            />
            <div
              className="pointer-events-none absolute bottom-4 left-5 right-5 text-[11px]"
              style={{ color: "var(--text-2)" }}
            >
              Each line is an incident moving through its life. As evidence
              accumulates, the ones heading for a breach separate from the ones
              that will hold.
            </div>
          </div>
        </div>
      </Section>

      {/* ── 4 · Beyond the curve ─────────────────────────────────────── */}
      <section id="modelling" className="scroll-mt-20">
        <Section tint>
          <Eyebrow>Beyond the curve</Eyebrow>
          <h2 className="display max-w-3xl text-[clamp(1.9rem,4.4vw,3rem)]">
            Four questions a score alone leaves open
          </h2>
          <p className="mt-5 max-w-2xl leading-relaxed" style={{ color: "var(--text-2)" }}>
            Whether the probabilities mean what they say. Whether the order of
            events carries anything. How much to trust a call at a given point.
            And what any of it is worth once escalating someone has a price.
          </p>
          <div className="mt-12">
            <Modelling />
          </div>
        </Section>
      </section>

      {/* ── 5 · Explorer ─────────────────────────────────────────────── */}
      <section id="explorer" className="scroll-mt-20">
        <Section>
          <Eyebrow>Try it</Eyebrow>
          <h2 className="display max-w-3xl text-[clamp(1.9rem,4.4vw,3rem)]">
            Drag the slider and watch the queue re-rank
          </h2>
          <p className="mt-5 max-w-2xl leading-relaxed" style={{ color: "var(--text-2)" }}>
            Real incidents from the held-out period, scored by the fitted model
            at every point in their life. At one event the ordering is close to
            arbitrary. By five or six, the ones that went on to breach have
            climbed. Reveal the outcomes to see how far off it was at each stage.
          </p>
          <div className="mt-10">
            <Explorer sample={sample} />
          </div>
        </Section>
      </section>

      {/* ── 6 · How it is built ──────────────────────────────────────── */}
      <section id="built" className="scroll-mt-20">
        <Section tint>
          <HowItsBuilt />
        </Section>
      </section>

      <Footer />
    </main>
  );
}

/* ── pieces ───────────────────────────────────────────────────────────── */

/**
 * The opening beat. Mark draws, headline lines rise in sequence, supporting
 * copy and buttons land last. Runs once on mount, never again on scroll.
 */
function Opening() {
  return (
    <div className="max-w-2xl">
      <Beat delay={0}>
        <KairosMark size={48} animate />
      </Beat>
      <h1 className="display mt-7 text-[clamp(2.7rem,6.4vw,4.6rem)]">
        <Beat delay={420}>
          Which incident
        </Beat>
        <Beat delay={520}>
          breaches next?
        </Beat>
      </h1>
      <Beat delay={680}>
        <p
          className="mt-6 max-w-lg text-lg leading-relaxed"
          style={{ color: "var(--text-2)" }}
        >
          KAIROS ranks a live incident queue by SLA-breach risk for whoever is triaging it, using only what
          is known at the moment you look. It also measures how early that call
          can be trusted, which turns out to be the harder question.
        </p>
      </Beat>
      <Beat delay={820}>
        <div className="mt-9 flex flex-wrap gap-3">
          <a
            href="#curve"
            className="rounded-lg px-5 py-3 text-sm font-medium transition-transform duration-150 hover:-translate-y-0.5"
            style={{ background: "var(--accent)", color: "var(--accent-ink)" }}
          >
            See how early
          </a>
          <a
            href="#explorer"
            className="rounded-lg px-5 py-3 text-sm font-medium transition-colors duration-150"
            style={{ border: "1px solid var(--hairline)", color: "var(--text-1)" }}
          >
            Try it on real incidents
          </a>
        </div>
      </Beat>
    </div>
  );
}

/** One step of the opening sequence: rise and fade, once, on a delay. */
/**
 * One step of the opening sequence.
 *
 * Always a block box. `display: contents` would be tidier markup, but an
 * element with no box has nothing to animate, so the beat would silently do
 * nothing.
 */
function Beat({ children, delay }: { children: React.ReactNode; delay: number }) {
  return (
    <span className="kairos-beat block" style={{ animationDelay: `${delay}ms` }}>
      {children}
    </span>
  );
}

// Solid rather than translucent. A sticky backdrop-filter promotes the whole
// document to a single composited layer, which on some GPU/driver pairs blanks
// the paint entirely; an opaque bar also reads cleaner sitting over data.
function Nav() {
  return (
    <header
      className="sticky top-0 z-50"
      style={{ background: "var(--ground)", borderBottom: "1px solid var(--hairline)" }}
    >
      <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-3.5">
        <KairosLockup />
        <nav className="flex items-center gap-6 text-sm" style={{ color: "var(--text-2)" }}>
          <a href="#curve" className="transition-colors duration-150 hover:text-[var(--text-1)]">
            Findings
          </a>
          <a href="#explorer" className="transition-colors duration-150 hover:text-[var(--text-1)]">
            Explorer
          </a>
          <a href="#built" className="transition-colors duration-150 hover:text-[var(--text-1)]">
            Method
          </a>
        </nav>
      </div>
    </header>
  );
}

function Section({ children, tint }: { children: React.ReactNode; tint?: boolean }) {
  const { ref, inView } = useInView<HTMLDivElement>();
  return (
    <div
      style={{
        background: tint ? "var(--surface-2)" : "transparent",
        borderTop: tint ? "1px solid var(--hairline)" : undefined,
        borderBottom: tint ? "1px solid var(--hairline)" : undefined,
      }}
    >
      <div
        ref={ref}
        className="mx-auto max-w-6xl px-6 py-24 sm:py-32"
        style={{
          opacity: inView ? 1 : 0,
          transform: inView ? "none" : "translateY(14px)",
          transition:
            "opacity 520ms cubic-bezier(0.22,1,0.36,1), transform 520ms cubic-bezier(0.22,1,0.36,1)",
        }}
      >
        {children}
      </div>
    </div>
  );
}

function Eyebrow({ children }: { children: React.ReactNode }) {
  return (
    <p
      className="mb-4 text-[11px] uppercase tracking-[0.18em]"
      style={{ color: "var(--accent)" }}
    >
      {children}
    </p>
  );
}

function Fact({ label, value, note }: { label: string; value: string; note: string }) {
  return (
    <div>
      <dt className="text-[11px] uppercase tracking-wider" style={{ color: "var(--text-2)" }}>
        {label}
      </dt>
      <dd className="num mt-1 text-2xl">{value}</dd>
      <dd className="text-[11px] leading-tight" style={{ color: "var(--text-2)" }}>
        {note}
      </dd>
    </div>
  );
}

function Callout({ children }: { children: React.ReactNode }) {
  return (
    <p
      className="rounded-lg p-4 text-sm leading-relaxed"
      style={{
        background: "var(--surface-2)",
        borderLeft: "3px solid var(--accent)",
        color: "var(--text-2)",
      }}
    >
      {children}
    </p>
  );
}

function Footer() {
  return (
    <footer style={{ borderTop: "1px solid var(--hairline)" }}>
      <div className="mx-auto flex max-w-6xl flex-col gap-6 px-6 py-12 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <KairosLockup />
          <p className="mt-3 max-w-md text-xs leading-relaxed" style={{ color: "var(--text-2)" }}>
            Every figure on this page is read from the study&apos;s own output.
            Rerun the pipeline and the page follows. Datasets: UCI Incident
            Management (Amaral, Fantinato &amp; Peres, 2018) and BPI Challenge
            2013 (Steeman, 2013).
          </p>
        </div>
        <div className="text-xs" style={{ color: "var(--text-2)" }}>
          <p>Abinash Prasana Selvanathan</p>
          <p className="mt-1">
            Method after Teinemaa et al. (2019) and Weytjens &amp; De Weerdt (2021)
          </p>
        </div>
      </div>
    </footer>
  );
}
