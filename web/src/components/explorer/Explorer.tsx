"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { BAND_COLOR, type ExplorerSample, type RiskBand } from "@/lib/data";

/**
 * The k-slider explorer.
 *
 * 478 real test incidents, scored by the actual fitted model at every prefix
 * length. Drag k and the queue re-ranks. At k=1 the ordering is close to noise;
 * by k=6 the breaches have visibly risen. That is the study's finding made
 * physical, and it uses the same numbers the paper reports -- nothing here is
 * simulated.
 *
 * These are the cases that ran to eight or more events, which is why the breach
 * rate is 73% rather than the 38% of the full test period. Long cases breach
 * more. Holding that cohort fixed is what lets the ranking be compared across k
 * at all, and the panel says so rather than quietly presenting it as typical.
 */

type Props = { sample: ExplorerSample };

const ROW_H = 15;
const GAP = 3;
const COLS = 26;

export function Explorer({ sample }: Props) {
  const [k, setK] = useState(1);
  const [reveal, setReveal] = useState(false);
  const [threshold, setThreshold] = useState(0.6);

  const cases = useMemo(
    () =>
      Object.entries(sample.cases).map(([id, c]) => ({
        id,
        y: c.y,
        p: c.p,
        band: c.band,
      })),
    [sample],
  );

  const ranked = useMemo(() => {
    return cases
      .map((c) => ({ ...c, score: c.p[k - 1] ?? 0, bandAtK: (c.band[k - 1] ?? "low") as RiskBand }))
      .sort((a, b) => b.score - a.score);
  }, [cases, k]);

  /**
   * Rank by case id, so the cells can stay in one fixed DOM order and move to
   * their new rank on a transform instead of being re-inserted by React. A
   * re-inserted node jumps; a transformed one travels, and the travelling is
   * the whole point of the slider.
   */
  const rankOf = useMemo(() => {
    const m = new Map<string, number>();
    ranked.forEach((c, i) => m.set(c.id, i));
    return m;
  }, [ranked]);

  const atK = useMemo(() => {
    const m = new Map<string, { score: number; band: RiskBand }>();
    for (const c of cases) {
      m.set(c.id, {
        score: c.p[k - 1] ?? 0,
        band: (c.band[k - 1] ?? "low") as RiskBand,
      });
    }
    return m;
  }, [cases, k]);

  // Cell geometry is measured rather than assumed, because the transform has
  // to be in pixels and the container is fluid.
  const gridRef = useRef<HTMLDivElement>(null);
  const [gridW, setGridW] = useState(0);
  const [animate, setAnimate] = useState(false);

  useEffect(() => {
    const el = gridRef.current;
    if (!el) return;
    setGridW(el.getBoundingClientRect().width);
    const ro = new ResizeObserver((entries) => setGridW(entries[0].contentRect.width));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // Enable the transition one frame after the first measurement, so cells are
  // placed rather than animated in from the corner.
  useEffect(() => {
    if (!gridW || animate) return;
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    const id = requestAnimationFrame(() => setAnimate(true));
    return () => cancelAnimationFrame(id);
  }, [gridW, animate]);

  const cellW = gridW > 0 ? (gridW - GAP * (COLS - 1)) / COLS : 0;
  const rows = Math.ceil(cases.length / COLS);
  const gridH = rows * ROW_H + (rows - 1) * GAP;

  const stats = useMemo(() => {
    let tp = 0, fp = 0, fn = 0, tn = 0;
    for (const c of ranked) {
      const flagged = c.score >= threshold;
      if (flagged && c.y === 1) tp++;
      else if (flagged && c.y === 0) fp++;
      else if (!flagged && c.y === 1) fn++;
      else tn++;
    }
    const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
    const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
    // Share of the true breaches that sit in the top decile of the ranking --
    // the number a triage lead actually cares about.
    const topN = Math.max(1, Math.round(ranked.length * 0.1));
    const inTop = ranked.slice(0, topN).filter((c) => c.y === 1).length;
    const totalBreach = ranked.filter((c) => c.y === 1).length;
    return {
      tp, fp, fn, tn, precision, recall, flagged: tp + fp,
      lift: totalBreach > 0 ? inTop / topN : 0,
      topN,
    };
  }, [ranked, threshold]);

  return (
    <div className="grid gap-8 lg:grid-cols-[1.35fr_1fr]">
      {/* the queue */}
      <div>
        <div
          className="rounded-xl p-5"
          style={{ background: "var(--surface-1)", border: "1px solid var(--hairline)" }}
        >
          <div className="mb-4 flex items-baseline justify-between">
            <span className="text-xs" style={{ color: "var(--text-2)" }}>
              {ranked.length} incidents, ranked by predicted risk
            </span>
            <span className="num text-xs" style={{ color: "var(--text-2)" }}>
              highest risk first
            </span>
          </div>

          <div
            ref={gridRef}
            className="relative w-full"
            style={{ height: gridH }}
            role="img"
            aria-label={`${ranked.length} test incidents laid out highest predicted risk first, after ${k} observed ${k === 1 ? "event" : "events"}.`}
          >
            {cellW > 0 &&
              cases.map((c) => {
                const rank = rankOf.get(c.id) ?? 0;
                const cur = atK.get(c.id)!;
                const flagged = cur.score >= threshold;
                const col = rank % COLS;
                const row = Math.floor(rank / COLS);
                const bg = reveal
                  ? c.y === 1
                    ? "var(--status-critical)"
                    : "var(--status-good)"
                  : BAND_COLOR[cur.band];
                return (
                  <div
                    key={c.id}
                    title={`${c.id} · p=${cur.score.toFixed(3)}${reveal ? (c.y ? " · breached" : " · met SLA") : ""}`}
                    style={{
                      position: "absolute",
                      left: 0,
                      top: 0,
                      width: cellW,
                      height: ROW_H,
                      background: bg,
                      opacity: reveal ? (c.y === 1 ? 0.95 : 0.28) : flagged ? 0.95 : 0.4,
                      borderRadius: 2,
                      transform: `translate3d(${(col * (cellW + GAP)).toFixed(2)}px, ${row * (ROW_H + GAP)}px, 0)`,
                      // No `will-change` here on purpose: 478 permanently
                      // promoted layers is enough to exhaust the compositor and
                      // blank the whole document on some drivers. The browser
                      // promotes each cell for the duration of its transition
                      // anyway, which is the only time it helps.
                      // Transitions, not keyframes, so dragging retargets
                      // mid-flight instead of restarting. The overshoot in the
                      // curve is what makes 478 cells read as settling into a
                      // new order rather than snapping to it. Delay is capped
                      // so the tail of the queue does not lag a whole second
                      // behind the head.
                      transition: animate
                        ? `transform 560ms cubic-bezier(0.34, 1.28, 0.64, 1) ${Math.min(rank * 0.5, 150).toFixed(0)}ms, opacity 180ms ease-out, background-color 180ms ease-out`
                        : "none",
                    }}
                    aria-hidden
                  />
                );
              })}
          </div>

          <div className="mt-4 flex flex-wrap items-center gap-4 text-[11px]" style={{ color: "var(--text-2)" }}>
            {reveal ? (
              <>
                <Swatch color="var(--status-critical)" label="actually breached" />
                <Swatch color="var(--status-good)" label="met SLA" />
              </>
            ) : (
              <>
                <Swatch color={BAND_COLOR.high} label="high risk (p ≥ 0.60)" />
                <Swatch color={BAND_COLOR.medium} label="medium (0.30–0.60)" />
                <Swatch color={BAND_COLOR.low} label="low (< 0.30)" />
              </>
            )}
          </div>
        </div>

        <p className="mt-3 text-xs leading-relaxed" style={{ color: "var(--text-2)" }}>
          These are the {ranked.length} test incidents that ran to eight or more events, so their
          breach rate is 73% rather than the 38% of the full test period. Long incidents breach more.
          Holding the same cases at every k is what makes the ranking comparable across the slider.
        </p>
      </div>

      {/* controls and readout */}
      <div className="flex flex-col gap-5">
        <div
          className="rounded-xl p-5"
          style={{ background: "var(--surface-1)", border: "1px solid var(--hairline)" }}
        >
          <label className="block">
            <div className="mb-2 flex items-baseline justify-between">
              <span className="text-sm font-medium">Events observed</span>
              <span className="num text-2xl" style={{ color: "var(--accent)" }}>
                k = {k}
              </span>
            </div>
            <input
              type="range"
              min={1}
              max={sample.k_max}
              step={1}
              value={k}
              onChange={(e) => setK(Number(e.target.value))}
              className="w-full accent-[var(--accent)]"
              aria-label={`Prefix length, currently ${k} of ${sample.k_max} events`}
            />
            <div className="num mt-1 flex justify-between text-[10px]" style={{ color: "var(--text-2)" }}>
              <span>1</span>
              <span>{sample.k_max}</span>
            </div>
          </label>

          <label className="mt-5 block">
            <div className="mb-2 flex items-baseline justify-between">
              <span className="text-sm font-medium">Alert threshold</span>
              <span className="num text-sm" style={{ color: "var(--text-2)" }}>
                {threshold.toFixed(2)}
              </span>
            </div>
            <input
              type="range"
              min={0.05}
              max={0.95}
              step={0.05}
              value={threshold}
              onChange={(e) => setThreshold(Number(e.target.value))}
              className="w-full accent-[var(--accent)]"
              aria-label={`Alert threshold, currently ${threshold.toFixed(2)}`}
            />
          </label>

          <button
            onClick={() => setReveal((v) => !v)}
            className="mt-5 w-full rounded-lg px-4 py-2.5 text-sm font-medium transition-colors duration-150"
            style={{
              background: reveal ? "var(--surface-2)" : "var(--accent)",
              color: reveal ? "var(--text-1)" : "var(--accent-ink)",
              border: "1px solid var(--hairline)",
            }}
          >
            {reveal ? "Hide what actually happened" : "Reveal what actually happened"}
          </button>
        </div>

        <div
          className="rounded-xl p-5"
          style={{ background: "var(--surface-1)", border: "1px solid var(--hairline)" }}
        >
          <div className="grid grid-cols-2 gap-4">
            <Stat label="Precision" value={stats.precision} hint="of those flagged, actually breached" />
            <Stat label="Recall" value={stats.recall} hint="of real breaches, caught" />
            <Stat
              label="Breaches in top 10%"
              value={stats.lift}
              hint={`${stats.topN} highest-risk incidents`}
            />
            <Stat
              label="Flagged"
              value={stats.flagged / ranked.length}
              hint={`${stats.flagged} of ${ranked.length}`}
            />
          </div>

          <table className="num mt-5 w-full text-xs">
            <caption className="mb-2 text-left text-[11px]" style={{ color: "var(--text-2)" }}>
              Confusion matrix at k = {k}, threshold {threshold.toFixed(2)}
            </caption>
            <tbody>
              <tr>
                <td className="py-1 pr-2 text-[11px]" style={{ color: "var(--text-2)" }}>
                  flagged &amp; breached
                </td>
                <td className="py-1 text-right">{stats.tp}</td>
                <td className="py-1 pl-4 pr-2 text-[11px]" style={{ color: "var(--text-2)" }}>
                  false alarm
                </td>
                <td className="py-1 text-right">{stats.fp}</td>
              </tr>
              <tr>
                <td className="py-1 pr-2 text-[11px]" style={{ color: "var(--text-2)" }}>
                  missed breach
                </td>
                <td className="py-1 text-right">{stats.fn}</td>
                <td className="py-1 pl-4 pr-2 text-[11px]" style={{ color: "var(--text-2)" }}>
                  correctly quiet
                </td>
                <td className="py-1 text-right">{stats.tn}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function Swatch({ color, label }: { color: string; label: string }) {
  return (
    <span className="flex items-center gap-1.5">
      <span style={{ width: 10, height: 10, borderRadius: 2, background: color }} />
      {label}
    </span>
  );
}

function Stat({ label, value, hint }: { label: string; value: number; hint: string }) {
  return (
    <div>
      <div className="text-[11px]" style={{ color: "var(--text-2)" }}>
        {label}
      </div>
      <div className="num text-2xl" style={{ color: "var(--text-1)" }}>
        {(value * 100).toFixed(0)}%
      </div>
      <div className="text-[10px] leading-tight" style={{ color: "var(--text-2)" }}>
        {hint}
      </div>
    </div>
  );
}
