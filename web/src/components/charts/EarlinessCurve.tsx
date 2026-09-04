"use client";

import { useMemo, useRef, useState } from "react";
import type { PerK } from "@/lib/data";

/**
 * The earliness curve: AUC as a function of how much of a trace is observed.
 *
 * Two cohorts, deliberately both shown. The variable cohort uses every prefix
 * available at each k, but its base rate climbs from 0.23 to 0.73 as short
 * cases exhaust, so it rises partly through composition. The fixed cohort
 * holds the same 478 cases at every k, so movement there is the model knowing
 * more. Only the second answers the question, and showing them together is the
 * only honest way to make that visible rather than asserted.
 *
 * Hand-rolled SVG rather than a chart library because the line has to draw
 * itself against scroll progress, and because the base-rate panel underneath
 * must share an x-axis without becoming a second y-axis.
 */

type Series = {
  id: string;
  label: string;
  color: string;
  dash?: string;
  points: PerK[];
};

type Props = {
  series: Series[];
  progress?: number; // 0..1, how much of the curve has been revealed
  baseRates?: { k: number; rate: number; n: number }[];
  yMin?: number;
  yMax?: number;
  caption?: string;
  /** Draw the 95% bootstrap interval as a filled band behind each line. */
  showBand?: boolean;
  /** A dot that rides the line as it draws, labelled with k and the value. */
  readHead?: boolean;
};

const W = 720;
const H = 340;
const BAR_H = 96;
const M = { top: 18, right: 92, bottom: 34, left: 52 };

export function EarlinessCurve({
  series,
  progress = 1,
  baseRates,
  yMin = 0.5,
  yMax = 0.95,
  caption,
  showBand = false,
  readHead = false,
}: Props) {
  const [hoverK, setHoverK] = useState<number | null>(null);
  const [showTable, setShowTable] = useState(false);
  const svgRef = useRef<SVGSVGElement>(null);

  const kMax = useMemo(
    () => Math.max(...series.flatMap((s) => s.points.map((p) => p.k))),
    [series],
  );
  const kMin = 1;

  const plotW = W - M.left - M.right;
  const plotH = H - M.top - M.bottom;

  const x = (k: number) => M.left + ((k - kMin) / (kMax - kMin)) * plotW;
  const y = (v: number) => M.top + plotH - ((v - yMin) / (yMax - yMin)) * plotH;

  const path = (pts: PerK[]) =>
    pts
      .slice()
      .sort((a, b) => a.k - b.k)
      .map((p, i) => `${i === 0 ? "M" : "L"} ${x(p.k).toFixed(1)} ${y(p.auc).toFixed(1)}`)
      .join(" ");

  const band = (pts: PerK[]) => {
    const ordered = pts
      .filter((p): p is PerK & { auc_lo: number; auc_hi: number } =>
        p.auc_lo !== undefined && p.auc_hi !== undefined)
      .sort((a, b) => a.k - b.k);
    if (ordered.length < 2) return "";
    const up = ordered.map((p) => `${x(p.k).toFixed(1)} ${y(p.auc_hi).toFixed(1)}`);
    const down = ordered
      .slice()
      .reverse()
      .map((p) => `${x(p.k).toFixed(1)} ${y(p.auc_lo).toFixed(1)}`);
    return `M ${up.join(" L ")} L ${down.join(" L ")} Z`;
  };

  /** Where the read-head sits: interpolated along the drawn part of the line. */
  const headAt = (pts: PerK[]) => {
    const o = pts.slice().sort((a, b) => a.k - b.k);
    if (!o.length) return null;
    const t = Math.max(0, Math.min(1, progress)) * (o.length - 1);
    const i = Math.floor(t);
    const f = t - i;
    const a = o[i];
    const b = o[Math.min(i + 1, o.length - 1)];
    return {
      x: x(a.k) + (x(b.k) - x(a.k)) * f,
      y: y(a.auc) + (y(b.auc) - y(a.auc)) * f,
      k: Math.round(a.k + (b.k - a.k) * f),
      auc: a.auc + (b.auc - a.auc) * f,
    };
  };

  const yTicks = useMemo(() => {
    const out: number[] = [];
    for (let v = Math.ceil(yMin * 10) / 10; v <= yMax + 1e-9; v += 0.1) out.push(Number(v.toFixed(1)));
    return out;
  }, [yMin, yMax]);

  const handleMove = (e: React.MouseEvent<SVGSVGElement>) => {
    const svg = svgRef.current;
    if (!svg) return;
    const rect = svg.getBoundingClientRect();
    const px = ((e.clientX - rect.left) / rect.width) * W;
    const k = Math.round(kMin + ((px - M.left) / plotW) * (kMax - kMin));
    setHoverK(k >= kMin && k <= kMax ? k : null);
  };

  const maxBase = baseRates ? Math.max(...baseRates.map((b) => b.rate)) : 1;

  /**
   * Where each series' direct label sits.
   *
   * The two cohorts converge to within 0.001 at k=8, so labels anchored to
   * their own endpoints print on top of each other. Lay them out top-down and
   * push any that come within 13px apart -- the label is what carries identity
   * for colour-vision-deficient readers, so it has to stay legible.
   */
  const labelY = useMemo(() => {
    const ends = series
      .map((s) => {
        const last = s.points.slice().sort((a, b) => a.k - b.k).at(-1);
        return last ? { id: s.id, y: y(last.auc), k: last.k } : null;
      })
      .filter((v): v is { id: string; y: number; k: number } => v !== null)
      .sort((a, b) => a.y - b.y);

    const out: Record<string, { y: number; k: number }> = {};
    let prev = -Infinity;
    for (const e of ends) {
      const placed = Math.max(e.y, prev + 13);
      out[e.id] = { y: placed, k: e.k };
      prev = placed;
    }
    return out;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [series, yMin, yMax]);

  return (
    <figure className="w-full">
      <div className="relative">
        <svg
          ref={svgRef}
          viewBox={`0 0 ${W} ${H + (baseRates ? BAR_H + 18 : 0)}`}
          className="w-full h-auto"
          role="img"
          aria-label={
            caption ??
            `Earliness curve: area under the ROC curve against prefix length, from k equals 1 to k equals ${kMax}.`
          }
          onMouseMove={handleMove}
          onMouseLeave={() => setHoverK(null)}
        >
          {/* recessive grid */}
          {yTicks.map((v) => (
            <g key={v}>
              <line
                x1={M.left}
                x2={M.left + plotW * Math.min(1, progress * 1.35 + 0.08)}
                y1={y(v)}
                y2={y(v)}
                stroke="var(--hairline)"
                strokeWidth="1"
              />
              <text
                x={M.left - 10}
                y={y(v) + 4}
                textAnchor="end"
                className="num"
                fontSize="11"
                fill="var(--text-2)"
              >
                {v.toFixed(1)}
              </text>
            </g>
          ))}

          {/* chance line -- 0.5 is the floor any model must beat */}
          {yMin <= 0.5 && (
            <g>
              <line
                x1={M.left}
                x2={M.left + plotW}
                y1={y(0.5)}
                y2={y(0.5)}
                stroke="var(--text-2)"
                strokeWidth="1"
                strokeDasharray="2 4"
                opacity="0.6"
              />
              <text x={M.left + 4} y={y(0.5) - 6} fontSize="10" fill="var(--text-2)">
                chance
              </text>
            </g>
          )}

          {/* x axis */}
          {Array.from({ length: kMax }, (_, i) => i + 1).map((k) => (
            <text
              key={k}
              x={x(k)}
              y={M.top + plotH + 20}
              textAnchor="middle"
              className="num"
              fontSize="11"
              fill="var(--text-2)"
            >
              {k}
            </text>
          ))}
          <text
            x={M.left + plotW / 2}
            y={M.top + plotH + 34}
            textAnchor="middle"
            fontSize="11"
            fill="var(--text-2)"
          >
            events observed (k)
          </text>

          {/* crosshair */}
          {hoverK !== null && (
            <line
              x1={x(hoverK)}
              x2={x(hoverK)}
              y1={M.top}
              y2={M.top + plotH}
              stroke="var(--accent)"
              strokeWidth="1"
              opacity="0.5"
            />
          )}

          {/* confidence bands, behind everything else */}
          {showBand &&
            series.map((s) => (
              <path
                key={`band-${s.id}`}
                d={band(s.points)}
                fill={s.color}
                opacity={0.1 * Math.min(1, progress * 1.4)}
                stroke="none"
              />
            ))}

          {/* series */}
          {series.map((s) => {
            const d = path(s.points);
            return (
              <g key={s.id}>
                <path
                  d={d}
                  fill="none"
                  stroke={s.color}
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  // Normalising the path to length 1 makes the dash maths
                  // independent of the path's real length, so the draw-on-
                  // scroll works the same at any viewport width.
                  pathLength={1}
                  strokeDasharray={s.dash ?? "1"}
                  strokeDashoffset={s.dash ? 0 : 1 - progress}
                  style={{ opacity: s.dash ? progress : 1 }}
                />
                {s.points
                  .slice()
                  .sort((a, b) => a.k - b.k)
                  .map((p) => (
                    <circle
                      key={p.k}
                      cx={x(p.k)}
                      cy={y(p.auc)}
                      r={hoverK === p.k ? 5 : 3.2}
                      fill={s.color}
                      stroke="var(--ground)"
                      strokeWidth="2"
                      opacity={progress > (p.k - 1) / kMax ? 1 : 0}
                      style={{ transition: "r 120ms ease-out" }}
                    />
                  ))}
                {/* direct label -- identity never rests on colour alone */}
                {(() => {
                  const pos = labelY[s.id];
                  if (!pos) return null;
                  return (
                    <text
                      x={x(pos.k) + 10}
                      y={pos.y + 4}
                      fontSize="11"
                      fill={s.color}
                      opacity={progress > 0.55 ? 1 : 0}
                      style={{ transition: "opacity 200ms ease-out" }}
                    >
                      {s.label}
                    </text>
                  );
                })()}
              </g>
            );
          })}

          {/* base-rate panel: shares the x-axis, never a second y-axis */}
          {baseRates && (
            <g transform={`translate(0, ${H + 14})`}>
              <text x={M.left} y={10} fontSize="10.5" fill="var(--text-2)">
                breach rate of the cohort at each k
              </text>
              {baseRates.map((b) => {
                const h = (b.rate / maxBase) * (BAR_H - 34);
                return (
                  <g key={b.k}>
                    <rect
                      x={x(b.k) - 12}
                      y={BAR_H - 16 - h}
                      width="24"
                      height={h}
                      fill="var(--text-2)"
                      opacity={hoverK === b.k ? 0.55 : 0.28}
                      rx="2"
                    />
                    <text
                      x={x(b.k)}
                      y={BAR_H - 4}
                      textAnchor="middle"
                      className="num"
                      fontSize="9.5"
                      fill="var(--text-2)"
                    >
                      {b.rate.toFixed(2)}
                    </text>
                  </g>
                );
              })}
            </g>
          )}
          {/* read-head: gives the draw a subject to follow */}
          {readHead && progress > 0.02 && progress < 0.995 &&
            (() => {
              const h = headAt(series[0].points);
              if (!h) return null;
              return (
                <g>
                  <line
                    x1={h.x}
                    x2={h.x}
                    y1={M.top}
                    y2={M.top + plotH}
                    stroke={series[0].color}
                    strokeWidth="1"
                    opacity="0.28"
                  />
                  <circle cx={h.x} cy={h.y} r="5.5" fill={series[0].color} />
                  <circle cx={h.x} cy={h.y} r="10" fill={series[0].color} opacity="0.18" />
                  <text
                    x={h.x + 13}
                    y={h.y - 9}
                    className="num"
                    fontSize="11"
                    fill="var(--text-1)"
                  >
                    {h.auc.toFixed(3)}
                  </text>
                  <text
                    x={h.x + 13}
                    y={h.y + 3}
                    className="num"
                    fontSize="9.5"
                    fill="var(--text-2)"
                  >
                    k = {h.k}
                  </text>
                </g>
              );
            })()}
        </svg>

        {/* tooltip */}
        {hoverK !== null && (
          <div
            className="pointer-events-none absolute rounded-lg px-3 py-2 text-xs"
            style={{
              left: `${(x(hoverK) / W) * 100}%`,
              top: 8,
              transform: "translateX(-50%)",
              background: "var(--surface-1)",
              border: "1px solid var(--hairline)",
              boxShadow: "0 8px 30px rgba(0,0,0,.28)",
              color: "var(--text-1)",
              minWidth: 150,
            }}
          >
            <div className="num mb-1 text-[11px]" style={{ color: "var(--text-2)" }}>
              k = {hoverK}
            </div>
            {series.map((s) => {
              const p = s.points.find((q) => q.k === hoverK);
              if (!p) return null;
              return (
                <div key={s.id} className="flex items-center justify-between gap-3">
                  <span className="flex items-center gap-1.5">
                    <span
                      style={{
                        width: 8,
                        height: 8,
                        borderRadius: 2,
                        background: s.color,
                        display: "inline-block",
                      }}
                    />
                    {s.label}
                  </span>
                  <span className="num">{p.auc.toFixed(3)}</span>
                </div>
              );
            })}
            {(() => {
              const p = series[0]?.points.find((q) => q.k === hoverK);
              return p ? (
                <div
                  className="num mt-1 pt-1 text-[10.5px]"
                  style={{ borderTop: "1px solid var(--hairline)", color: "var(--text-2)" }}
                >
                  n = {p.n_k.toLocaleString("en-GB")}
                </div>
              ) : null;
            })()}
          </div>
        )}
      </div>

      <figcaption className="mt-3 flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap items-center gap-4 text-xs" style={{ color: "var(--text-2)" }}>
          {series.map((s) => (
            <span key={s.id} className="flex items-center gap-2">
              <svg width="22" height="8" aria-hidden>
                <line
                  x1="0"
                  y1="4"
                  x2="22"
                  y2="4"
                  stroke={s.color}
                  strokeWidth="2"
                  strokeDasharray={s.dash}
                />
              </svg>
              {s.label}
            </span>
          ))}
        </div>
        <button
          onClick={() => setShowTable((v) => !v)}
          className="text-xs underline underline-offset-4"
          style={{ color: "var(--text-2)" }}
        >
          {showTable ? "hide table" : "view as table"}
        </button>
      </figcaption>

      {showTable && (
        <div className="mt-3 overflow-x-auto">
          <table className="w-full text-xs num" style={{ borderCollapse: "collapse" }}>
            <thead>
              <tr style={{ color: "var(--text-2)" }}>
                <th className="p-2 text-left">k</th>
                {series.map((s) => (
                  <th key={s.id} className="p-2 text-right">
                    {s.label} AUC
                  </th>
                ))}
                <th className="p-2 text-right">n</th>
                <th className="p-2 text-right">base rate</th>
              </tr>
            </thead>
            <tbody>
              {Array.from({ length: kMax }, (_, i) => i + 1).map((k) => {
                const ref = series[0]?.points.find((p) => p.k === k);
                return (
                  <tr key={k} style={{ borderTop: "1px solid var(--hairline)" }}>
                    <td className="p-2">{k}</td>
                    {series.map((s) => {
                      const p = s.points.find((q) => q.k === k);
                      return (
                        <td key={s.id} className="p-2 text-right">
                          {p ? p.auc.toFixed(3) : "—"}
                        </td>
                      );
                    })}
                    <td className="p-2 text-right">{ref?.n_k.toLocaleString("en-GB") ?? "—"}</td>
                    <td className="p-2 text-right">{ref?.base_rate_k.toFixed(3) ?? "—"}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </figure>
  );
}
