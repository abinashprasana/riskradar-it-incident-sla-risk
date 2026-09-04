import { ImageResponse } from "next/og";

export const alt = "KAIROS — Incident SLA Breach Risk, Scored Early";
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";
// Required under `output: "export"` -- the card is baked once at build time.
export const dynamic = "force-static";

/**
 * Social card, rendered at build time. Draws the real fixed-cohort earliness
 * curve rather than a decorative shape, so the preview shows the finding.
 */
export default async function Image() {
  // Fixed cohort, k = 1..8 (artifacts/uci_servicenow/agg/per_k.csv). Kept in
  // step with the same series the page draws.
  const auc = [0.6185, 0.6311, 0.6925, 0.725, 0.7718, 0.803, 0.8508, 0.8874];
  const W = 520, H = 220, lo = 0.55, hi = 0.95;
  const pts = auc.map((v, i) => [
    (i / (auc.length - 1)) * W,
    H - ((v - lo) / (hi - lo)) * H,
  ]);
  const d = pts.map(([x, y], i) => `${i ? "L" : "M"}${x.toFixed(1)} ${y.toFixed(1)}`).join(" ");

  return new ImageResponse(
    (
      <div
        style={{
          width: "100%", height: "100%", display: "flex", flexDirection: "column",
          justifyContent: "space-between", background: "#0b1016", padding: 64,
          fontFamily: "sans-serif",
        }}
      >
        <div style={{ display: "flex", flexDirection: "column" }}>
          <div style={{ color: "#2f9fd0", fontSize: 22, letterSpacing: 6 }}>KAIROS</div>
          <div style={{ color: "#e9eef3", fontSize: 62, lineHeight: 1.1, marginTop: 18, maxWidth: 820 }}>
            Which incident breaches next?
          </div>
          <div style={{ color: "#9aa9b7", fontSize: 26, marginTop: 18 }}>
            Risk from partial information. AUC 0.619 at one event, 0.887 at eight.
          </div>
        </div>
        <div style={{ display: "flex", alignItems: "flex-end", justifyContent: "space-between" }}>
          <svg width={W} height={H}>
            <path d={d} fill="none" stroke="#2f9fd0" strokeWidth="5"
                  strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          <div style={{ color: "#9aa9b7", fontSize: 20, display: "flex" }}>
            prefix-based · out-of-time
          </div>
        </div>
      </div>
    ),
    size,
  );
}
