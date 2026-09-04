"use client";

/**
 * The KAIROS threshold mark.
 *
 * A rising curve crossing a horizontal baseline, with the crossing marked by a
 * single filled dot. The curve's inflection meeting the vertical stem reads as
 * an abstract K. It is the study in one glyph: something becomes knowable at a
 * particular moment, and the mark points at that moment.
 *
 * Inline SVG rather than an image file so it inherits `currentColor` and can be
 * animated. The dot is the only filled element and the only place the accent
 * hue is allowed to appear.
 */

type Props = {
  size?: number;
  animate?: boolean;
  className?: string;
  title?: string;
};

export function KairosMark({ size = 24, animate = false, className, title }: Props) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      className={className}
      role={title ? "img" : "presentation"}
      aria-label={title}
      aria-hidden={title ? undefined : true}
    >
      {/* baseline -- the threshold being crossed */}
      <path
        d="M2 16 H22"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
        opacity="0.35"
        className={animate ? "kairos-base" : undefined}
      />
      {/* vertical stem -- the K's spine, and the k=1 axis */}
      <path
        d="M4 4 V20"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
        opacity="0.35"
        className={animate ? "kairos-base" : undefined}
      />
      {/* the curve: flat, then rising through the baseline */}
      <path
        d="M4 19 C 9 19, 12 18, 14.5 16 C 17.5 13.5, 19 8.5, 21 4.5"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
        className={animate ? "kairos-curve" : undefined}
      />
      {/* the crossing -- the only filled element, the only accent */}
      <circle
        cx="14.5"
        cy="16"
        r="2.4"
        fill="var(--accent)"
        className={animate ? "kairos-dot" : undefined}
      />

      {animate && (
        <style>{`
          .kairos-curve {
            stroke-dasharray: 26;
            stroke-dashoffset: 26;
            animation: kairos-draw 900ms cubic-bezier(0.22, 1, 0.36, 1) forwards;
          }
          .kairos-base { opacity: 0; animation: kairos-fade 400ms ease-out 120ms forwards; }
          .kairos-dot {
            opacity: 0;
            transform-origin: 14.5px 16px;
            animation: kairos-land 120ms ease-out 820ms forwards;
          }
          @keyframes kairos-draw { to { stroke-dashoffset: 0; } }
          @keyframes kairos-fade { to { opacity: 0.35; } }
          /* never from scale(0) -- nothing appears out of nothing */
          @keyframes kairos-land {
            from { opacity: 0; transform: scale(0.9); }
            to   { opacity: 1; transform: scale(1); }
          }
          @media (prefers-reduced-motion: reduce) {
            .kairos-curve, .kairos-base, .kairos-dot {
              animation: none;
              stroke-dashoffset: 0;
              opacity: 1;
            }
            .kairos-base { opacity: 0.35; }
          }
        `}</style>
      )}
    </svg>
  );
}

export function KairosLockup({ animate = false }: { animate?: boolean }) {
  return (
    <span className="inline-flex items-center gap-2.5">
      <KairosMark size={26} animate={animate} title="KAIROS" />
      <span
        className="display text-[1.35rem] tracking-[0.14em]"
        style={{ color: "var(--text-1)" }}
      >
        KAIROS
      </span>
    </span>
  );
}
