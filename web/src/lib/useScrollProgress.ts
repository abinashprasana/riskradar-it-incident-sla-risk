"use client";

import { useEffect, useLayoutEffect, useRef, useState } from "react";

/**
 * Progress of an element through the viewport, 0 to 1.
 *
 * Driven by scroll position rather than a timer, so every animation bound to it
 * is scrubbable and reversible -- the viewer stays in control of the pace. Reads
 * are batched into a rAF to keep layout thrash off the scroll thread.
 *
 * Returns 1 immediately under reduced motion, so anything gated on progress is
 * simply present rather than animating in.
 */
export function useScrollProgress<T extends HTMLElement>(
  opts: { start?: number; end?: number } = {},
) {
  const { start = 0.9, end = 0.25 } = opts;
  const ref = useRef<T>(null);
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      // Reading a media query needs `window`, so this cannot move out of the
      // effect. One write, once; there is no cascade to trigger.
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setProgress(1);
      return;
    }

    let frame = 0;
    const measure = () => {
      frame = 0;
      const el = ref.current;
      if (!el) return;
      const rect = el.getBoundingClientRect();
      const vh = window.innerHeight;
      const from = vh * start;
      const to = vh * end;
      const p = (from - rect.top) / (from - to);
      setProgress(Math.max(0, Math.min(1, p)));
    };

    const onScroll = () => {
      if (!frame) frame = requestAnimationFrame(measure);
    };

    measure();
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onScroll, { passive: true });
    return () => {
      if (frame) cancelAnimationFrame(frame);
      window.removeEventListener("scroll", onScroll);
      window.removeEventListener("resize", onScroll);
    };
  }, [start, end]);

  return { ref, progress };
}

/**
 * Fires once when an element first enters the viewport.
 *
 * Fails *open*, deliberately. A reveal-on-scroll effect that starts at
 * opacity 0 will hide the content permanently if the observer never fires --
 * on a tall section, an unusual viewport, a blocked API, or simply a browser
 * that disagrees about ratios. Content is never worth risking for an entrance
 * animation, so there are three independent ways to become visible: the
 * observer, an immediate on-mount position check, and a timeout backstop.
 */
export function useInView<T extends HTMLElement>() {
  const ref = useRef<T>(null);
  // "idle" renders visible. An element only becomes "hidden" once JS has run
  // and confirmed it is below the fold, so a failure anywhere in this hook
  // leaves the content on screen rather than erasing it.
  const [phase, setPhase] = useState<"idle" | "hidden" | "shown">("idle");

  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    if (
      window.matchMedia("(prefers-reduced-motion: reduce)").matches ||
      typeof IntersectionObserver === "undefined" ||
      el.getBoundingClientRect().top < window.innerHeight
    ) {
      setPhase("shown");
      return;
    }

    // Runs before paint, so hiding here never flashes.
    setPhase("hidden");

    const io = new IntersectionObserver(
      ([e]) => {
        if (e.isIntersecting) {
          setPhase("shown");
          io.disconnect();
        }
      },
      { threshold: 0, rootMargin: "0px 0px -10% 0px" },
    );
    io.observe(el);

    const backstop = window.setTimeout(() => setPhase("shown"), 3000);
    return () => {
      io.disconnect();
      window.clearTimeout(backstop);
    };
  }, []);

  return { ref, inView: phase !== "hidden" };
}
