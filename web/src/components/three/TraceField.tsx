"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import * as THREE from "three";

/**
 * The trace field.
 *
 * Every line is one incident trace. Traces that breach bend away from the
 * decision horizon; traces that hold stay flat against it. As `resolve` rises
 * from 0 to 1 the field separates by outcome, which is what an AUC of 0.89
 * looks like as geometry rather than as a number.
 *
 * `resolve` is driven by scroll position, not by a clock, so the motion is
 * scrubbable and reversible and the viewer stays in control of it.
 *
 * Rendering is one instanced draw call. The field pauses entirely when it
 * scrolls out of view, and never mounts at all under reduced-motion or on
 * small screens -- the page carries a still image there instead.
 */

const VERT = /* glsl */ `
  attribute float aBreach;    // 0 or 1 -- the case's true outcome
  attribute float aSeed;      // per-trace randomness
  attribute float aAlong;     // 0..1 along this trace's own length
  attribute float aPhase;     // where this trace's evidence packet starts

  uniform float uTime;
  uniform float uResolve;     // 0 = unknown, 1 = fully separated
  uniform float uBias;        // 0 = full bleed, 1 = cleared off the left

  varying float vBreach;
  varying float vAlong;
  varying float vFade;
  varying float vPhase;
  varying float vDepth;

  void main() {
    vBreach = aBreach;
    vPhase = aPhase;

    // Along-ness comes from the attribute, not from position.x, because every
    // trace now starts and ends somewhere different. Deriving it from x would
    // make a short trace look like the middle of a long one.
    vAlong = aAlong;

    vec3 p = position;

    // Traces leave the origin bunched and open out along their length. Starting
    // them at full spread meant every line crossed every other line and the
    // field read as cross-hatch; starting them together makes the same geometry
    // read as what it is, a queue that begins undifferentiated and separates.
    p.y *= mix(0.18, 1.0, aAlong);

    // Each trace bows slightly, in its own direction. Perfectly straight lines
    // en masse read as a ruled grid; a little curvature reads as paths.
    p.y += sin(aAlong * 3.14159) * (aSeed - 0.5) * 0.8;

    // Idle drift: the field is alive but tells you nothing yet.
    float drift = sin(uTime * 0.22 + aSeed * 6.28) * 0.35
                + cos(uTime * 0.13 + aSeed * 3.14) * 0.22;
    p.y += drift * (1.0 - uResolve * 0.75);

    // Resolution: breaching traces climb away from the horizon, holding
    // traces flatten onto it. Separation grows along the trace, because
    // evidence accumulates with k.
    float lift = (aBreach * 2.0 - 1.0) * vAlong * vAlong * 3.6;
    p.y += lift * uResolve;

    // Slight convergence toward the horizon line at the far end.
    p.z += sin(aSeed * 12.9) * 0.4 * (1.0 - uResolve * 0.3);

    vFade = smoothstep(0.0, 0.22, vAlong) * (1.0 - smoothstep(0.80, 1.0, vAlong));

    // Composition mask. uBias > 0 fades the left of the field out, which is
    // where the headline sits. Legibility comes from the field not being there,
    // not from dimming it until the motion is pointless.
    float side = smoothstep(-4.0, 6.0, p.x);
    vFade *= mix(1.0, side, uBias);

    vec4 mv = modelViewMatrix * vec4(p, 1.0);

    // Depth cue. Without it the far traces are as loud as the near ones and
    // the field flattens into a hatch.
    vDepth = clamp((-mv.z - 5.0) / 26.0, 0.0, 1.0);

    gl_Position = projectionMatrix * mv;
  }
`;

const FRAG = /* glsl */ `
  precision highp float;

  uniform vec3 uHold;
  uniform vec3 uBreach;
  uniform vec3 uIdle;
  uniform float uResolve;
  uniform float uOpacity;
  uniform float uTime;

  varying float vBreach;
  varying float vAlong;
  varying float vFade;
  varying float vPhase;
  varying float vDepth;

  void main() {
    vec3 outcome = mix(uHold, uBreach, vBreach);
    vec3 c = mix(uIdle, outcome, uResolve);

    // The evidence packet: a bright band that runs the length of each trace,
    // every trace on its own phase. This is the sentence under the card drawn
    // literally -- an incident moving through its life -- and it is what stops
    // 560 static lines reading as texture.
    // Two packets per trace, half a length apart, so the field is never mostly
    // dark between passes. One alone left long dead stretches.
    float t = uTime * 0.085 + vPhase;
    float d1 = vAlong - fract(t);
    float d2 = vAlong - fract(t + 0.5);
    float pulse = exp(-(d1 * d1) * 15.0) + exp(-(d2 * d2) * 15.0) * 0.55;

    // A standing brightness toward the leading edge, so a trace still has
    // direction between packets.
    float head = smoothstep(0.18, 1.0, vAlong);
    head = head * head;

    c += head * 0.11 + pulse * 0.26;

    float a = uOpacity * vFade * (0.085 + head * 0.20 + pulse * 0.55);
    a *= mix(1.0, 0.45, vDepth);
    gl_FragColor = vec4(c, a);
  }
`;

/**
 * Deterministic PRNG (mulberry32).
 *
 * The field's geometry is built inside a useMemo, which React may discard and
 * recompute. Math.random() there would hand back a different field each time --
 * a real correctness problem, not a style one, since the layout would shift
 * under the viewer and server and client would disagree. A fixed seed makes the
 * field reproducible: the same incidents land in the same places every render.
 */
function makeRng(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function hexOf(varName: string, fallback: string) {
  if (typeof window === "undefined") return fallback;
  const v = getComputedStyle(document.documentElement).getPropertyValue(varName).trim();
  return v || fallback;
}

const SEGMENTS = 24;

function Field({
  resolve,
  count,
  bias,
  density,
}: {
  resolve: number;
  count: number;
  bias: number;
  density: number;
}) {
  const ref = useRef<THREE.LineSegments>(null);
  const matRef = useRef<THREE.ShaderMaterial>(null);
  const { invalidate } = useThree();
  const smoothed = useRef(0);

  const geometry = useMemo(() => {
    const rand = makeRng(0x4b41_4952); // "KAIR"
    const positions: number[] = [];
    const breach: number[] = [];
    const seed: number[] = [];
    const along: number[] = [];
    const phase: number[] = [];

    // Base rate 0.38 -- the real proportion in the UCI test period, so the
    // field's composition is not invented.
    for (let i = 0; i < count; i++) {
      const isBreach = rand() < 0.38 ? 1 : 0;
      const s = rand();
      const ph = rand();
      const y0 = (rand() - 0.5) * 7.5;
      const z0 = (rand() - 0.5) * 22;

      // Every trace has its own span. Incidents do not all start and finish
      // together, and when they did, all 560 heads landed on the same plane
      // and stacked into one bright vertical seam at the right of the frame.
      const xa = -10.5 + rand() * 4.5;
      const xb = 2.5 + rand() * 8.5;

      for (let j = 0; j < SEGMENTS; j++) {
        const t0 = j / SEGMENTS;
        const t1 = (j + 1) / SEGMENTS;
        positions.push(xa + (xb - xa) * t0, y0, z0, xa + (xb - xa) * t1, y0, z0);
        breach.push(isBreach, isBreach);
        seed.push(s, s);
        along.push(t0, t1);
        phase.push(ph, ph);
      }
    }

    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    g.setAttribute("aBreach", new THREE.Float32BufferAttribute(breach, 1));
    g.setAttribute("aSeed", new THREE.Float32BufferAttribute(seed, 1));
    g.setAttribute("aAlong", new THREE.Float32BufferAttribute(along, 1));
    g.setAttribute("aPhase", new THREE.Float32BufferAttribute(phase, 1));
    return g;
  }, [count]);

  const uniforms = useMemo(
    () => ({
      uTime: { value: 0 },
      uResolve: { value: 0 },
      uOpacity: { value: 1 },
      uBias: { value: bias },
      uIdle: { value: new THREE.Color(hexOf("--text-2", "#9aa9b7")).multiplyScalar(0.38) },
      uHold: { value: new THREE.Color(hexOf("--series-1", "#2f9fd0")).multiplyScalar(0.85) },
      uBreach: { value: new THREE.Color(hexOf("--series-6", "#e05c58")) },
    }),
    [bias],
  );

  useEffect(() => {
    invalidate();
  }, [resolve, invalidate]);

  useFrame((state, delta) => {
    if (!matRef.current) return;
    matRef.current.uniforms.uTime.value = state.clock.elapsedTime;
    // Ease toward the scroll target rather than snapping, so a fast scroll
    // still reads as the field settling rather than teleporting.
    smoothed.current += (resolve - smoothed.current) * Math.min(1, delta * 4);
    matRef.current.uniforms.uResolve.value = smoothed.current;
    // Density ramps in over the first second so the field arrives with the
    // headline instead of being already there when the page paints.
    const intro = Math.min(1, state.clock.elapsedTime / 1.0);
    matRef.current.uniforms.uOpacity.value = intro * intro * density;
    // Barely any yaw. At -0.16 the depth spread sheared every horizontal
    // trace into its own diagonal and the field turned into a hatch.
    if (ref.current) ref.current.rotation.y = -0.05 + smoothed.current * 0.03;
  });

  return (
    <lineSegments ref={ref} geometry={geometry} frustumCulled>
      <shaderMaterial
        ref={matRef}
        vertexShader={VERT}
        fragmentShader={FRAG}
        uniforms={uniforms}
        transparent
        depthWrite={false}
        blending={THREE.AdditiveBlending}
      />
    </lineSegments>
  );
}

/**
 * The decision horizon: the line outcomes separate around.
 *
 * Drawn on its own shader rather than as a flat plane. A plane painted an
 * unbroken rule from one edge of the frame to the other, which read as a
 * stray hairline across the headline instead of as a horizon, and under
 * `bias` it cut straight through the type it was supposed to stay clear of.
 * The fade at both ends and the bias mask are the entire point of it.
 */
const HORIZON_VERT = /* glsl */ `
  varying float vX;
  void main() {
    vX = uv.x;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

const HORIZON_FRAG = /* glsl */ `
  precision highp float;
  uniform vec3 uColor;
  uniform float uResolve;
  uniform float uBias;
  varying float vX;

  void main() {
    // Dissolve into the ground at both ends.
    float ends = smoothstep(0.0, 0.16, vX) * (1.0 - smoothstep(0.84, 1.0, vX));
    // And clear off the left entirely when the composition is biased.
    float side = smoothstep(0.24, 0.62, vX);
    float a = ends * mix(1.0, side, uBias) * (0.16 + uResolve * 0.34);
    gl_FragColor = vec4(uColor, a);
  }
`;

function Horizon({ resolve, bias }: { resolve: number; bias: number }) {
  const matRef = useRef<THREE.ShaderMaterial>(null);
  const uniforms = useMemo(
    () => ({
      uColor: { value: new THREE.Color(hexOf("--accent", "#2f9fd0")) },
      uResolve: { value: resolve },
      uBias: { value: bias },
    }),
    // Colour and bias are fixed for the life of the field; `resolve` is
    // pushed per frame below rather than rebuilding the uniform object.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [bias],
  );

  useFrame(() => {
    if (matRef.current) matRef.current.uniforms.uResolve.value = resolve;
  });

  return (
    <mesh position={[0, 0, 0]}>
      <planeGeometry args={[26, 0.02]} />
      <shaderMaterial
        ref={matRef}
        vertexShader={HORIZON_VERT}
        fragmentShader={HORIZON_FRAG}
        uniforms={uniforms}
        transparent
        depthWrite={false}
      />
    </mesh>
  );
}

export function TraceField({
  resolve = 0,
  bias = 0,
  density = 1,
}: {
  resolve?: number;
  /** 0 keeps the field full-bleed; 1 clears it off the left for type. */
  bias?: number;
  /**
   * Scales both trace count and opacity. A full-bleed hero can carry the whole
   * field; the same field inside a 420px card is a solid mass, because additive
   * blending compounds with how many traces cross a given pixel and a small box
   * crosses far more of them.
   */
  density?: number;
}) {
  const [{ enabled, count }, setConfig] = useState({ enabled: false, count: 430 });

  useEffect(() => {
    let webgl = false;
    try {
      const c = document.createElement("canvas");
      webgl = !!(c.getContext("webgl2") || c.getContext("webgl"));
    } catch {
      webgl = false;
    }
    const motion = window.matchMedia("(prefers-reduced-motion: reduce)");

    const read = () => ({
      enabled: !motion.matches && window.innerWidth >= 640 && webgl,
      count: window.innerWidth < 1280 ? 260 : 430,
    });

    const apply = () => {
      const next = read();
      // Re-read on resize as well as on mount. Checking once meant a window
      // narrowed past the breakpoint kept a full field running behind body
      // copy, which is exactly where it must not be.
      setConfig((prev) =>
        prev.enabled === next.enabled && prev.count === next.count ? prev : next,
      );
    };

    // Capability detection needs `window`, so it cannot happen during render
    // or in a lazy initialiser.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setConfig(read());

    let t: ReturnType<typeof setTimeout>;
    const onResize = () => {
      clearTimeout(t);
      t = setTimeout(apply, 160);
    };
    window.addEventListener("resize", onResize);
    motion.addEventListener("change", apply);
    return () => {
      clearTimeout(t);
      window.removeEventListener("resize", onResize);
      motion.removeEventListener("change", apply);
    };
  }, []);

  // The still fallback is not a degraded experience -- the sections carry the
  // whole argument in text and charts regardless of whether this renders.
  if (!enabled) {
    return (
      <div
        aria-hidden
        className="absolute inset-0 fieldfallback"
      />
    );
  }

  return (
    <div className="absolute inset-0" aria-hidden>
      <Canvas
        camera={{ position: [0, 1.4, 15], fov: 42 }}
        gl={{ antialias: true, powerPreference: "high-performance", alpha: true }}
        dpr={[1, 2]}
        style={{ background: "transparent" }}
      >
        <Field
          resolve={resolve}
          count={Math.round(count * density)}
          bias={bias}
          density={density}
        />
        <Horizon resolve={resolve} bias={bias} />
      </Canvas>
    </div>
  );
}
