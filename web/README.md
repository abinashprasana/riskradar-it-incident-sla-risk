# KAIROS — presentation layer

The web front end for the study in the parent repository. Static Next.js export,
no server, no database, no analytics.

```bash
npm install
npm run dev      # local
npm run build    # static export to out/
npx serve out    # preview the built site
```

## Where the numbers come from

Nothing on this site is typed by hand. Three JSON files under `public/data/` are
generated from the study's own artifacts:

| File | Contents |
|---|---|
| `summary.json` | headline figures, split configuration, data-quality counts |
| `per_k.json` | every experimental arm's per-prefix-length curve |
| `explorer_sample.json` | 478 real scored test incidents for the k-slider |

Regenerate them after any pipeline change, from the repository root:

```bash
python scripts/run_all.py          # includes the export step
# or just the export, if the arms are already current:
python scripts/export_web_data.py
```

`export_web_data.py` reads `artifacts/` and refits nothing, so the site cannot
drift from what the experiments reported. The explorer's scores come from
`artifacts/uci_servicenow/agg/scored_test.csv`, which requires the headline arm
to have been run with `--dump-scored` (`run_all.py` passes it).

They are imported directly rather than fetched, so the page has no loading state
and no network failure mode.

## Design decisions worth knowing before editing

**Colour is validated, not chosen.** The six chart series in `globals.css` are
the output of the data-viz validator and pass all six computable checks in both
light and dark. If you change a series hue, re-run it:

```bash
node <dataviz-skill>/scripts/validate_palette.js \
  "#2f9fd0,#d55181,#c97a22,#199e70,#8478e0,#e05c58" \
  --mode dark --surface "#0b1016"
```

Tritan separation sits at 6.4, inside the 6–8 floor band, which is permissible
only when colour is not the sole carrier of identity. Every series therefore
also owns a dash pattern and a direct label. Do not remove those.

Status colours (`--status-good`, `--status-warn`, `--status-critical`) are
reserved for risk bands. Never reuse them as a series hue.

**Reveal-on-scroll fails open.** `useInView` renders content visible by default
and only hides an element after JS has confirmed it is below the fold, with a
three-second backstop. Content is never worth risking for an entrance animation.

**No `mix-blend-mode` on full-bleed layers, and no `backdrop-filter` on the
sticky header.** Both promote the document to a single composited layer, which
blanks the paint entirely on some GPU and driver combinations. This was a real
bug here, not a hypothetical.

**Motion.** Enter and exit use ease-out; interactive transitions stay under
300ms; transitions rather than keyframes for anything rapidly re-triggerable;
nothing animates from `scale(0)`; only `transform` and `opacity` are animated.
Every animation should be explainable in one sentence.

`prefers-reduced-motion` disables the 3D field and all scroll choreography. The
field also does not mount below 640px or without WebGL. The page carries its
whole argument in text and charts regardless.

## Deploy

`npm run build` produces `out/`, servable anywhere. Vercel needs no
configuration; Netlify and GitHub Pages work from the same directory.

CI (`.github/workflows/web.yml`) runs `tsc --noEmit`, `next lint` and the build
on every push touching `web/`.
