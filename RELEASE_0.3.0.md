# 0.3.0 — no stops: scene problems are warnings, failures fall back to a plain photo; props lit by the room

## Never stop the workflow
- **Geometry no longer blocks.** Support/contact/penetration/frame/prop problems are warnings in the status line
  ("Предупреждение (MiniMax исправит): …"). `block_on_fail` is kept for old graphs but has no effect.
- **3DA Native Guides** (`3DA · native H3 guides + gate`) never raises. New input `mode` (1 = plain AIO photo, 2 = 3DA)
  and two new outputs appended at the end: `mode` (effective: 2 = H3 on the 3DA scene, 1 = fallback) and `status`
  (one plain line). `body_guides` is a lazy input: in mode 1 the whole 3DA chain is not executed.
- **Technical failures fall back.** No scene (step 1 not run), a failed location, an empty 3D window/manifest,
  SceneEdits without transforms, SceneExport without a scene, a broken cache — every 3DA node returns placeholders
  plus a flag; Native Guides turns it into `mode = 1` and "3DA не сработал: <причина> — сделано обычное фото".
  Drive your switches with `ImpactSwitch` (`select_on_execution`) from the `mode` output.
- Location: new last output `ok`; an error is not cached (a new cache key after each failure).
  Prop: a bad prop is skipped with a warning instead of an error. Native Composite: returns the H3 frame on any
  problem; an outside-composite check failure is a warning in its metrics.
- Old test props `binder_notebook`, `ceramic_vase` are hidden from the prop list (`"hidden": true` in prop.json hides any).

## Reliable caches
- `plate.json` is written atomically (temp file + replace) with numpy-safe values; the auto profile is copied atomically,
  `profile.json` last.
- A broken or partial AUTO location (unreadable/incomplete plate.json or profile.json, leftovers of an unfinished
  profile) is moved aside (`<id>.broken-<time>`, never deleted) and prepared again from the photo.

## Props in the room light (with shadows)
- The prop is rendered with Cycles in the same light as the body playblast: the key light (plate.json/spec `light`,
  else the brightest area of the photo), ambient in its colour and the photo re-emitted as bounce light.
  Own shadows of the prop and the body's shadow on it come from this pass.
- A second pass renders only the prop's **cast shadow** on the room and on the body (shadow catchers) —
  `props_shadow.png`. It is baked into `init`, `playblast` and the restoration layer `prop_repair`, and applied to the
  room in the final composite.
- After H3 only the fine texture of the rigid prop is restored; colour and light come from H3. Where H3 drew something
  else in place of the prop, the rendered prop is kept, so its 3D shape is never redrawn.
- `--neutral-light` keeps the pre-0.3.0 neutral EEVEE prop render for diagnostics.

Example workflows are unchanged. AIO graphs built for 0.2.x keep working (new inputs/outputs are optional/appended).
