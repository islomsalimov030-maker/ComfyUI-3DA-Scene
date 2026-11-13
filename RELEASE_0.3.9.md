# 0.3.9 — shadow and margin knobs on the 3D-control; the prop is always drawn by H3

- **3D-control** (all optional): «Тени» (H3 / render hint), «body_margin» (room around the body for hair, 2.0),
  «shadow_strength» (how much H3 may change the shadow zone, 0.7), «shadow_opacity» (render shadow density, 0.6).
  One output `settings` → `3DA Place Pose` input `settings`. config.yaml values remain a fallback.
- The prop is always drawn by H3 together with the body (`prop_render` and `prop_by_h3` removed).
- Changing these knobs re-runs only the final scene check (step «Генерация»), not the scene preparation.
