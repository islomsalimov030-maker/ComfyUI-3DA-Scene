# 0.3.7 — one shadow rule for character and prop; more room for hair and shadows

- **3D-control → shadows**: «Тени от H3» (default) — H3 draws the shadows of the character and the prop, no render
  shadow is pasted; «Тени от рендера» — the Cycles shadow of the character and the prop (room light) is baked into the
  H3 guide and composited on the room after H3. `3DA Place Pose`: optional `render_shadows` input.
- **Strength band** around the body (where H3 may draw hair and shadows) = cell × 2 × raster/1344 — wider than before and
  the same relative width at 2 MP. `strength_margin` in config.yaml (1.0 = previous band).
- 3D-control `prop_render` and `shadows` are optional inputs: an old browser tab no longer fails with
  «Required input missing».
