# 0.3.6 — the prop is drawn by H3 (or pasted exactly), one rule for shadows

- **3D-control → prop_render**: «H3 рисует предмет» (default) — body and prop go to H3 as one scene (prop rendered
  without the Cycles shadow, strength covers body and prop, the prompt names the prop, so the prop RefMod is used);
  H3 draws both and both shadows; no paste afterwards. «Точная вклейка» — previous behaviour: H3 draws the body,
  the rendered prop and its Cycles shadow are pasted.
- `3DA Place Pose`: optional `prop_by_h3` input (wire it from 3D-control output `prop_by_h3`).
- With «H3 рисует предмет» the `3DA Prop` `prop_strength` sets how much H3 may redraw the prop (0.6 recommended).
