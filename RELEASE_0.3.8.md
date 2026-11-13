# 0.3.8 — shadows are drawn inside the H3 area, not cut by its edge

- The area where H3 may draw now includes the zone of expected shadows of the character and the prop (from the Cycles
  render in the room light), with the same soft edge as around the body: shadows are no longer cut by a hard edge.
  `shadow_strength` in config.yaml (0.7).
- «Тени от рендера»: the render shadow is softer and lighter (`render_shadow_opacity`, 0.6) and goes to H3 as a hint in
  the guide; H3 draws it — it is not multiplied over the result a second time.
- Fixed: tiny render-shadow values no longer extend the composite area to the whole frame.
