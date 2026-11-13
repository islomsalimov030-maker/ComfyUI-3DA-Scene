# 0.3.12 — clean shadows that actually apply; margin around the prop

- `clean_shadows` (test): only the person and the prop are excluded, not the whole strength band around the body.
  In 0.3.11 the shadow zone lay inside that band, so the effect was ~0. Now, over the whole H3 area outside the person
  and the prop, the result is the room photo × the blurred H3/photo luminance ratio (darkening only): shadow shape from
  H3, floor texture from the photo, no mosaic.
- New optional input `person_mask` on «3DA · final room + rigid props»: a person mask of the H3 frame
  (e.g. SAM3 «person»). Without it the render silhouette with a 2 % margin is used.
- Final composite metrics: `clean_shadows` {applied, person_source, area_px, person_px, keep_px, weight_px, zone_px,
  darkened_px, mean_ratio_in_shadow}.
- The band around the prop now has the body strength (like the band around the body); the prop itself keeps
  `prop_strength`, so its shape stays.
