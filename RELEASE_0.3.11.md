# 0.3.11 — test: clean shadows

- 3D-control → `clean_shadows` (test, off by default): in the shadow zone (outside the band around the character and
  the prop) the result is the room photo × the blurred H3/photo luminance ratio (darkening only). The shadow shape and
  softness come from H3, the floor texture from the photo — no noise. Near the character and the prop nothing changes.
