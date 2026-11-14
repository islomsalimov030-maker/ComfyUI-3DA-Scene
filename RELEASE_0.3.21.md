# 0.3.21 — body shape from measurements

- SAM 3D Body pulls the shape estimated from photos towards an average body. New node **3DA · Персонаж: мерки**
  (`3DABodyMeasure`): fits the MHR shape to bust / waist / hips / thigh girths and crotch height (cm, relative to the
  character height), starting from the current shape in `body_json`. With `write` on, it saves the shape and
  `measurements_cm` into `body.json` (backup kept). Defaults are the measurements of Stacy's MPFB mannequin (162 cm).
- The status shows target → achieved for each measurement and the largest deviation.
