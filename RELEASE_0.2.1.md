# 0.2.1 — hotfix: new locations failed to prepare

`plate_geometry.py` (0.2.0 floor detection) put numpy booleans into the floor report, and `json.dump` of `plate.json`
raised "Object of type bool is not JSON serializable". The Location node then returned an error and every following
node failed ("location profile.json: required file is missing", "Сначала выполните группу 1").

- The report values are plain Python booleans; `plate.json` is written with a numpy-scalar fallback.
- Existing locations are unaffected. A location that failed under 0.2.0 is prepared again on the next run (nothing was cached for it).

Test: `tests/test_plate_frame.py` checks that every floor report is plain JSON (fails on 0.2.0, passes on 0.2.1).
