# 0.3.15 — a new location or pose no longer leaves the old scene in the 3D window

- When the geometry checks found no support (gap = inf), scene preparation failed with «Out of range float values are
  not JSON compliant: inf»: the 3D scene was not exported and the 3D window kept the previous pose and room.
  Not-found distances are now written as ±1e9 (NaN as null) and the scene is prepared; the problem stays a warning.
- Russian warnings show «?» instead of a huge number for a not-found distance.
