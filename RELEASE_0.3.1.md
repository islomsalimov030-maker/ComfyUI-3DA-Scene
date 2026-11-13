# 0.3.1 — scene preparation no longer falls back on numpy values

- Some poses put a numpy bool (named `bool` in numpy 2) into the scene report. The CLI could not write its result
  ("Object of type bool is not JSON serializable"), so step 1 fell back: the 3D window was not updated and the final
  made a plain photo. The CLI now writes numpy scalars/arrays as plain JSON values.
