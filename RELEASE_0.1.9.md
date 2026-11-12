# 0.1.9

Reopening a workflow that contains ImpactSwitch (for example after switching ComfyUI tabs) failed with "t.findInputSlot is not a function" from Impact Pack's legacy reconnect and locked the UI.

- web/extension.js: beforeConfigureGraph normalizes graphs that contain 3DASceneExport before core loads them: drops floatingLinks: null and restores ImpactSwitch control slots (select, sel_mode) ahead of the data slots, remapping link target slots, in the root graph and in subgraphs.
- web/cameraLock.js: the room camera preset also locks FOV (separate core setter), so the default 35° no longer overrides it.
- web/sceneSelection.js: uniform scaling now applies to props as well as the body; no per-axis stretching.

No new nodes, dependencies, models or weights.
