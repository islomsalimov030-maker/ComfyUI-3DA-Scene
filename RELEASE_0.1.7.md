# 0.1.7 candidate

The 0.1.6 cold run reached SAM hand refinement and failed when an internal float32 camera estimate was concatenated with half-precision pose estimates and passed into a half-precision linear layer.

- Add a server-only autocast context for the existing SAM prediction call, using the loaded backbone dtype and the patcher's load device. This covers internal refinement tensors, not only the input image.
- Keep hand refinement enabled. Do not replace weights, promote the full model to float32, change core files or alter the PC path.
- Persist full redacted stage stdout/stderr alongside each request so a downstream SceneExport error does not hide the original traceback.

Local tests reproduce the mixed-input linear failure, verify fp16/bf16 correction, unchanged model weights and no leaked autocast state. Actual server success remains unverified.

No new nodes, dependencies, models or weights. Publication, installation and another restart require explicit approval; the single 0.1.6 restart was already used.
