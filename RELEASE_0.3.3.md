# 0.3.3 — «3D-контроль» runs the steps and really switches 3DA off

- **3DA · 3D-контроль**: new `step` field — «Подготовка сцены» / «Генерация». It switches the two step groups
  (`▶ Подготовка сцены` / `▶ Генерация`, old names `▶ Шаг 1…` / `▶ Шаг 2…` also work) — no separate group switch.
- 3D control OFF: character/prop/step are dimmed and all 3DA nodes (location, poses, props, scene edits/export, the
  3D window, 3DA previews) are muted, so nothing of 3DA runs; the photo is the plain AIO one. Modes are set when the
  fields change and again right before every Run.
- `3DA Native Guides`: `body_guides` is optional (lazy); `3DA Native Composite`: scene maps are optional — a muted 3DA
  chain never fails graph validation.
