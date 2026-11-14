# 0.3.25 — «3D-контроль»: main and fine settings

- Knob order: control_3d, character, step, prop, body_margin; then the fine settings (⚙): body, shadows, shadow_strength, shadow_opacity, clean_shadows — under a thin «тонкие настройки» divider. Outputs unchanged.
- Workflows saved before 0.3.25 are migrated once on load (values matched by name from the old order, `properties.ng3da_layout = 2`).
- clean_shadows tooltip no longer says «Тест».
