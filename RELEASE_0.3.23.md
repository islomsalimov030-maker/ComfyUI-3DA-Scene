# 0.3.23 — choose the render body in «3D-контроль»

- **3DA · 3D-контроль**: new field `body` — «Стейси MPFB (пальцы, лицо)» (default) or «MHR (SAM 3D Body)».
  It goes with `settings` to Place Pose; the panel choice wins over `config.yaml` `body_model`.
- Placement and checks always use the MHR body; the 3D window shows the MPFB body only when it was rendered.
- 3D-control muting (3D off / Подготовка / Генерация) and the 3D-window hand-off also work when the service 3DA
  nodes are inside a subgraph (workflow AIO_manual_V3.2.8 keeps them out of the artist's way).
