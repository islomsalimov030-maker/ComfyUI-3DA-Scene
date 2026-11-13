# 0.3.2 — Export/Run never blocked by the 3D window; «3D-контроль» node

- The 3D window no longer throws "3DA: объекты сцены ещё не загружены" while saving its state. That throw blocked
  Workflow → Export and Run (the frontend then reported "Cannot read properties of undefined (reading 'workflow')").
  If the scene is not loaded yet, the last captured transforms are kept (or none — 3DA then falls back, no error).
- New node **3DA · 3D-контроль (пульт)** (`3DAControl`): `control_3d` on/off, `character` (Стейси), `prop`.
  Outputs `mode` (2 = 3DA, 1 = plain AIO photo, pose drawn by MiniMax from the prompt), `prop` (list type — connects
  to the `prop_id` of «3DA Prop») and `character`. With the control off the character/prop fields are dimmed and
  inactive. A prop missing on the server is not a validation error — it is simply not placed.
