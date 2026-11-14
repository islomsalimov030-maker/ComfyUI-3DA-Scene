# 0.3.24 — 3D window works with service nodes in a subgraph

- «3DA · Подготовить сцену» inside a subgraph (AIO_manual_V3.2.8) now hands the new scene to the 3D window
  (the api `executed` event is used; onExecuted of an inner node is not called).
- `app.graph.subgraphs` is a Map in the current LiteGraph; the extension now reads both Map and object.
