# 0.3.19 — frame prompt labels for AIO V3

- The frame prompt is written with AIO labels: <Picture 2> room, <Picture 3> pose, <Picture 4…> extra photos.
  In the 3DA H3 branch they now become: room → <Picture 1> (plate), pose → <Video 1> (3D render),
  extra photo k → <Picture k−1>. Before, every label ≥ 2 was shifted by +1 and pointed to the wrong image.
- `user_refs` = number of extra photos (Photo 4…) connected to the 3DA H3 node after the plate and identity.
- The pose photo should not be a 3DA H3 reference (the pose comes from the 3D render) — unwire it from the 3DA H3 node in your workflow.
