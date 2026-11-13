# 0.3.17 — character in frame when the room has no support surface

- Fallback placement (no floor or furniture found, e.g. a close-up of a wall): the whole character is now centred in
  the frame, at the distance where it fits with a margin but in front of the wall. Before, only the legs were visible,
  the gizmo arrows were outside the 3D window (the character could not be moved) and the prop was out of frame.
- 3D window: «В кадр» also works for the character — its centre goes to the frame centre at the same depth.
