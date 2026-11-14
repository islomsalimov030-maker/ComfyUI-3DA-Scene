# 0.3.22 — Stacy's own MPFB mannequin in the render

- **3DA Place Pose**: the render body (playblast, init, silhouette, labels, props) is Stacy's MPFB mannequin
  (fingers, toes, female shape, face, eyes; no hair, no clothes) posed onto the placed MHR body: joint directions
  incl. torso twist, clavicles, finger chains, head/hand/foot frames; moved onto the MHR torso; vertical contact so
  the mannequin keeps the MHR body's clearance where it touches the room (feet on the floor, seat on the seat).
  Placement, contacts and checks still use the MHR body. Any problem -> warning and the MHR body as before.
- `config.yaml` `body_model: mhr` switches back to the MHR mesh (default `mpfb`). The mannequin file is
  `mannequin` (assets/mannequin_162.blend); hair, clothes and their masks are removed at render time.
- 3D window shows the same MPFB body.
- **«В кадр»** for the character: horizontally to the frame centre, feet on the floor (no floor found -> height
  unchanged). Before, the body was centred on the camera axis in height too and hung 47–87 cm above the floor.
