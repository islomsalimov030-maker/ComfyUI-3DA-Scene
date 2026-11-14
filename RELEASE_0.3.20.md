# 0.3.20 — the character's own body shape

- New node **3DA · Персонаж: телосложение** (`3DABodyProfile`): 1–6 photos of the character (batches are split) →
  SAM 3D Body → median MHR shape/scale. With `write` on, it is saved into `body_json` (default
  `characters/stacy/body.json`); the previous file is kept as `body.json.bak_<date>`. Run once per character.
- **3DA Place Pose**: when `body.json` has a shape, only the pose and hands come from the pose photo; the body is
  built with the character's shape (cached per pose photo and shape). Without a shape it works as before and the
  report warns that the body is taken from the pose photo.
- The mannequin is the MHR body: no hair and no clothes, with fingers, toes and a face.
- H3 scene prompt: natural hands and feet (five fingers, five toes or the described footwear); the mannequin's smooth
  face and bald head are never copied.
