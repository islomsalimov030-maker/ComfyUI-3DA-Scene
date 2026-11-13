# 0.2.0 — any floor, any frame, panel megapixels, clear rejections (server restart needed)

## Floors and occlusion (part 1)

- 3D window occluder = the room geometry of the location (`plate_mesh.obj`, the same mesh the renderer occludes with), depth-only, 1 cm farther from the camera; the infinite y = −0.01 plane of 0.1.10 is gone. New module `web/roomGeometry.js`.
- 3D window labels match the server checks: "inside the floor / furniture / wall by N cm", "pressed in — soft surfaces only", "no support — hangs N cm" (only when the support is visible in the frame), "no part of the body in the frame" (partly out of frame is allowed), body scale is a warning; an old scene without `room_mesh` asks to run group 1 again.
- Scene export (`scene_edit.export`): `input/3d/3da_scene/room_<sha16>.obj`, `room_mesh` and `raster` in the setup.
- Locations without a visible floor (`plate_geometry`): floor = the lowest level with no more than 3 % of the scene below it (a bed filling the bottom of the frame is not the floor); no floor → `floor.visible=false`, camera height prior 1.5 m; a thin floor strip no longer tilts the normal. Old location caches are unchanged.
- Penetration (`pose_fit.penetration` and the 3D window alike): 35 cm cap along the surface normal; below the visible floor — full depth without the cap.

## Part 2

- **Scene without props.** `3DAProp.prop_id` has "(нет предмета)" as its first choice: the chain passes through unchanged, so the same workflow works with only the character or with any registered prop.
- **Panel megapixels.** `3DAPlacePose` has an optional `megapixels` input (0 = 1344×768 as before). The render before H3, the guides, the prop layers and therefore the native H3 width/height use a raster of that area with the plate framing (multiples of 16). The full-resolution room photo is kept once per location (`plate_photo.png`); older locations fall back to `plate_rgb.png`. The 3D window stays 1344×768 (normalised intrinsics, same framing).
- **Clothing is not a 3DA task.** The hard-coded outfit (T-shirt / jeans / sneakers) is removed from the scene prompt. `3DANativeGuides` takes the user's frame prompt (`frame_prompt`) and merges it into the scene prompt: plain text goes into `detailed_description` after `[Shot 1]`; a six-section prompt is merged section by section. Picture numbering in the native branch: 1 = room plate, 2 = identity reference, 3–5 = the user's `ref_image_1..3`; "Image N" / `<Picture N>` of the user's prompt are renumbered accordingly and get definitions/retention lines.
- **Frame is a warning.** A body or prop partly (a prop even fully) out of the frame no longer blocks H3 (`warnings` in the report and in the status line); only a body with no visible point is rejected (`out_of_frame`). Auto placement still prefers fully framed candidates.
- **Readable rejections.** `GeometryRejected` and the first status line give Russian reasons with numbers: "Персонаж: внутри пола на 15 см (допуск 3 см)", "нет опоры — висит на 12 см", "Предмет 1 (stick): нет опоры — висит на 51 см", "…пересекает тело". An empty manifest / 3D window asks to run group 1 first.
- **Server prop checks** (`prop_checks.py`, the 3D window rules): penetration p95 > 3 cm (`prop_penetration`); a prop not touching the body that hangs > 5 cm over a visible horizontal surface (`prop_floating`; support not visible → not checked); a mesh overlap without depth (resting contact) is no longer a rejection.
- **3D window and server aligned on small objects at a contact:** the window measures depth along the normal of the nearest visible surface point (±3 z-buffer cells), as the server does, instead of the cell under the pixel.

Not in this release: body shape from the character's `body.json` (SAM 3D Body shape/scale) — planned for 0.2.1.

No new dependencies, models or weights. Tests in ComfyUI-Farm: `tests/test_3da_v020.py`, `tests/test_prop_validation_10a.py`, `tests/test_3da_aio.py`, `tests/web/test_room_geometry.mjs`.
