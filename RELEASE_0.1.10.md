# 0.1.10 (frontend only, no server restart)

## The 3D window shows the floor (13.11)

Problem found on a new location and pose (prompt ac98ba5f): the viewport draws the body and props over a flat room photo, while the renderer before H3 occludes them by the room geometry. With a camera looking down, moving the body 40 cm below the floor and scaling it ×1.26 looked like a normal placement in the 3D window; the render showed only the head and shoulders above the floor and the final gate rejected the scene (out_of_frame). The transforms themselves were applied exactly (the head projects to the same pixels in both).

- A depth-only floor plane (y = −0.01 m, the fitted room floor) is added to the viewport scene. It writes depth only, so the photo stays visible and anything below the floor is hidden, like in the render. It is removed with the scene adapter and is not part of the exported scene or of the manifest.
- A status label in the 3DA toolbar: "Пол и рост: ок" (green) or, in red, "<объект>: ниже пола на N см" (beyond 6 cm, placement.yaml `pen_soft_m`) and "Персонаж: масштаб ×S — рост изменится так же".
- The scale mode is labelled "Масштаб (равномерно)" for props too (uniform since 0.1.9).

No Python, model, weight or dependency changes. Test: `node tests/web/test_scene_floor.mjs` in ComfyUI-Farm.
