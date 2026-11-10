# 3DA Scene (ComfyUI custom nodes)

Manual 3D placement of a body pose (and optional props) into a prepared location photo.
The nodes fit the pose to the location geometry, render a Blender playblast / init / strength
guides and hand them to an image model graph (example: `examples/3DA_scene_server_v1.json`).

Nodes: `3DALocation`, `3DAPlacePose`, `3DAProp`, `3DASceneEdits`, `3DASceneExport`,
`3DAH3Remote`, `3DANativeGuides`, `3DANativeComposite`.

## Install
ComfyUI-Manager → search "3DA Scene", or `comfy node install comfyui-3da-scene`.
`install.py` installs the `bpy` wheel on Python 3.13 (no deps, numpy/torch untouched) and OpenCV
only when no `cv2` is present. On other Python versions set `blender_path` (Blender 5.x).

## Data
Per-user data lives in `data_root` (default `<ComfyUI>/user/default/3da_data`):
`locations/<id>/profile.json …`, `characters/<id>/body.json`, `props/<id>/prop.json`,
`assets/<mannequin>.blend`, `cache/`. Upload a prepared tree with
`python scripts/push_data.py --source <tree> --url <ComfyUI URL> --location <id> --character <id>`
(credentials only via `COMFY_USER` / `COMFY_PASS`). No data, images or meshes ship with the package.

Geometry for new references uses ComfyUI core nodes (`LoadMoGeModel`/`MoGeInference`,
`SAM3DBody_*`); their weights must be present in `models/`. Precomputed `cache/pose_ref` can be uploaded instead.

## Licenses
- This package: GPL-3.0 (see LICENSE).
- `bpy` (Blender as a Python module): GPL-3.0.
- OpenCV: Apache-2.0. SciPy: BSD-3. HTTPX: BSD-3. PyYAML: MIT.
- Model weights are not included; check the licenses of the MoGe, SAM 3D Body and any image/video
  model you use with the example graph before commercial use.
