"""E10c: one-off export of the MPFB mannequin rig for the 3D window (web/mpfbRig.bin.gz).

The window used to deform the static GLB with geodesic weights, the server poses the real MPFB armature in Blender.
This script dumps what the window needs to repeat the server: rest bone matrices (armature space), the bone hierarchy,
the rest mesh (same topology and vertex order as render/pose/body_render.obj: masks + subsurf 1 as in
pose_tools.pose_mannequin_to_mhr) and its skin weights (top 8 bones per vertex, normalised).

    blender -b --factory-startup --python tools/scene3d/mpfb_rig_dump.py -- <mannequin.blend> <out.bin.gz>

Binary layout (little endian, then gzip): u32 header_len | header JSON (utf-8, padded with spaces to 8 bytes) |
sections listed in header["sections"] (name -> [offset, dtype, count]; offsets from the end of the header).
"""
import gzip
import hashlib
import json
from pathlib import Path
import struct
import sys

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]

TOP = 8      # top 4 left up to 2.7 cm vs Blender around the face/fingers; top 8 keeps the max error under 1 cm
SUBSURF = 1
LICENSE = ("MPFB2 / MakeHuman: the mannequin mesh, rig and skin weights are MakeHuman system assets (CC0 1.0); "
           "the MPFB add-on code (GPL-3.0) is not included, only data derived from the assets.")


def load(mannequin):
    """Open the mannequin, drop hair/clothes/helpers exactly as the server does, add the same subsurf."""
    import bpy
    import pose_tools
    bpy.ops.wm.open_mainfile(filepath=str(mannequin))
    rig = bpy.data.objects['stacy_rig']
    body = bpy.data.objects['stacy_body']
    removed = pose_tools.prepare_mannequin(rig, body)
    sub = body.modifiers.get('render_smooth') or body.modifiers.new('render_smooth', 'SUBSURF')
    sub.levels = sub.render_levels = SUBSURF
    return rig, body, removed


def evaluated(body):
    """Evaluated body mesh (new datablock, object space of the body; vertex groups kept by the subsurf)."""
    import bpy
    dg = bpy.context.evaluated_depsgraph_get()
    ev = body.evaluated_get(dg)
    mesh = bpy.data.meshes.new_from_object(ev, depsgraph=dg)
    mesh.calc_loop_triangles()
    return mesh


def rest_state(rig, body, top=TOP):
    """Everything the window needs, as numpy arrays (see module docstring)."""
    import bpy
    import numpy as np
    from mathutils import Matrix, Vector
    for pb in rig.pose.bones:
        pb.rotation_mode = 'QUATERNION'
        pb.matrix_basis = Matrix.Identity(4)
    bpy.context.view_layer.update()
    bones = list(rig.data.bones)
    index = {b.name: i for i, b in enumerate(bones)}
    assert all(b.parent is None or index[b.parent.name] < i for i, b in enumerate(bones)), 'parents must come first'
    parent = np.array([index[b.parent.name] if b.parent else -1 for b in bones], np.int32)
    rest = np.array([[list(r) for r in b.matrix_local] for b in bones], np.float64)       # armature space
    length = np.array([b.length for b in bones], np.float64)
    mesh = evaluated(body)
    world = np.array([tuple(body.matrix_world @ v.co) for v in mesh.vertices], np.float64)
    rig_inv = rig.matrix_world.inverted()
    v_arm = np.array([tuple(rig_inv @ Vector(p)) for p in world], np.float64)
    names = {g.index: g.name for g in body.vertex_groups}
    n = len(mesh.vertices)
    idx = np.zeros((n, top), np.uint8)
    wgt = np.zeros((n, top), np.float64)
    dropped = np.zeros(n)
    for v in mesh.vertices:
        ws = sorted(((g.weight, index[names[g.group]]) for g in v.groups
                     if names.get(g.group) in index and g.weight > 0), reverse=True)
        total = sum(w for w, _ in ws)
        kept = ws[:top]
        s = sum(w for w, _ in kept)
        if total <= 0:
            raise ValueError(f'vertex {v.index} has no bone weights')
        dropped[v.index] = 1 - s / total
        for k, (w, b) in enumerate(kept):
            idx[v.index, k], wgt[v.index, k] = b, w / s
    tris = np.array([t.vertices[:] for t in mesh.loop_triangles], np.uint32)
    return dict(names=[b.name for b in bones], parent=parent, rest=rest, length=length, v_arm=v_arm, idx=idx, wgt=wgt,
                tris=tris, dropped=dropped, rig_world=np.array([list(r) for r in rig.matrix_world], np.float64),
                body_world=np.array([list(r) for r in body.matrix_world], np.float64))


def write_asset(state, path, source, top=TOP):
    import numpy as np
    sections, blobs, offset = {}, [], 0

    def add(name, array, dtype):
        nonlocal offset
        data = np.ascontiguousarray(array, dtype=dtype).tobytes()
        sections[name] = [offset, np.dtype(dtype).str, int(np.size(array))]
        blobs.append(data)
        offset += len(data)
    add('rest', state['rest'], '<f8')
    add('length', state['length'], '<f8')
    add('parent', state['parent'], '<i4')
    add('v_arm', state['v_arm'], '<f4')
    add('idx', state['idx'], '<u1')
    add('wgt', np.round(state['wgt'] * 65535), '<u2')
    add('tris', state['tris'], '<u4')
    header = dict(format='3da-mpfb-rig-1', license=LICENSE, source=source, top=top, subsurf=SUBSURF,
                  n_vertices=int(len(state['v_arm'])), n_triangles=int(len(state['tris'])), bones=state['names'],
                  rig_world=state['rig_world'].tolist(), body_world=state['body_world'].tolist(), sections=sections)
    raw = json.dumps(header, separators=(',', ':')).encode('utf-8')
    raw += b' ' * (-len(raw) % 8)
    payload = struct.pack('<I', len(raw)) + raw + b''.join(blobs)
    Path(path).write_bytes(gzip.compress(payload, 9, mtime=0))
    return len(payload)


def main():
    import bpy
    argv = sys.argv[sys.argv.index('--') + 1:]
    mannequin, out = Path(argv[0]), Path(argv[1])
    top = int(argv[2]) if len(argv) > 2 else TOP
    rig, body, removed = load(mannequin)
    state = rest_state(rig, body, top)
    source = dict(mannequin=mannequin.name, mannequin_sha256=hashlib.sha256(mannequin.read_bytes()).hexdigest(),
                  blender=bpy.app.version_string, removed=removed)
    size = write_asset(state, out, source, top)
    print('E10C_DUMP', json.dumps(dict(out=str(out), raw_bytes=size, gz_bytes=out.stat().st_size,
                                     vertices=len(state['v_arm']), triangles=len(state['tris']),
                                     bones=len(state['names']), dropped_max=float(state['dropped'].max()),
                                     dropped_mean=float(state['dropped'].mean()))))


if __name__ == '__main__':
    main()
