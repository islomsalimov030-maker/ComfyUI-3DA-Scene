"""Candidate-only extra passes on the saved render scene; invoked through threeda_blender."""
import argparse
import sys
from pathlib import Path

import bpy

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
from render_h3_maps import mask_material, render_pass


def main():
    args = sys.argv[sys.argv.index('--') + 1:]
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--expand', type=int, required=True)
    a = parser.parse_args(args)
    if a.expand < 3 or a.expand % 2 != 1:
        raise ValueError('expand must be odd and >= 3')
    bpy.ops.wm.open_mainfile(filepath=a.scene)
    sc = bpy.context.scene
    body = bpy.data.objects.get('body_v2')
    if body is None or body.hide_render:
        raise ValueError('render body is unavailable')
    width = sc.render.resolution_x * sc.render.resolution_percentage // 100
    height = sc.render.resolution_y * sc.render.resolution_percentage // 100
    if width <= 0 or height <= 0:
        raise ValueError('invalid render resolution')
    n = a.expand
    sc.render.resolution_x, sc.render.resolution_y = width * n, height * n
    sc.render.resolution_percentage = 100
    if sc.camera.data.type == 'PERSP':
        sc.camera.data.lens /= n
    elif sc.camera.data.type == 'ORTHO':
        sc.camera.data.ortho_scale *= n
    else:
        raise ValueError('unsupported camera projection')
    sc.camera.data.shift_x /= n
    sc.camera.data.shift_y /= n
    sc.eevee.taa_render_samples = 1
    sc.render.film_transparent = False
    sc.world = bpy.data.worlds.new('candidate_black_world')
    sc.world.use_nodes = True
    sc.world.node_tree.nodes.get('Background').inputs['Color'].default_value = (0, 0, 0, 1)
    sc.view_settings.view_transform = 'Standard'
    for ob in sc.objects:
        ob.pass_index = 1 if ob == body else 0
    hidden = [(ob, ob.hide_render) for ob in sc.objects if ob.type == 'MESH' and ob != body]
    output = Path(a.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    full = output.with_name('candidate_visual_body.png')
    visible = output.with_name('candidate_visual_visible.png')
    for ob, _ in hidden:
        ob.hide_render = True
    render_pass(sc, str(full), engine='BLENDER_EEVEE', override=mask_material())
    for ob, old in hidden:
        ob.hide_render = old
    render_pass(sc, str(visible), engine='BLENDER_EEVEE', override=mask_material())


if __name__ == '__main__':
    main()
