"""CPU tests for render-mask measurements and candidate-worker integration."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools/scene3d/posev2'))
import candidate_visual as V
import candidates as C
import candidates_cli as CLI


def test_masks_occlusion_and_cropping():
    body = np.zeros((30, 30), np.uint8)
    body[8:22, 8:22] = 255
    visible = body.copy()
    visible[8:22, 10:15] = 0
    m = V.measure(body, visible, (10, 10, 20, 20))
    assert m['height_fraction'] == 1.4
    assert m['cropped_edges'] == ['left', 'right', 'top', 'bottom']
    assert m['body_pixels_in_frame'] == 100 and m['visible_pixels'] == 50
    assert m['visible_fraction'] == 0.5
    _, comp = C.score({'label': 'floor'}, 'ok', {'visual_metrics': m}, 'stand', C.load_config())
    assert comp['visibility'] == 0.833 and comp['frame_height'] == 0 and comp['crop'] == 0


def test_body_is_outside_frame_or_exceeds_canvas():
    body = np.zeros((20, 20), np.uint8)
    body[2:4, 2:4] = 255
    m = V.measure(body, body, (5, 5, 15, 15))
    assert m['visible_fraction'] == 0 and m['cropped_edges'] == ['left', 'top']
    body[0, 0] = 255
    with pytest.raises(ValueError, match='exceeds expanded canvas'):
        V.measure(body, body, (5, 5, 15, 15))
    with pytest.raises(ValueError, match='empty'):
        V.measure(np.zeros_like(body), body, (5, 5, 15, 15))


def test_worker_invokes_render_stage_and_propagates(monkeypatch, tmp_path):
    import threeda_blender
    directory = tmp_path / 'render' / 'pose'
    directory.mkdir(parents=True)
    (directory / 'scene.blend').write_bytes(b'scene')
    result = {'silhouette': str(directory / 'silhouette.png'), 'report': {'status': 'ok'}}
    captured = {}
    monkeypatch.setattr(CLI, 'patch', lambda *args: None)
    monkeypatch.setattr(CLI, 'add_support_depth', lambda request, config, result: result)
    monkeypatch.setattr(CLI.pack_cli, 'do_place', lambda request, config: result)
    def command(config, script, args):
        captured['script'], captured['args'] = script, args
        return args
    def stage(args, label):
        from PIL import Image
        assert label == 'candidate silhouette passes'
        output = Path(args[args.index('--out') + 1])
        body = np.zeros((30, 30), np.uint8)
        body[12:17, 12:16] = 255
        vis = body.copy()
        vis[12:17, 13:16] = 0
        Image.fromarray(body).save(output.with_name('candidate_visual_body.png'))
        Image.fromarray(vis).save(output.with_name('candidate_visual_visible.png'))
    monkeypatch.setattr(threeda_blender, 'command', command)
    monkeypatch.setattr(CLI.pack_cli, 'run_stage', stage)
    out = CLI.geometry_worker({}, {})
    assert Path(captured['script']).name == 'candidate_visual_blender.py'
    assert Path(captured['args'][1]) == directory / 'scene.blend'
    assert C.visibility_fraction(out['report']) == 0.25
    assert C.score({'label': 'floor'}, 'ok', out['report'], 'stand', C.load_config())[1]['visibility'] == .417


def test_missing_render_metrics_are_explicit(monkeypatch, tmp_path):
    result = {'silhouette': str(tmp_path / 'silhouette.png'), 'report': {}}
    CLI.add_visual_metrics({}, result)
    assert 'visual_metrics_error' in result['report']
    assert C.visibility_fraction(result['report']) is None
    assert C.frame_components(result['report'], C.load_config()) == (None, None)
