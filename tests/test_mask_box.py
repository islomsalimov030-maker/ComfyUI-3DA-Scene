import importlib.util
import json
from pathlib import Path

import torch


spec = importlib.util.spec_from_file_location('mask_box', Path(__file__).parents[1] / 'auto' / 'mask_box.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def inputs():
    plate = torch.zeros((1, 100, 200, 3))
    candidates = json.dumps({'chosen': 3, 'candidates': [{'index': 3, 'height_m': 2.3}]})
    silhouette = torch.zeros((1, 100, 200))
    return plate, candidates, silhouette


def test_bbox_margin_and_inward_feather():
    plate, candidates, silhouette = inputs()
    silhouette[:, 30:70, 80:120] = 1
    mask, box, preview = module.MaskBox().execute(plate, candidates, silhouette, margin=.25, feather_px=5)
    box = json.loads(box)
    assert (box['x'], box['y'], box['w'], box['h']) == (70, 20, 60, 60)
    assert mask[0, 20, 70] < 1 and mask[0, 25, 75] == 1
    assert mask[0, :20].count_nonzero() == 0
    assert preview.shape == plate.shape


def test_bbox_clipped_at_edges():
    plate, candidates, silhouette = inputs()
    silhouette[:, :20, :30] = 1
    mask, text, _ = module.MaskBox().execute(plate, candidates, silhouette, margin=.5, feather_px=0)
    box = json.loads(text)
    assert (box['x'], box['y'], box['w'], box['h']) == (0, 0, 45, 30)
    assert mask[0, 29, 44] == 1 and mask[0, 30, 45] == 0


def test_fallback_empty_candidates_and_invalid_pick():
    plate, candidates, silhouette = inputs()
    mask, text, _ = module.MaskBox().execute(plate, '{"candidates":[]}', feather_px=0)
    box = json.loads(text)
    assert box['warning'] and box['x'] > 0 and box['y'] > 0 and mask.sum() > 0
    silhouette[:, 10:20, 10:20] = 1
    _, text, _ = module.MaskBox().execute(plate, candidates, silhouette, pick=2)
    assert json.loads(text)['warning']


def test_empty_candidates_uses_character_height_from_passport():
    plate, _, _ = inputs()
    kwargs = dict(margin=0, feather_px=0, min_frac=0)
    _, default, _ = module.MaskBox().execute(plate, '{"candidates": []}', **kwargs)
    _, taller, _ = module.MaskBox().execute(
        plate, '{"candidates": []}', passport='{"character": {"height_m": 3.0}}', **kwargs)
    _, embedded, _ = module.MaskBox().execute(
        plate, '{"candidates": [], "passport": {"height_m": 3.0}}', **kwargs)
    assert json.loads(taller)['h'] > json.loads(default)['h']
    assert json.loads(embedded)['h'] == json.loads(taller)['h']
    assert json.loads(taller)['warning']


def test_real_candidates_contract_honours_pick_and_rejects_silent_fallback():
    plate, _, silhouette = inputs()
    silhouette[:, 20:40, 30:60] = 1
    data = {'chosen': 2, 'pick': 2, 'candidates': [
        {'index': 0, 'height_m': 2.3}, {'index': 2, 'height_m': 2.3}]}
    _, text, _ = module.MaskBox().execute(plate, json.dumps(data), silhouette, pick=2, margin=0, feather_px=0)
    box = json.loads(text)
    assert (box['x'], box['y'], box['w'], box['h']) == (30, 20, 30, 20)
    assert not box['warning']
    data['pick'] = 9
    _, text, _ = module.MaskBox().execute(plate, json.dumps(data), silhouette)
    assert 'pick' in json.loads(text)['warning']


def test_place_candidates_outputs_are_paired_with_selected_silhouette(monkeypatch, tmp_path):
    from test_candidates import fake_execute, load_nodes, room

    nodes = load_nodes()
    execute, _ = fake_execute({'surf_1': 'ok', 'surf_4': 'ok'})

    def with_silhouette(self, *args, **kwargs):
        outputs = list(execute(self, *args, **kwargs))
        mask = torch.zeros((1, 48, 64))
        if args[6] == 'surf_1':
            mask[:, 5:15, 10:20] = 1
        else:
            mask[:, 20:30, 30:40] = 1
        outputs[3] = mask
        return tuple(outputs)

    monkeypatch.setattr(nodes.PlacePose, 'execute', with_silhouette)
    monkeypatch.setenv('THREEDA_CAND_CACHE', str(tmp_path / 'cache'))
    plate = torch.zeros((1, 48, 64, 3))
    boxes = []
    for pick in (0, 1):
        outputs = nodes.PlaceCandidates().execute(
            'missing/profile.json', torch.zeros((1, 8, 8, 3)), 'sit_edge', '', '', 2.3, '',
            'grey', 1., 101, json.dumps(room()), n_candidates=4, pick=pick)
        assert json.loads(outputs[10])['chosen'] == pick
        _, text, _ = module.MaskBox().execute(plate, outputs[10], outputs[3], pick=pick,
                                               margin=0, feather_px=0, min_frac=0)
        box = json.loads(text)
        assert not box['warning']
        boxes.append((box['x'], box['y'], box['w'], box['h']))
        # При штатном соединении pick задаётся в источнике; -1 берёт его выбор.
        _, auto_text, _ = module.MaskBox().execute(plate, outputs[10], outputs[3],
                                                    margin=0, feather_px=0, min_frac=0)
        assert json.loads(auto_text) == box
        # Другой pick на приёмнике не может получить недоступный силуэт.
        fallback_mask, mismatch, _ = module.MaskBox().execute(plate, outputs[10], outputs[3], pick=1-pick,
                                                              margin=0, feather_px=0, min_frac=0)
        mismatch_box = json.loads(mismatch)
        assert 'силуэт доступен только' in mismatch_box['warning']
        assert fallback_mask.sum() > 0
        assert (mismatch_box['x'], mismatch_box['y'], mismatch_box['w'], mismatch_box['h']) != boxes[-1]
    assert boxes == [(10, 5, 10, 10), (30, 20, 10, 10)]


def test_corrupt_inputs_fall_back_with_warning():
    plate, candidates, silhouette = inputs()
    for raw, mask in [('bad json', silhouette), ('[]', silhouette),
                      (candidates, torch.full((1, 100, 200), float('nan'))),
                      (candidates, torch.zeros((1, 100, 200)))]:
        out, text, _ = module.MaskBox().execute(plate, raw, mask)
        assert json.loads(text)['warning']
        assert out.sum() > 0


def test_invalid_numeric_options_do_not_crash():
    plate, candidates, silhouette = inputs()
    silhouette[:, 30:70, 80:120] = 1
    mask, text, _ = module.MaskBox().execute(
        plate, candidates, silhouette, margin=float('nan'), feather_px='bad',
        min_frac=None, max_frac=float('inf'))
    assert mask.isfinite().all()
    assert json.loads(text)['w'] > 0


def test_mismatched_aspect_ratio_is_not_misrepresented_as_bbox():
    plate, candidates, _ = inputs()
    silhouette = torch.ones((1, 30, 30))
    _, text, _ = module.MaskBox().execute(plate, candidates, silhouette)
    assert 'пропорции' in json.loads(text)['warning']


def test_missing_silhouette_is_not_misrepresented_as_candidate_bbox():
    plate, candidates, _ = inputs()
    _, text, _ = module.MaskBox().execute(plate, candidates)
    assert 'силуэт' in json.loads(text)['warning']


def test_preprojected_silhouette_handles_different_raster_aspect():
    plate, candidates, _ = inputs()
    render_mask = torch.zeros((1, 48, 64))
    render_mask[:, 5:15, 10:20] = 1
    # Внешняя цепочка масштабирует нормализованный растр 3DA до плиты, без crop.
    projected = torch.nn.functional.interpolate(render_mask[:, None], size=(100, 200),
                                                mode='nearest-exact')[:, 0]
    _, text, _ = module.MaskBox().execute(plate, candidates, projected,
                                          margin=0, feather_px=0, min_frac=0)
    box = json.loads(text)
    ys, xs = torch.where(projected[0] > 0)
    assert not box['warning']
    assert (box['x'], box['y'], box['w'], box['h']) == (
        int(xs.min()), int(ys.min()), int(xs.max()-xs.min()+1), int(ys.max()-ys.min()+1))
