import importlib.util
import json
from pathlib import Path

import torch

from test_candidates import fake_execute, load_nodes, room


def load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / 'auto' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_place_pack_box_required_inputs_for_each_pick(monkeypatch, tmp_path):
    nodes = load_nodes()
    execute, _ = fake_execute({'surf_1': 'ok', 'surf_4': 'ok'})

    def with_silhouette(self, *args, **kwargs):
        result = list(execute(self, *args, **kwargs))
        mask = torch.zeros((1, 48, 64))
        if args[6] == 'surf_1':
            mask[:, 5:15, 10:20] = 1
        else:
            mask[:, 20:30, 30:40] = 1
        result[3] = mask
        return tuple(result)

    monkeypatch.setattr(nodes.PlacePose, 'execute', with_silhouette)
    monkeypatch.setenv('THREEDA_CAND_CACHE', str(tmp_path / 'cache'))
    pack, box = load('mask_box_pack').MaskBoxPack(), load('mask_box').MaskBox()
    assert set(box.INPUT_TYPES()['required']) == {'plate', 'candidates_json'}
    for pick, expected in [(0, (10, 5, 10, 10)), (1, (30, 20, 10, 10))]:
        out = nodes.PlaceCandidates().execute('missing/profile.json', torch.zeros((1, 8, 8, 3)),
            'sit_edge', '', '', 2.3, '', 'grey', 1., 101, json.dumps(room()), n_candidates=4, pick=pick)
        packed, = pack.execute(out[10], out[3])
        _, result, _ = box.execute(torch.zeros((1, 48, 64, 3)), packed, margin=0, feather_px=0, min_frac=0)
        result = json.loads(result)
        assert not result['warning']
        assert tuple(result[k] for k in ('x', 'y', 'w', 'h')) == expected
    _, result, _ = box.execute(torch.zeros((1, 48, 64, 3)), packed, pick=0)
    assert json.loads(result)['warning']


def test_invalid_packed_silhouette_falls_back():
    box = load('mask_box').MaskBox()
    payload = {'chosen': 0, 'candidates': [{'index': 0}],
               'silhouettes': {'0': {'shape': [2, 2], 'zlib_base64': 'bad'}}}
    mask, result, _ = box.execute(torch.zeros((1, 48, 64, 3)), json.dumps(payload))
    assert json.loads(result)['warning'] and mask.sum() > 0
