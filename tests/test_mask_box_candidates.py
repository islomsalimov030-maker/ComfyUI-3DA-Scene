import importlib.util
import json
import sys
import types
from pathlib import Path

import torch

from test_candidates import fake_execute, load_nodes, room


ROOT = Path(__file__).resolve().parents[1]


def load_adapter(monkeypatch, nodes):
    package = types.ModuleType('e12_test_package')
    package.__path__ = [str(ROOT)]
    auto = types.ModuleType('e12_test_package.auto')
    auto.__path__ = [str(ROOT / 'auto')]
    monkeypatch.setitem(sys.modules, 'e12_test_package', package)
    monkeypatch.setitem(sys.modules, 'e12_test_package.auto', auto)
    monkeypatch.setitem(sys.modules, 'e12_test_package.nodes', nodes)
    spec = importlib.util.spec_from_file_location('e12_test_package.auto.mask_box_candidates', ROOT / 'auto/mask_box_candidates.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MaskBoxCandidates


def arguments():
    return dict(plate=torch.zeros((1, 48, 64, 3)), location='missing/profile.json',
                pose_ref=torch.zeros((1, 8, 8, 3)), pose_class='sit_edge', support_parts='',
                secondary_contacts='', height_m=2.3, body_json='', proxy='grey', strength=1., seed=101,
                objects=json.dumps(room()), n_candidates=4, margin=0, feather_px=0, min_frac=0)


def test_select_another_candidate_from_one_cached_set(monkeypatch, tmp_path):
    nodes = load_nodes()
    execute, calls = fake_execute({'surf_1': 'ok', 'surf_4': 'ok'})

    def with_silhouette(self, *args, **kwargs):
        out = list(execute(self, *args, **kwargs))
        mask = torch.zeros((1, 48, 64))
        if args[6] == 'surf_1':
            mask[:, 5:15, 10:20] = 1
        else:
            mask[:, 20:30, 30:40] = 1
        out[3] = mask
        return tuple(out)

    monkeypatch.setattr(nodes.PlacePose, 'execute', with_silhouette)
    monkeypatch.setenv('THREEDA_CAND_CACHE', str(tmp_path / 'cache'))
    place_execute = nodes.PlaceCandidates.execute
    selections = []

    def observe(self, *args, **kwargs):
        outputs = place_execute(self, *args, **kwargs)
        selections.append(json.loads(outputs[10])['chosen'])
        return outputs

    monkeypatch.setattr(nodes.PlaceCandidates, 'execute', observe)
    adapter = load_adapter(monkeypatch, nodes)()
    _, text, _, selected = adapter.execute(pick=1, **arguments())
    box = json.loads(text)
    assert (box['x'], box['y'], box['w'], box['h']) == (30, 20, 10, 10)
    assert not box['warning']
    assert json.loads(selected)['cache'] == 'hit'
    assert json.loads(selected)['chosen'] == 1
    assert selections == [0, 1]
    assert len(calls) == 3


def test_missing_candidate_falls_back_using_passport(monkeypatch, tmp_path):
    nodes = load_nodes()
    execute, _ = fake_execute({'surf_1': 'ok', 'surf_4': 'ok'})
    monkeypatch.setattr(nodes.PlacePose, 'execute', execute)
    monkeypatch.setenv('THREEDA_CAND_CACHE', str(tmp_path / 'cache'))
    adapter = load_adapter(monkeypatch, nodes)()
    _, text, _, _ = adapter.execute(pick=99, passport='{"height_m":3.0}', **arguments())
    box = json.loads(text)
    assert box['warning'] and box['h'] > 30


def test_unverified_second_run_is_not_mistaken_for_same_candidate(monkeypatch):
    nodes = load_nodes()
    adapter = load_adapter(monkeypatch, nodes)()
    candidates = [{'index': 0, 'status': 'ok'}, {'index': 1, 'status': 'ok'}]

    def execute(self, pick, **kwargs):
        chosen = 0 if pick == -1 else pick
        mask = torch.ones((1, 48, 64))
        summary = json.dumps({'chosen': chosen, 'pick': pick, 'candidates': candidates, 'cache': 'off'})
        return (None, None, None, mask, None, None, None, None, None, None, summary, None)

    monkeypatch.setattr(nodes.PlaceCandidates, 'execute', execute)
    _, text, _, _ = adapter.execute(pick=1, **arguments())
    assert 'кэше' in json.loads(text)['warning']
    _, text, _, _ = adapter.execute(pick=-1, **arguments())
    assert not json.loads(text)['warning']
