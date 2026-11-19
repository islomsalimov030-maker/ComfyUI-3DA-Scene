"""E6: роль «зрение» (auto/vision.py, узел 3DAVisionPick), кэш кандидатов (auto/cand_cache.py) и variant_set
у 3DAPlaceCandidates — без GPU и сервера.
Запуск: python -m pytest tests/test_vision.py -q"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from auto import cand_cache as K, vision as V  # noqa: E402
from test_candidates import C, CFG, fake_execute, load_nodes, room  # noqa: E402

VCFG = V.load_config()
CANDS = json.dumps({'candidates': [{'index': 0, 'score': .5}, {'index': 1, 'score': .7},
                                   {'index': 2, 'score': None, 'status': 'skipped_time'}]})


# ------------------------------------------------------------------------------------------------ разбор ответа
def test_parse_plain_fenced_and_text_around():
    assert V.parse_answer('{"best": 1, "reason": "ok"}')['best'] == 1
    assert V.parse_answer('```json\n{"best": 0, "problems": {"1": "floating"}}\n```')['best'] == 0
    assert V.parse_answer('Sure! {bad} then {"best": -1} done')['best'] == -1
    assert V.parse_answer('no json here') is None and V.parse_answer(None) is None


def test_decide_pick_and_problems():
    info = V.decide('{"best": 1, "reason": "on sofa", "problems": {"0": "floating", "1": "ok", "7": "ok", '
                    '"0x": "ok", "2": "weird"}}', CANDS, 0, VCFG)
    assert info['pick'] == 1 and not info['all_bad'] and info['answer_ok']
    assert info['problems'] == {0: 'floating', 1: 'ok'}            # чужие номера и метки отброшены
    assert V.decide('{"best": "#0"}', CANDS, 0, VCFG)['pick'] == 0


def test_decide_out_of_range_and_broken_give_minus_one():
    for text in ('{"best": 2}', '{"best": 5}', '{"best": -3}', '{"best": true}', '{"best": 1.5}', '{"best": "x"}',
                 '{"reason": "no best"}', 'garbage', '', None, '[1, 2]'):
        info = V.decide(text, CANDS, 0, VCFG)
        assert info['pick'] == -1 and info['all_bad'] is False, text
    assert V.decide('{"best": 1}', 'not json', 0, VCFG)['pick'] == -1       # битый candidates_json — нет номеров


def test_coordinates_from_model_are_ignored():
    info = V.decide('{"best": 0, "dx": 0.5, "body_dyaw": 90, "xyz": [1, 2, 3]}', CANDS, 0, VCFG)
    assert info['pick'] == 0
    assert not any(k in info for k in ('dx', 'body_dyaw', 'xyz'))


def test_iterations_limited_to_three():
    assert VCFG['max_rounds'] == 3
    bad = '{"best": -1, "problems": {"0": "floating", "1": "wrong_support"}}'
    r0, r1, r2 = (V.decide(bad, CANDS, k, VCFG) for k in range(3))
    assert r0['all_bad'] and r0['next_variant_set'] == 1 and not r0['artist']
    assert r1['next_variant_set'] == 2 and not r1['artist']
    assert r2['next_variant_set'] is None and r2['artist'] and r2['note'].startswith('художнику: ')
    assert 'висит' in r2['note'] and 'не та опора' in r2['note']
    for k in range(3, 6):                                          # дальше третьего — никогда
        assert V.next_variant_set(k, True, VCFG) is None
    assert V.next_variant_set(0, False, VCFG) is None


def test_node_signature_and_never_raises():
    N = V.VisionPick
    assert N.CATEGORY == '3DA/auto' and N.RETURN_TYPES == ('INT', 'BOOLEAN', 'STRING')
    assert N.RETURN_NAMES == ('pick', 'all_bad', 'note')
    out = N().execute('{"best": 1}', CANDS)['result']
    assert out[0] == 1 and out[1] is False and json.loads(out[2].split('\n', 1)[1])['pick'] == 1
    for args in ((None, None), (object(), 5), ('{"best": 1}', CANDS, 'bad round')):
        res = N().execute(*args)['result']
        assert res[0] == -1 and res[1] is False
    from auto import nodes as AN
    assert AN.NODE_CLASS_MAPPINGS['3DAVisionPick'] is V.VisionPick
    assert AN.NODE_DISPLAY_NAME_MAPPINGS['3DAVisionPick'] == '3DA · выбор кандидата (зрение)'


# ------------------------------------------------------------------------------------------------ variant_set
def objs_labels():
    objs = C.enrich(room(), None, None, CFG)
    return objs, C.classify(objs, CFG)


def test_variant_set_zero_is_old_plan():
    objs, labels = objs_labels()
    for pc in ('stand', 'sit_edge', 'lie', 'auto'):
        assert C.plan(pc, objs, labels, {}, 4, CFG, 2.3, '') == C.plan(pc, objs, labels, {}, 4, CFG, 2.3, '', 0)


def test_variant_set_one_other_yaws():
    objs, labels = objs_labels()
    base = C.plan('sit_edge', objs, labels, {}, 4, CFG, 2.3, '')
    v1 = C.plan('sit_edge', objs, labels, {}, 4, CFG, 2.3, '', 1)
    assert len(v1) == 4 and all(c['body_dyaw'] in CFG['variant_sets']['yaws_deg'] for c in v1)
    assert v1[0]['body_dyaw'] == 180 and v1[0]['support_id'] == base[0]['support_id']
    st = C.plan('stand', objs, labels, {}, 4, CFG, 2.3, '', 1)
    assert [c['body_dyaw'] for c in st] == [float(y) for y in CFG['variant_sets']['yaws_deg'][:4]]


def test_variant_set_two_neighbors_then_shifts():
    objs, labels = objs_labels()
    base = C.plan('sit_edge', objs, labels, {}, 2, CFG, 2.3, '')
    used = {c['support_id'] for c in base}
    v2 = C.plan('sit_edge', objs, labels, {}, 2, CFG, 2.3, '', 2)
    neigh = [c for c in v2 if c['variant'] == 'neighbor']
    assert neigh and not used & {c['support_id'] for c in neigh}
    st = C.plan('stand', objs, labels, {}, 4, CFG, 2.3, '', 2)
    assert [(c['body_dx'], c['body_dy']) for c in st] == [tuple(map(float, s)) for s in CFG['variant_sets']['shifts_m']]


# ------------------------------------------------------------------------------------------------ кэш
def test_pack_roundtrip_exact():
    img = torch.from_numpy(np.random.default_rng(0).integers(0, 256, (1, 6, 5, 3)).astype(np.float32) / 255)
    raw = torch.rand(1, 4, 4)                                       # не k/255 — хранится как есть
    obj = (img, raw, 'текст', {'plate': img, 'n': 3, 'x': None}, [1.5, True])
    arrays = {}
    back = K.unpack(json.loads(json.dumps(K.pack(obj, arrays))), arrays)
    assert torch.equal(back[0], img) and back[0].dtype == img.dtype and arrays['a0'].dtype == np.uint8
    assert torch.equal(back[1], raw) and back[2] == 'текст' and torch.equal(back[3]['plate'], img)
    assert back[3]['n'] == 3 and back[4] == [1.5, True] and isinstance(back, tuple)


def run_cached(N, monkeypatch, tmp_path, pick, variant_set=0, statuses=None):
    execute, calls = fake_execute(statuses or {'surf_1': 'wrong_surface', 'surf_4': 'ok'})
    monkeypatch.setattr(N.PlacePose, 'execute', execute)
    monkeypatch.setenv('THREEDA_CAND_CACHE', str(tmp_path / 'cache'))
    out = N.PlaceCandidates().execute('missing/profile.json', torch.zeros((1, 8, 8, 3)), 'sit_edge', '', '', 2.30, '',
                                      'grey', 1., 101, json.dumps(room()), n_candidates=4, pick=pick,
                                      variant_set=variant_set)
    return out, calls


def test_cache_second_call_with_other_pick_does_not_recompute(monkeypatch, tmp_path):
    N = load_nodes()
    out1, calls1 = run_cached(N, monkeypatch, tmp_path, -1)
    s1 = json.loads(out1[10])
    assert len(calls1) == 3 and s1['cache'] == 'miss' and s1['chosen'] == 1
    out2, calls2 = run_cached(N, monkeypatch, tmp_path, 0)
    s2 = json.loads(out2[10])
    assert calls2 == [] and s2['cache'] == 'hit' and s2['chosen'] == 0 and s2['pick'] == 0
    assert json.loads(out2[5].split('\n', 1)[1])['candidate_index'] == 0
    assert s2['candidates'] == s1['candidates']
    for k in (0, 1, 2, 3, 6, 7, 8):                                  # выходы кандидата 0 — как при прямом счёте
        assert out2[k].shape == out1[k].shape
    assert torch.equal(out2[11], out1[11]) and out2[9] == {'n': 1}
    direct, _ = run_node_nocache(N, monkeypatch, 0)
    assert torch.equal(out2[0], direct[0]) and out2[5] == direct[5]
    out3, calls3 = run_cached(N, monkeypatch, tmp_path, -1)          # снова лучший — из кэша, тот же результат
    assert calls3 == [] and out3[5] == out1[5]
    out4, calls4 = run_cached(N, monkeypatch, tmp_path, -1, variant_set=1)   # другой набор — другой ключ
    assert len(calls4) > 0 and json.loads(out4[10])['variant_set'] == 1
    assert all(c['body_dyaw'] != 0 for c in calls4)


def run_node_nocache(N, monkeypatch, pick):
    monkeypatch.delenv('THREEDA_CAND_CACHE', raising=False)
    execute, calls = fake_execute({'surf_1': 'wrong_surface', 'surf_4': 'ok'})
    monkeypatch.setattr(N.PlacePose, 'execute', execute)
    out = N.PlaceCandidates().execute('missing/profile.json', torch.zeros((1, 8, 8, 3)), 'sit_edge', '', '', 2.30, '',
                                      'grey', 1., 101, json.dumps(room()), n_candidates=4, pick=pick)
    assert json.loads(out[10])['cache'] == 'off'
    return out, calls


def test_cache_key_depends_on_inputs_not_pick(monkeypatch, tmp_path):
    N = load_nodes()
    run_cached(N, monkeypatch, tmp_path, -1)
    _, calls = run_cached(N, monkeypatch, tmp_path, 1, statuses={'surf_1': 'ok', 'surf_4': 'ok'})
    assert calls == []                                               # тот же вход — кэш (статусы фейка не входят в ключ)
    monkeypatch.setenv('THREEDA_CAND_CACHE', str(tmp_path / 'cache'))
    execute, calls = fake_execute({'surf_1': 'ok'})
    monkeypatch.setattr(N.PlacePose, 'execute', execute)
    N.PlaceCandidates().execute('missing/profile.json', torch.ones((1, 8, 8, 3)), 'sit_edge', '', '', 2.30, '',
                                'grey', 1., 101, json.dumps(room()), n_candidates=4, pick=-1)
    assert len(calls) == 3                                           # другой pose_ref — пересчёт


def test_cache_broken_entry_recomputes_and_prune(monkeypatch, tmp_path):
    N = load_nodes()
    run_cached(N, monkeypatch, tmp_path, -1)
    entry = next(p for p in (tmp_path / 'cache').iterdir() if not p.name.startswith('tmp_'))
    (entry / 'cand_1.npz').write_bytes(b'broken')
    out, calls = run_cached(N, monkeypatch, tmp_path, -1)
    assert len(calls) == 3 and json.loads(out[10])['cache'] == 'miss'
    c = K.CandCache(tmp_path / 'p', 'k', keep=2, tmp_age_s=86400)
    for i in range(4):
        (tmp_path / 'p' / f'e{i}').mkdir(parents=True)
    c.root.mkdir(exist_ok=True)
    c.prune()
    assert len([p for p in c.root.iterdir()]) == 2


def test_cache_survives_location_dir_rewrite(monkeypatch, tmp_path):
    """3DALocation переписывает папку профиля при каждом своём прогоне — ключ кэша от этого не меняется."""
    N = load_nodes()
    loc = tmp_path / 'locations' / 'auto_abc' / 'profile.json'
    loc.parent.mkdir(parents=True)
    loc.write_text(json.dumps({'loc_id': 'auto_abc', 'objects': [], 't': 1}), encoding='utf-8')
    monkeypatch.setenv('THREEDA_CAND_CACHE', str(tmp_path / 'cache'))

    def call(pick):
        execute, calls = fake_execute({'surf_1': 'wrong_surface', 'surf_4': 'ok'})
        monkeypatch.setattr(N.PlacePose, 'execute', execute)
        out = N.PlaceCandidates().execute(str(loc), torch.zeros((1, 8, 8, 3)), 'sit_edge', '', '', 2.30, '', 'grey',
                                          1., 101, json.dumps(room()), n_candidates=4, pick=pick)
        return json.loads(out[10]), calls
    s1, c1 = call(-1)
    assert s1['cache'] == 'miss' and len(c1) == 3
    loc.write_text(json.dumps({'loc_id': 'auto_abc', 'objects': [], 't': 2}), encoding='utf-8')
    (loc.parent / 'objects_preview.png').write_bytes(b'new')
    s2, c2 = call(0)
    assert s2['cache'] == 'hit' and c2 == [] and s2['chosen'] == 0


def test_cache_tracks_profile_geometry_and_referenced_resources(monkeypatch, tmp_path):
    N = load_nodes()
    loc = tmp_path / 'locations' / 'profile.json'
    loc.parent.mkdir(parents=True)
    grid = loc.parent / 'surface.npz'
    profile = {'loc_id': 'room', 'objects': [{'id': 'surf_1', 'h0': 0.5, 'top_grid_npz': grid.name}],
               'anchors': [], 't': 1}
    loc.write_text(json.dumps(profile), encoding='utf-8')
    np.savez_compressed(grid, origin=[0., 0.], mask=[[1, 1], [0, 0]])
    monkeypatch.setenv('THREEDA_CAND_CACHE', str(tmp_path / 'cache'))

    def call(pick):
        execute, calls = fake_execute({'surf_1': 'wrong_surface', 'surf_4': 'ok'})
        monkeypatch.setattr(N.PlacePose, 'execute', execute)
        out = N.PlaceCandidates().execute(str(loc), torch.zeros((1, 8, 8, 3)), 'sit_edge', '', '', 2.30, '', 'grey',
                                          1., 101, json.dumps(room()), n_candidates=4, pick=pick)
        return json.loads(out[10]), calls

    summary, calls = call(-1)
    assert summary['cache'] == 'miss' and calls
    profile['t'] = 2
    loc.write_text(json.dumps(profile, indent=2), encoding='utf-8')
    np.savez_compressed(grid, mask=[[1, 1], [0, 0]], origin=[0., 0.])
    summary, calls = call(0)
    assert summary['cache'] == 'hit' and not calls
    profile['objects'][0]['h0'] = 0.6
    loc.write_text(json.dumps(profile), encoding='utf-8')
    summary, calls = call(-1)
    assert summary['cache'] == 'miss' and calls
    np.savez_compressed(grid, origin=[0., 0.], mask=[[1, 0], [0, 0]])
    summary, calls = call(-1)
    assert summary['cache'] == 'miss' and calls
    (loc.parent / 'plate_mesh.obj').write_text('mesh 1', encoding='utf-8')
    summary, calls = call(-1)
    assert summary['cache'] == 'miss' and calls
    (loc.parent / 'plate_mesh.obj').write_text('mesh 2', encoding='utf-8')
    summary, calls = call(-1)
    assert summary['cache'] == 'miss' and calls


def test_cache_skips_transient_oom_fallback(monkeypatch, tmp_path):
    """E12: fallback от CUDA OOM не попадает в кэш (иначе S19 получал пустой силуэт при свободной памяти)."""
    N = load_nodes()
    execute, calls = fake_execute({})

    def oom_execute(self, *args, **kwargs):
        outs = list(execute(self, *args, **kwargs))
        outs[5] = 'status=fallback — 3DA не сработал: Error: Failed to retain CUDA context (Out of memory)\nx'
        return tuple(outs)
    monkeypatch.setattr(N.PlacePose, 'execute', oom_execute)
    monkeypatch.setenv('THREEDA_CAND_CACHE', str(tmp_path / 'cache'))
    args = ('missing/profile.json', torch.zeros((1, 8, 8, 3)), 'sit_edge', '', '', 2.30, '', 'grey', 1., 101,
            json.dumps(room()))
    out1 = N.PlaceCandidates().execute(*args, n_candidates=2)
    assert json.loads(out1[10])['cache'] == 'off'
    assert not list((tmp_path / 'cache').glob('*/summary.json'))
    n1 = len(calls)
    out2 = N.PlaceCandidates().execute(*args, n_candidates=2)       # память освободилась — считаем заново
    assert len(calls) > n1 and json.loads(out2[10])['cache'] == 'off'


def test_transient_failure_markers():
    assert K.CandCache.transient_failure([{'status': 'fallback', 'error': 'Failed to retain CUDA context'}])
    assert K.CandCache.transient_failure([{'status': 'fallback', 'error': 'CUDA Out of memory'}])
    assert not K.CandCache.transient_failure([{'status': 'fallback', 'error': 'no_target: нет опоры'}])
    assert not K.CandCache.transient_failure([{'status': 'ok', 'error': None}, {'status': 'skipped_time'}])
