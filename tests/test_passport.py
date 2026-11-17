"""E2: паспорт локации (tools/scene3d/posev2/passport.py), узел 3DALocationPassport и вход passport
у 3DAPlaceCandidates — без GPU и сервера. Запуск: python -m pytest tests/test_passport.py -q"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools/scene3d/posev2')]
import candidates as C  # noqa: E402
import passport as P  # noqa: E402

CFG = P.load_config()


def cells(x0, y0, x1, y1, step=.1):
    return [[x, y] for x in np.arange(x0, x1, step) for y in np.arange(y0, y1, step)]


def room():
    """Как в test_candidates: пол, сиденье дивана + верх спинки, тумба, кровать из двух кусков, стена."""
    return [
        {'id': 'floor_1', 'type': 'floor', 'height_m': 0.0, 'area_m2': 8.0, 'cells_xy': cells(-2, -2, 2, 2, .25)},
        {'id': 'surf_1', 'type': 'other', 'height_m': 0.47, 'area_m2': 0.9, 'cells_xy': cells(0, 1, 1.8, 1.5)},
        {'id': 'surf_2', 'type': 'other', 'height_m': 0.92, 'area_m2': 0.3, 'cells_xy': cells(0, 1.9, 1.8, 2.1)},
        {'id': 'surf_3', 'type': 'other', 'height_m': 0.72, 'area_m2': 0.2, 'cells_xy': cells(-1.5, 1, -1.1, 1.4)},
        {'id': 'surf_4', 'type': 'other', 'height_m': 0.52, 'area_m2': 1.2, 'cells_xy': cells(-3, -3, -2.2, -1.5)},
        {'id': 'surf_5', 'type': 'other', 'height_m': 0.55, 'area_m2': 1.2, 'cells_xy': cells(-2.2, -3, -1.4, -1.5)},
        {'id': 'wall_1', 'type': 'wall', 'height_m': 2.4},
    ]


PLATE_INFO = {'R_world_from_cv': [[1, 0, 0], [0, 0, 1], [0, -1, 0]], 'cam_height_m': 1.5, 'fov_x_deg': 70.0,
              'floor': {'visible': True}}


def built(**kw):
    return P.build(json.dumps(room()), None, None, PLATE_INFO, CFG, kw.get('pid', 'room1'), kw.get('sha1', 'abc'))


def by_type(p):
    return {o['type']: o for o in p['objects']}


# ------------------------------------------------------------------------------------------------ схема
def test_schema_and_content():
    p = built()
    assert P.validate(p) == []
    assert p['version'] == 1 and p['id'] == 'room1' and p['image_sha1'] == 'abc' and p['confirmed'] is False
    assert p['room_scale'] is None and p['room_scale_flag'] == 'room_scale_unknown'      # логика E1
    assert p['camera_height_m'] == 1.5 and p['floor_visible'] is True
    t = by_type(p)
    assert set(t) == {'floor', 'sofa', 'nightstand', 'bed'}
    assert p['objects'][0]['type'] == 'floor' and [o['id'] for o in p['objects']] == ['o1', 'o2', 'o3', 'o4']
    sofa = t['sofa']
    assert sofa['backrest'] is True and sofa['backrest_ids'] == ['surf_2'] and sofa['front_from'] == 'backrest'
    assert sofa['front_dir'][1] < -0.9                       # от спинки (y≈2) к сиденью (y≈1.25)
    assert [pl['surface_ids'] for pl in sofa['support_planes']] == [['surf_1']]          # спинка — не опора
    assert sofa['support_planes'][0]['kind'] == 'seat' and sofa['support_planes'][0]['height_m'] == 0.47
    assert {pl['kind'] for pl in t['bed']['support_planes']} == {'bed'} and len(t['bed']['support_planes']) == 2
    assert t['nightstand']['support_planes'][0]['kind'] == 'top'
    assert t['floor']['front_dir'] is None and t['floor']['support_planes'][0]['kind'] == 'floor'
    for o in p['objects']:
        assert len(o['bbox_m']) == 2 and o['bbox_m'][1][2] >= o['bbox_m'][0][2]
        for pl in o['support_planes']:
            assert len(pl['polygon']) >= 3
    json.dumps(p, allow_nan=False)                         # JSON без NaN


def test_validate_catches_errors():
    p = built()
    bad = json.loads(json.dumps(p))
    bad['version'] = 2
    bad['objects'][1]['support_planes'][0]['kind'] = 'chair'
    bad['confirmed'] = 'yes'
    err = P.validate(bad)
    assert any('version' in e for e in err) and any('kind' in e for e in err) and any('confirmed' in e for e in err)
    assert P.loads('{oops')[0] is None and P.loads('')[0] is None and P.loads('') == (None, [])
    assert P.loads(json.dumps({'3da_error': 'x'}))[0] is None


def test_hull_and_front_without_backrest():
    poly = P.hull([[0, 0], [1, 0], [1, 1], [0, 1], [.5, .5]])
    assert len(poly) == 4
    # длинная узкая скамья перед камерой (камера в 0,0): лицом к камере, по короткой оси
    bench = cells(-1, 2, 1, 2.4)
    center = [0.0, 2.2]
    d, src = P.front_of(bench, center, None, [0.0, 0.0])
    assert src == 'camera_long_side' and d[1] < -0.99


# ------------------------------------------------------------------------------------------------ кэш
def test_cache_second_call_does_not_rebuild(tmp_path):
    calls = []

    def build():
        calls.append(1)
        return built(sha1='s1')
    p1, src1, _ = P.get_or_build(tmp_path, 'room1', 's1', build)
    p2, src2, why = P.get_or_build(tmp_path, 'room1', 's1', build)
    assert (src1, src2, why) == ('built', 'cache', 'кэш') and len(calls) == 1
    assert p2['computed_at'] == p1['computed_at']
    assert (tmp_path / 'room1.json').is_file()
    # другая картинка под тем же id, файл не подтверждён — пересчёт и перезапись
    p3, src3, why3 = P.get_or_build(tmp_path, 'room1', 's2', lambda: built(sha1='s2'))
    assert src3 == 'built' and 'другой картинки' in why3 and json.loads((tmp_path / 'room1.json').read_text('utf-8'))['image_sha1'] == 's2'


def test_confirmed_edit_is_kept(tmp_path):
    p, _, _ = P.get_or_build(tmp_path, 'room1', 's1', lambda: built(sha1='s1'))
    path = tmp_path / 'room1.json'
    data = json.loads(path.read_text(encoding='utf-8'))
    data['confirmed'] = True
    data['objects'][2]['type'] = 'bench'                    # человек поправил тип
    path.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
    p2, src, _ = P.get_or_build(tmp_path, 'room1', 's1', lambda: pytest.fail('не должен пересчитывать'))
    assert src == 'cache' and p2['confirmed'] is True and p2['objects'][2]['type'] == 'bench'
    # подтверждённый файл другой картинки не перезаписывается
    p3, src3, _ = P.get_or_build(tmp_path, 'room1', 's9', lambda: built(sha1='s9'))
    assert src3 == 'built_not_saved' and json.loads(path.read_text('utf-8'))['confirmed'] is True
    assert any('не перезаписан' in n for n in p3['notes'])
    # сломанный JSON — пересчёт (не подтверждён: файл не читается)
    path.write_text('{broken', encoding='utf-8')
    _, src4, why4 = P.get_or_build(tmp_path, 'room1', 's1', lambda: built(sha1='s1'))
    assert src4 == 'built' and 'не годен' in why4


# ------------------------------------------------------------------------------------------------ кандидаты из паспорта
def test_candidates_from_passport_match_classify():
    objs0 = room()
    lab0 = C.classify(objs0, CFG)
    p = built()
    objs, lab, rs, flags, notes = P.to_candidates(p, json.dumps(room()), CFG)
    assert rs is None and flags == ['room_scale_unknown'] and notes == []
    for pc in ('sit_edge', 'lie', 'stand', 'recline'):
        a = [(s['id'], s['label']) for s in C.supports_for(pc, objs0, lab0, {}, CFG)]
        b = [(s['id'], s['label']) for s in C.supports_for(pc, objs, lab, {}, CFG)]
        assert a == b, pc
    plan0 = C.plan('lie', objs0, lab0, {}, 4, CFG, 2.3)
    plan1 = C.plan('lie', objs, lab, {}, 4, CFG, 2.3)
    key = ('variant', 'target_object', 'instance_hint', 'label', 'merge_ids', 'floor_id')
    assert [{k: c.get(k) for k in key} for c in plan0] == [{k: c.get(k) for k in key} for c in plan1]
    assert 'surf_2' not in {o['id'] for o in objs}         # спинка не опора


def test_passport_edits_drive_candidates():
    p = built()
    t = by_type(p)
    t['sofa']['type'] = 'bench'                             # человек: это скамья
    p['objects'] = [o for o in p['objects'] if o['type'] != 'bed']    # кровать удалена
    p['room_scale'] = 1.3
    objs, lab, rs, flags, _ = P.to_candidates(p, json.dumps(room()), CFG)
    assert [s['label'] for s in C.supports_for('sit_edge', objs, lab, {}, CFG)] == ['bench']
    assert rs == 1.3 and flags == ['room_scale_suspect']
    p['room_scale'] = 1.05
    assert P.to_candidates(p, json.dumps(room()), CFG)[3] == []
    with pytest.raises(ValueError):
        P.to_candidates(p, json.dumps([{'id': 'surf_77', 'type': 'other', 'height_m': .5}]), CFG)


def test_split_baseline():
    sup = {'variant': 'support'}
    base = {'variant': 'baseline'}
    assert P.split_baseline([sup, base], CFG) == ([sup], [base])
    assert P.split_baseline([base], CFG) == ([base], [])               # без мебели — пол сразу
    assert P.split_baseline([sup, base], dict(CFG, passport=dict(CFG['passport'], defer_baseline=False))) == ([sup, base], [])


def test_preview_draws():
    p = built()
    im = Image.new('RGB', (672, 384), (90, 90, 90))
    out = P.render_preview(im, p, PLATE_INFO)
    assert out.size == im.size and np.abs(np.asarray(out, int) - np.asarray(im, int)).sum() > 0
    assert P.render_preview(im, p, None).size == im.size      # без plate.json — только подписи и список


# ------------------------------------------------------------------------------------------------ узлы
def load_nodes():
    sys.path[:0] = [str(ROOT), str(ROOT / 'tools/scene3d')]
    spec = importlib.util.spec_from_file_location('threeda_nodes_passport_test', ROOT / 'nodes.py')
    N = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(N)
    N.config = lambda: {'repo': str(ROOT)}
    return N


def test_passport_node_signature():
    N = load_nodes()
    it = N.LocationPassport.INPUT_TYPES()
    assert list(it['required']) == ['location', 'objects', 'plate'] and list(it['optional']) == ['passport_id']
    assert N.LocationPassport.RETURN_TYPES == ('STRING', 'IMAGE', 'BOOLEAN')
    assert N.LocationPassport.RETURN_NAMES == ('passport', 'preview', 'needs_review')
    assert N.LocationPassport.CATEGORY == '3DA/auto'
    assert N.NODE_DISPLAY_NAME_MAPPINGS['3DALocationPassport'] == '3DA · паспорт локации'
    opt = N.PlaceCandidates.INPUT_TYPES()['optional']
    assert list(opt)[-1] == 'passport' and opt['passport'][0] == 'STRING' and opt['passport'][1]['forceInput'] is True


def test_passport_node_cache_and_confirm(monkeypatch, tmp_path):
    N = load_nodes()
    monkeypatch.setenv('THREEDA_PASSPORT_DIR', str(tmp_path))
    builds = []
    orig = P.build

    def counting(*a, **k):
        builds.append(1)
        return orig(*a, **k)
    monkeypatch.setattr(P, 'build', counting)
    monkeypatch.setattr(N, 'passport_helpers', lambda: P)
    plate = torch.rand((1, 48, 64, 3))
    key0 = N.LocationPassport.IS_CHANGED()
    text, preview, review = N.LocationPassport().execute('missing/profile.json', json.dumps(room()), plate)
    p = json.loads(text)
    assert P.validate(p) == [] and review is True and preview.shape == plate.shape and len(builds) == 1
    pixels = np.clip(np.rint(plate[0].numpy() * 255), 0, 255).astype('uint8')
    assert p['image_sha1'] == P.image_sha1(pixels) and p['id'] == p['image_sha1'][:16]
    path = tmp_path / (p['id'] + '.json')
    assert path.is_file()
    text2, _, _ = N.LocationPassport().execute('missing/profile.json', json.dumps(room()), plate)
    assert len(builds) == 1 and json.loads(text2)['computed_at'] == p['computed_at']       # второй вызов — из кэша
    data = json.loads(path.read_text(encoding='utf-8'))
    data['confirmed'] = True
    path.write_text(json.dumps(data), encoding='utf-8')
    assert N.LocationPassport.IS_CHANGED() != key0                                         # правка файла — новый ключ
    _, _, review3 = N.LocationPassport().execute('missing/profile.json', json.dumps(room()), plate)
    assert review3 is False and len(builds) == 1
    # свой passport_id
    text4, _, _ = N.LocationPassport().execute('missing/profile.json', json.dumps(room()), plate, 'living_room')
    assert json.loads(text4)['id'] == 'living_room' and (tmp_path / 'living_room.json').is_file()


def test_passport_node_never_raises(monkeypatch, tmp_path):
    N = load_nodes()
    monkeypatch.setenv('THREEDA_PASSPORT_DIR', str(tmp_path))
    plate = torch.rand((1, 8, 8, 3))
    err = json.dumps({'status': 'error', 'error': 'status=error\nmoge failed'})
    text, img, review = N.LocationPassport().execute(err, '[]', plate)
    assert '3da_error' in json.loads(text) and img is plate and review is True
    text, _, _ = N.LocationPassport().execute('x', '[]', plate, '../evil')
    assert '3da_error' in json.loads(text)


def fake_execute(statuses):
    calls = []

    def execute(self, location, pose_ref, pose_class, target_object, support_parts, secondary_contacts,
                instance_hint, height_m, body_json, proxy, strength, seed, block_on_fail=False,
                body_dx=0., body_dy=0., body_dz=0., body_dyaw=0., body_dpitch=0., body_droll=0., pose_text='',
                props='[]', body_transform='', megapixels=0., settings=None):
        calls.append(instance_hint or target_object)
        st = statuses.get(instance_hint or target_object, 'fallback')
        img = torch.full((1, 48, 64, 3), .1 * len(calls))
        mask = torch.zeros((1, 48, 64))
        if st == 'fallback':
            text = 'status=fallback — 3DA не сработал: x\nstatus=error\nx'
        else:
            text = st + ' — сцена\n' + json.dumps({'report': {'status': st, 'support_object': instance_hint or 'floor_1',
                                                              'support_gap_cm': 1.0, 'max_penetration_cm': 1.0,
                                                              'frame': {'ok': True, 'visible': True}}})
        return (img, img, mask, mask, 'prompt', text, img, img, mask, {'n': len(calls)})
    return execute, calls


def run_cand(N, monkeypatch, statuses, passport=None, pose_class='sit_edge'):
    execute, calls = fake_execute(statuses)
    monkeypatch.setattr(N.PlacePose, 'execute', execute)
    kw = {} if passport is None else {'passport': passport}
    out = N.PlaceCandidates().execute('missing/profile.json', torch.zeros((1, 8, 8, 3)), pose_class, '', '', 2.30, '',
                                      'grey', 1., 101, json.dumps(room()), n_candidates=4, **kw)
    return out, calls


def test_candidates_without_passport_unchanged(monkeypatch):
    """Без входа passport — как в E1 (test_candidates.test_node_picks_best_and_pick): те же вызовы и сводка."""
    N = load_nodes()
    out, calls = run_cand(N, monkeypatch, {'surf_1': 'wrong_surface', 'surf_4': 'ok'})
    assert calls == ['surf_1', 'surf_4', 'floor']
    summary = json.loads(out[10])
    assert 'passport' not in summary and [c['status'] for c in summary['candidates']] == ['wrong_surface', 'ok', 'fallback']
    assert 'паспорт' not in out[5].split('\n', 1)[0]
    out2, calls2 = run_cand(N, monkeypatch, {'surf_1': 'wrong_surface', 'surf_4': 'ok'}, passport='')
    def strip(s):
        return [{k: v for k, v in c.items() if k != 'seconds'} for c in s['candidates']]
    summary2 = json.loads(out2[10])                         # пустая строка passport — то же самое
    assert calls2 == calls and 'passport' not in summary2 and strip(summary2) == strip(summary)


def test_candidates_with_passport(monkeypatch):
    N = load_nodes()
    p = built()
    # мебель дала тело — запасной «пол» не ставится (экономия времени)
    out, calls = run_cand(N, monkeypatch, {'surf_1': 'wrong_surface', 'surf_4': 'ok'}, json.dumps(p))
    assert calls == ['surf_1', 'surf_4']
    summary = json.loads(out[10])
    assert summary['passport']['used'] is True and summary['passport']['baseline'] == 'skipped'
    assert json.loads(out[5].split('\n', 1)[1])['support_object'] == 'surf_4:bed'
    assert 'паспорт room1' in out[5].split('\n', 1)[0]
    # мебель не дала тела — «пол» всё-таки ставится
    out, calls = run_cand(N, monkeypatch, {'floor': 'wrong_surface'}, json.dumps(p))
    assert calls[-1] == 'floor' and json.loads(out[10])['passport']['baseline'] == 'ran'
    # правка человека: диван — это скамья, кровати нет
    t = by_type(p)
    t['sofa']['type'] = 'bench'
    p['objects'] = [o for o in p['objects'] if o['type'] != 'bed']
    _, calls = run_cand(N, monkeypatch, {'surf_1': 'ok'}, json.dumps(p))
    assert calls == ['surf_1']
    # негодный паспорт — как без него, с причиной
    out, calls = run_cand(N, monkeypatch, {'surf_1': 'wrong_surface', 'surf_4': 'ok'}, '{"version": 1}')
    assert calls == ['surf_1', 'surf_4', 'floor']
    s = json.loads(out[10])['passport']
    assert s['used'] is False and s['problems'] and 'не принят' in out[5].split('\n', 1)[0]


def test_candidates_cache_with_passport(monkeypatch, tmp_path):
    """E6-кэш кандидатов: паспорт входит в ключ; при попадании в кэш сведения о паспорте те же."""
    N = load_nodes()
    monkeypatch.setenv('THREEDA_CAND_CACHE', str(tmp_path / 'cache'))
    p = json.dumps(built())
    st = {'surf_1': 'wrong_surface', 'surf_4': 'ok'}
    out1, calls1 = run_cand(N, monkeypatch, st, p)
    out2, calls2 = run_cand(N, monkeypatch, st, p)
    s1, s2 = json.loads(out1[10]), json.loads(out2[10])
    assert calls1 == ['surf_1', 'surf_4'] and calls2 == [] and s2['cache'] == 'hit'
    assert s2['passport'] == s1['passport'] and 'паспорт room1' in out2[5].split('\n', 1)[0]
    _, calls3 = run_cand(N, monkeypatch, st)                  # без паспорта — свой ключ, считает заново
    assert calls3 == ['surf_1', 'surf_4', 'floor']
