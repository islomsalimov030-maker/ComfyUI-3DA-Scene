"""E1: кандидаты постановки (tools/scene3d/posev2/candidates.py) и узел 3DAPlaceCandidates без GPU и сервера.
Запуск: python -m pytest tests/test_candidates.py -q"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools/scene3d/posev2')]
import candidates as C  # noqa: E402

CFG = C.load_config()


def cells(x0, y0, x1, y1, step=.1):
    return [[x, y] for x in np.arange(x0, x1, step) for y in np.arange(y0, y1, step)]


def room():
    """Синтетическая гостиная: пол, сиденье дивана + верх спинки, тумба, кровать из двух корзин высоты, стена."""
    return [
        {'id': 'floor_1', 'type': 'floor', 'height_m': 0.0, 'area_m2': 8.0, 'cells_xy': cells(-2, -2, 2, 2, .25)},
        {'id': 'surf_1', 'type': 'other', 'height_m': 0.47, 'area_m2': 0.9, 'cells_xy': cells(0, 1, 1.8, 1.5)},
        {'id': 'surf_2', 'type': 'other', 'height_m': 0.92, 'area_m2': 0.3, 'cells_xy': cells(0, 1.9, 1.8, 2.1)},
        {'id': 'surf_3', 'type': 'other', 'height_m': 0.60, 'area_m2': 0.2, 'cells_xy': cells(-1.5, 1, -1.1, 1.4)},
        {'id': 'surf_4', 'type': 'other', 'height_m': 0.52, 'area_m2': 1.2, 'cells_xy': cells(-3, -3, -2.2, -1.5)},
        {'id': 'surf_5', 'type': 'other', 'height_m': 0.55, 'area_m2': 1.2, 'cells_xy': cells(-2.2, -3, -1.4, -1.5)},
        {'id': 'wall_1', 'type': 'wall', 'height_m': 2.4},
    ]


def labels_of(objs):
    return {k: v['label'] for k, v in C.classify(objs, CFG).items()}


def test_geometry_labels():
    lab = labels_of(room())
    assert lab['floor_1'] == 'floor'
    assert lab['surf_1'] == 'sofa'          # сиденье + более высокий кусок рядом (спинка)
    assert lab['surf_2'] == 'backrest'      # верх спинки — не опора
    assert lab['surf_3'] == 'nightstand'    # маленький верх 0.6 м
    assert lab['surf_4'] == lab['surf_5'] == 'bed'   # два куска одной кровати → одна группа 1.6 × 1.5 м
    assert 'wall_1' not in lab


def test_supports_by_class_sit_not_nightstand():
    objs = room()
    lab = C.classify(objs, CFG)
    sit = [s['label'] for s in C.supports_for('sit_edge', objs, lab, {}, CFG)]
    assert sit[0] == 'sofa' and 'nightstand' not in sit and 'floor' not in sit
    assert sit.count('bed') == 1             # куски одной кровати — один вариант
    lie = [s['label'] for s in C.supports_for('lie', objs, lab, {}, CFG)]
    assert lie[0] == 'bed' and lie[-1] == 'floor'
    assert [s['label'] for s in C.supports_for('stand', objs, lab, {}, CFG)] == ['floor']
    assert [s['label'] for s in C.supports_for('kneel', objs, lab, {}, CFG)] == ['floor']


def test_synonyms_and_rank():
    assert C.support_rank('sit_edge', 'nightstand', CFG) is None
    assert C.support_rank('sit_edge', 'table', CFG) is None             # стол сидя — нельзя
    assert C.support_rank('lie', 'table', CFG)[0] == 1                  # лёжа — как массажный стол (по синониму)
    assert C.support_rank('sit_edge', 'sofa', CFG) == (0, 0)
    assert C.matches_word('sofa', 'couch', CFG) and C.matches_word('floor', 'mat', CFG)
    assert not C.matches_word('bed', 'sofa', CFG)
    # смысловое имя старого профиля сильнее геометрической метки
    assert C.semantic_label({'id': 'sofa_1', 'type': 'sofa'}, {'sofa_1': {'label': 'bench'}}) == 'sofa'


def test_hint_goes_first():
    objs = room()
    lab = C.classify(objs, CFG)
    order = [s['label'] for s in C.supports_for('sit_edge', objs, lab, C.parse_hint('{"support": "bed"}'), CFG)]
    assert order[0] == 'bed'
    assert C.parse_hint('не json') == {} and C.parse_hint('') == {}


def test_plan_counts_and_variants():
    objs = room()
    lab = C.classify(objs, CFG)
    sit = C.plan('sit_edge', objs, lab, {}, 4, CFG, 2.30)
    assert [c['support_id'] for c in sit] == ['surf_1', 'surf_4', None]  # диван, кровать (тумба — нет)
    assert sit[-1]['variant'] == 'baseline' and sit[-1]['target_object'] == 'floor'   # запасной «как сейчас»
    assert len(C.plan('sit_edge', objs, lab, {}, 2, CFG, 2.30)) == 2
    assert sit[0]['target_object'] == 'other' and sit[0]['instance_hint'] == 'surf_1'
    stand = C.plan('stand', objs, lab, {'near_wall': True}, 4, CFG, 2.30)
    assert [c['variant'] for c in stand] == ['floor', 'wall', 'yaw+25', 'yaw-25']
    assert stand[0]['target_object'] == 'floor' and 'back:wall' in stand[1]['secondary_contacts']
    assert len(C.plan('stand', objs, lab, {}, 1, CFG, 2.30)) == 1
    assert C.plan('auto', objs, lab, {}, 4, CFG, 2.30)[0]['variant'] == 'auto'


def test_room_scale():
    scale, flags = C.room_scale(room(), {}, CFG)
    assert scale is None and flags == ['room_scale_unknown']          # авто-поверхности без смысловых имён
    small = [{'id': 'floor_1', 'type': 'floor', 'height_m': 0.0}, {'id': 'sofa_1', 'type': 'sofa', 'height_m': 0.30},
             {'id': 'table_1', 'type': 'table', 'height_m': 0.50}, {'id': 'bed_1', 'type': 'bed', 'height_m': 0.35}]
    scale, flags = C.room_scale(small, {}, CFG)
    assert abs(scale - 1.5) < .01 and flags == ['room_scale_suspect']
    ok = [{'id': 'f', 'type': 'floor', 'height_m': 0.0}, {'id': 's', 'type': 'sofa', 'height_m': 0.46}]
    assert C.room_scale(ok, {}, CFG) == (0.978, [])
    cands = [dict(C.plan('sit_edge', small, {}, {}, 2, CFG, 2.30)[0])]
    extra = C.derived(cands, [{'report': {}}], CFG, 1, ['room_scale_suspect'], 1.5)
    assert extra[-1]['variant'] == 'room_scale' and abs(extra[-1]['height_m'] - 2.30 / 1.5) < .01


def rep(status, gap=1.0, pen=1.0, frame_ok=True):
    return {'status': status, 'support_gap_cm': gap, 'max_penetration_cm': pen, 'frame': {'ok': frame_ok, 'visible': True},
            'orient': {'total_change_deg': 5.0}, 'angle_to_camera_deg': {'ref': 10.0, 'plate': 20.0}}


def test_score_ranking_and_pick():
    sofa = {'label': 'sofa'}
    bed = {'label': 'bed'}
    s_ok, comp = C.score(sofa, 'ok', rep('ok'), 'sit_edge', CFG)
    s_partial, _ = C.score(sofa, 'partial', rep('partial', gap=4.0), 'sit_edge', CFG)
    s_wrong, _ = C.score(sofa, 'wrong_surface', rep('wrong_surface', gap=4.0, pen=40.0), 'sit_edge', CFG)
    s_bed, _ = C.score(bed, 'ok', rep('ok'), 'sit_edge', CFG)
    s_fb, _ = C.score(sofa, 'fallback', {}, 'sit_edge', CFG)
    assert 0 <= s_fb < s_wrong < s_partial < s_ok <= 1
    assert s_bed < s_ok                                   # та же постановка, но опора ниже в списке класса
    assert set(comp) == {'status', 'gap', 'penetration', 'pose_match', 'support_class', 'visibility'}
    entries = [{'index': 0, 'score': .4}, {'index': 1, 'score': .9}, {'index': 2, 'score': .9}, {'index': 3, 'score': None}]
    assert C.choose(entries, -1)['index'] == 1
    assert C.choose(entries, 2)['index'] == 2
    assert C.choose(entries, 7)['index'] == 1            # нет такого номера — лучший
    assert C.choose([{'index': 0, 'score': None}], -1) is None


def test_report_parsing_and_actual_support():
    text = 'ok — сцена\n' + json.dumps({'report': {'status': 'partial', 'support_object': 'surf_1'}})
    st, r = C.report_of(text)
    assert st == 'partial' and r['support_object'] == 'surf_1'
    assert C.report_of('status=fallback — 3DA не сработал\nstatus=error\nx')[0] == 'fallback'
    objs = room()
    lab = C.classify(objs, CFG)
    assert C.actual_support(r, {}, lab, objs) == 'surf_1:sofa'
    assert C.actual_support({'support_object': 'other'}, {}, lab, objs) == 'other'


def test_mask_cells(tmp_path):
    mask = np.zeros((10, 10), bool)
    mask[2:4, 5:7] = True
    np.savez(tmp_path / 'surf_1_top.npz', origin=np.array([1.0, 2.0]), mask=mask)
    xy = C.mask_cells(tmp_path / 'surf_1_top.npz', CFG)
    assert len(xy) == 4 and abs(xy[0][0] - (1.0 + 2.5 * CFG['cell_m'])) < 1e-9
    assert C.mask_cells(tmp_path / 'missing.npz', CFG) is None


# ------------------------------------------------------------------------------------------------ узел
def load_nodes():
    nodes = ROOT / 'nodes.py'
    sys.path[:0] = [str(ROOT), str(ROOT / 'tools/scene3d')]
    spec = importlib.util.spec_from_file_location('threeda_nodes_cand_test', nodes)
    N = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(N)
    N.config = lambda: {'repo': str(ROOT)}
    return N


def fake_execute(statuses):
    calls = []

    def execute(self, location, pose_ref, pose_class, target_object, support_parts, secondary_contacts,
                instance_hint, height_m, body_json, proxy, strength, seed, block_on_fail=False,
                body_dx=0., body_dy=0., body_dz=0., body_dyaw=0., body_dpitch=0., body_droll=0., pose_text='',
                props='[]', body_transform='', megapixels=0., settings=None):
        calls.append(dict(target_object=target_object, instance_hint=instance_hint, body_dyaw=body_dyaw))
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


def test_node_signature_and_compat():
    N = load_nodes()
    old = N.PlacePose.INPUT_TYPES()
    new = N.PlaceCandidates.INPUT_TYPES()
    for k, v in old['required'].items():
        if k not in N.PlaceCandidates.SKIP:
            assert new['required'][k] == v                     # те же имена и умолчания
    assert 'target_object' not in new['required'] and 'body_dx' not in new['required']
    assert new['required']['n_candidates'][1]['default'] == 4 and new['required']['pick'][1]['default'] == -1
    assert N.PlaceCandidates.RETURN_NAMES[:10] == N.PlacePose.RETURN_NAMES
    assert N.PlaceCandidates.RETURN_NAMES[10:] == ('candidates_json', 'contact_sheet')
    assert N.PlaceCandidates.CATEGORY == '3DA/auto'
    assert N.NODE_DISPLAY_NAME_MAPPINGS['3DAPlaceCandidates'] == '3DA · постановка: кандидаты'


def run_node(N, monkeypatch, statuses, pick=-1, pose_class='sit_edge'):
    execute, calls = fake_execute(statuses)
    monkeypatch.setattr(N.PlacePose, 'execute', execute)
    objs = [{k: v for k, v in o.items()} for o in room()]
    out = N.PlaceCandidates().execute('missing/profile.json', torch.zeros((1, 8, 8, 3)), pose_class, '', '', 2.30, '',
                                      'grey', 1., 101, json.dumps(objs), n_candidates=4, pick=pick)
    return out, calls


def test_node_picks_best_and_pick(monkeypatch):
    N = load_nodes()
    out, calls = run_node(N, monkeypatch, {'surf_1': 'wrong_surface', 'surf_4': 'ok'})
    assert [c['instance_hint'] for c in calls] == ['surf_1', 'surf_4', '']
    assert len(out) == 12
    first, second = out[5].split('\n', 1)
    data = json.loads(second)
    assert data['candidate_index'] == 1 and data['status'] == 'ok' and data['support_object'] == 'surf_4:bed'
    assert 'кандидат 1/3' in first
    summary = json.loads(out[10])
    assert [c['status'] for c in summary['candidates']] == ['wrong_surface', 'ok', 'fallback']
    assert out[11].shape[0] == 1 and out[11].shape[-1] == 3
    assert out[9] == {'n': 2}                                 # body_guides выбранного кандидата
    out, _ = run_node(N, monkeypatch, {'surf_1': 'wrong_surface', 'surf_4': 'ok'}, pick=0)
    assert json.loads(out[5].split('\n', 1)[1])['candidate_index'] == 0


def test_node_never_raises(monkeypatch):
    N = load_nodes()
    monkeypatch.setattr(N, 'candidates_helpers', lambda: (_ for _ in ()).throw(RuntimeError('boom')))
    out = N.PlaceCandidates().execute('x', torch.zeros((1, 8, 8, 3)), 'stand', '', '', 2.30, '', 'grey', 1., 101, '[]')
    assert len(out) == 12 and out[5].startswith('status=fallback') and json.loads(out[10])['error']
    out, _ = run_node(N, monkeypatch, {}, pose_class='stand')        # все кандидаты упали — запасной путь
    assert out[5].startswith('status=fallback')


def test_retries_zone_relax_once():
    sofa = {'variant': 'support', 'label': 'sofa', 'support_id': 'surf_1'}
    out = C.retries(sofa, 'no_target', ['zone_too_small'], CFG)
    assert len(out) == 1 and out[0]['zone_min_frac'] == CFG['zone_relax_frac'] and out[0]['variant'] == 'support+zone'
    assert C.retries(out[0], 'no_target', ['zone_too_small'], CFG) == []          # только один раз
    assert C.retries(sofa, 'no_target', ['other_flag'], CFG) == []
    assert C.retries(dict(sofa, label='floor'), 'no_target', ['zone_too_small'], CFG) == []
    assert C.score(sofa, 'fallback', {}, 'sit_edge', CFG)[0] == 0.0            # пустой вариант — 0


# ------------------------------------------------------------------------------------------------ candidates_cli
def test_cli_zone_relax_and_cache_key(monkeypatch):
    import candidates_cli as cc
    import manual_pose
    import place
    import reference_place
    for mod, name in ((place, 'zone_too_small'), (place, 'place'), (reference_place, 'place'),
                      (manual_pose, 'placement_key')):
        monkeypatch.setattr(mod, name, getattr(mod, name))                    # вернуть после теста
    monkeypatch.setattr(manual_pose, 'placement_key', lambda *a, **k: 'abc')
    d = {'zone_mask_area_m2': 0.2, 'body_footprint_area_m2': 0.3}
    assert place.zone_too_small(d)                                             # штатно: 0.2 < 0.3 — нет цели
    cc.patch({}, {'farm_root': '.', 'comfy': '.'}, {})
    assert place.zone_too_small(d) and manual_pose.placement_key() == 'abc'    # без опции — как было
    cc.patch({'e1_zone_min_frac': 0.5}, {'farm_root': '.', 'comfy': '.'}, {})
    assert not place.zone_too_small(d) and place.zone_too_small(dict(d, zone_mask_area_m2=0.1))
    assert manual_pose.placement_key() != 'abc'                                # свой ключ кэша


def test_cli_returns_report_on_no_target(monkeypatch):
    import candidates_cli as cc
    import place
    import reference_place
    rep = {'status': 'no_target', 'flags': ['zone_too_small'], 'shot_spec': {'pose_class': 'sit_edge'},
           'zone_diagnostics': {'zone_mask_area_m2': 0.2, 'body_footprint_area_m2': 0.3, 'support_hull': {'x': 1}}}
    monkeypatch.setattr(place, 'place', lambda *a, **k: {'status': 'no_target', 'rep': rep})
    monkeypatch.setattr(reference_place, 'place', reference_place.place)

    def do_place(request, config):
        place.place()
        raise ValueError('placement has no body geometry (status=no_target)')
    monkeypatch.setattr(cc.pack_cli, 'do_place', do_place)
    out = cc.geometry_worker({}, {'farm_root': '.', 'comfy': '.'})
    assert out['status'] == 'error' and out['place_status'] == 'no_target'
    assert out['report']['flags'] == ['zone_too_small']
    assert out['report']['zone_diagnostics'] == {'zone_mask_area_m2': 0.2, 'body_footprint_area_m2': 0.3,
                                                 'support_vertex_count': None, 'zero_support_cause': None}


def test_node_no_target_report_and_retry(monkeypatch):
    N = load_nodes()
    seen = []

    def run_cli(script, command, fields, images, timeout=600, check=True):
        seen.append((script, fields.get('instance_hint'), fields.get('e1_zone_min_frac'), timeout, check))
        return {'status': 'error', 'error': 'placement has no body geometry (status=no_target)', 'place_status': 'no_target',
                'report': {'status': 'no_target', 'flags': ['zone_too_small'],
                           'zone_diagnostics': {'zone_mask_area_m2': 0.2, 'body_footprint_area_m2': 0.3}}}
    monkeypatch.setattr(N, 'run_cli', run_cli)
    objs = room()
    out = N.PlaceCandidates().execute('missing/profile.json', torch.zeros((1, 8, 8, 3)), 'sit_edge', '', '', 2.30, '',
                                      'grey', 1., 101, json.dumps(objs), n_candidates=4)
    assert out[5].startswith('status=fallback') and len(out) == 12
    assert all(s[0].endswith('candidates_cli.py') and s[3] == CFG['cli_timeout_s'] and s[4] is False for s in seen)
    # диван, диван+мягкая зона, кровать, кровать+мягкая зона, запасной «пол» (пол без повтора)
    assert [(s[1], s[2]) for s in seen] == [('surf_1', None), ('surf_1', .5), ('surf_4', None), ('surf_4', .5), ('', None)]
    summary = json.loads(out[10])
    assert summary['candidates'][0]['place_status'] == 'no_target'
    assert summary['candidates'][0]['zone']['zone_mask_area_m2'] == 0.2
    assert N.PlacePose().runner.__func__ is N.PlacePose.runner                 # у старого узла — прежний invoke
