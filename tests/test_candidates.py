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
        {'id': 'surf_3', 'type': 'other', 'height_m': 0.72, 'area_m2': 0.2, 'cells_xy': cells(-1.5, 1, -1.1, 1.4)},
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
    assert lab['surf_3'] == 'nightstand'    # маленький верх 0.72 м (выше сиденья)
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
    lie = C.plan('lie', objs, lab, {}, 2, CFG, 2.30)
    assert lie and lie[0]['support_parts'] == 'pelvis,torso'
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


def test_candidate_cli_adds_camera_depth(monkeypatch, tmp_path):
    sys.path.insert(0, str(ROOT / 'tools/scene3d'))
    import candidates_cli as CLI

    class Plate:
        def __init__(self, path):
            self.R = np.array([[0., 0., -1.], [1., 0., 0.], [0., -1., 0.]])

        def project(self, points):
            return np.array([0.]), np.array([0.]), np.array([1.25])

    PF = type('PF', (), {'Plate': Plate})

    monkeypatch.setattr(CLI.pack_cli, 'required_file', lambda value, label: tmp_path / 'profile.json')
    monkeypatch.setattr(CLI.pack_cli, 'read_json', lambda path: {
        'loc_id': 'loc', 'objects': [{'id': 'surf_1', 'center_m': [1., 2., .5]}]})
    monkeypatch.setattr(CLI.pack_cli, 'location_id', lambda value: value)
    monkeypatch.setattr(CLI.pack_cli, 'assets_for', lambda config, loc_id, profile_dir: tmp_path)
    monkeypatch.setattr(CLI.pack_cli, 'modules', lambda config: (PF, None))
    result = {'report': {'target_instance': 'surf_1'}}
    assert CLI.add_support_depth({'location': 'profile'}, {}, result) is result
    assert result['report']['support_depth_m'] == 1.25
    assert np.allclose(result['report']['candidate_offset_basis_xy'], [[0., -1.], [1., 0.]])


def test_principal_axis_for_same_support_offsets():
    result = C.principal_axis(cells(0, 0, 1.2, 0.3, .1), CFG)
    assert result is not None
    axis, length, center = result
    assert abs(abs(axis[0]) - 1.0) < 1e-6 and length > 1.0
    assert np.allclose(center, [0.55, 0.1], atol=0.02)


def test_derived_frame_shifts_require_bad_frame_evidence():
    cand = {'variant': 'support', 'support_id': 'surf_1', 'label': 'sofa', 'body_dx': 0., 'body_dy': 0.}
    missing = C.derived([cand], [{'report': {}}], CFG, 1, [], None, cand)
    assert missing == []
    bad = {'visual_metrics': {'height_fraction': 1.125, 'cropped_edges': ['right', 'bottom']}}
    shifted = C.derived([cand], [{'report': bad}], CFG, 1, [], None, cand)
    assert len(shifted) == len(CFG['frame_shifts_m']) + len(CFG['frame_depth_shifts_m']['farther'])
    assert shifted[1]['support_id'] == 'surf_1' and shifted[1]['body_dx'] == CFG['frame_shifts_m'][1][0]
    assert [x['body_dy'] for x in shifted[-len(CFG['frame_depth_shifts_m']['farther']):]] == \
        CFG['frame_depth_shifts_m']['farther']


def test_derived_rotated_support_centers_fitted_pelvis_and_shifts_thirds():
    points = [[x - y, x + y] for x, y in cells(0, 0, 2.1, .3, .1)]
    axis, length, center = C.principal_axis(points, CFG)
    cand = {'variant': 'support', 'support_id': 'surf_1', 'label': 'sofa', 'body_dx': .2, 'body_dy': -.1,
            'support_axis_xy': axis, 'support_length_m': length, 'support_center_xy': center}
    basis = np.array([[0., -1.], [1., 0.]])
    pelvis = np.array([0.4, -0.2])
    bad = {'visual_metrics': {'visible_fraction': .3}, 'candidate_offset_basis_xy': basis.tolist(),
           'candidate_pelvis_xy': pelvis.tolist()}
    shifted = C.derived([cand], [{'report': bad}], CFG, 1, [], None, cand)
    assert len(shifted) == len(CFG['frame_shifts_thirds'])
    for entry, fraction in zip(shifted, CFG['frame_shifts_thirds']):
        delta_camera = np.array([entry['body_dx'] - .2, entry['body_dy'] + .1])
        world_result = pelvis + basis @ delta_camera
        expected = np.array(center) + np.array(axis) * length * fraction / 3
        assert np.allclose(world_result, expected)
        assert entry['support_id'] == cand['support_id']
    # Without exact fitted-pelvis/basis data, fallback remains camera-relative.
    for invalid in (None, [[0., 0.], [0., 0.]], [[float('nan'), 0.], [0., 1.]]):
        bad['candidate_offset_basis_xy'] = invalid
        fallback = C.derived([cand], [{'report': bad}], CFG, 1, [], None, cand)
        delta = [fallback[1]['body_dx'] - .2, fallback[1]['body_dy'] + .1]
        assert np.allclose(delta, CFG['frame_shifts_m'][1])


def test_derived_depth_shifts_follow_frame_height():
    cand = {'variant': 'support', 'support_id': 'floor_1', 'label': 'floor', 'body_dx': 0., 'body_dy': .1}
    too_large = {'report': {'visual_metrics': {'height_fraction': .95, 'cropped_edges': ['top'],
                                               'visible_fraction': 1.0}}}
    farther = C.derived([cand], [too_large], CFG, 1, [], None, cand)
    assert [x['body_dy'] for x in farther[-len(CFG['frame_depth_shifts_m']['farther']):]] == [
        0.1 + x for x in CFG['frame_depth_shifts_m']['farther']]
    too_small = {'report': {'visual_metrics': {'height_fraction': .10, 'cropped_edges': [],
                                               'visible_fraction': 1.0}}}
    closer = C.derived([cand], [too_small], CFG, 1, [], None, cand)
    assert [x['body_dy'] for x in closer[-len(CFG['frame_depth_shifts_m']['closer']):]] == [
        0.1 - x for x in CFG['frame_depth_shifts_m']['closer']]


def test_derived_depth_variant_is_created_for_each_small_candidate():
    good = {'variant': 'support', 'support_id': 'surf_1', 'label': 'sofa', 'body_dx': 0., 'body_dy': 0.}
    small = {'variant': 'support', 'support_id': 'surf_2', 'label': 'bench', 'body_dx': .2, 'body_dy': .3}
    reports = [
        {'report': {'visual_metrics': {'height_fraction': .5, 'cropped_edges': [], 'visible_fraction': .9}}},
        {'report': {'visual_metrics': {'height_fraction': .1, 'cropped_edges': [], 'visible_fraction': .9}}},
    ]
    out = C.derived([good, small], reports, CFG, 2, [], None, good)
    depth = [x for x in out if x['variant'].startswith('frame_depth')]
    assert len(depth) == len(CFG['frame_depth_shifts_m']['closer'])
    assert all(x['support_id'] == 'surf_2' for x in depth)
    assert [x['body_dy'] for x in depth] == [.3 - d for d in CFG['frame_depth_shifts_m']['closer']]


def test_derived_recline_pitch_variants_for_high_torso_gap():
    cand = {'variant': 'support', 'support_id': 'surf_4', 'label': 'bed', 'support_parts': 'pelvis,torso',
            'body_dx': 0., 'body_dy': 0., 'body_dpitch': 0.}
    report = {'visual_metrics': {'height_fraction': .55, 'cropped_edges': [], 'visible_fraction': .95},
              'contacts': {'torso': {'role': 'secondary', 'gap_cm': 31.9}}}
    out = C.derived([cand], [{'report': report}], CFG, 1, [], None, cand)
    assert [x['body_dpitch'] for x in out] == CFG['recline_pitch_shifts_deg']
    assert all(x['origin'] == 'derived' and x['support_id'] == 'surf_4' for x in out)


def test_derived_occlusion_only_and_all_candidates_requirement():
    cand = {'variant': 'support', 'support_id': 'surf_1', 'label': 'sofa', 'body_dx': 0., 'body_dy': 0.,
            'support_axis_xy': [0., 1.], 'support_length_m': 1.8}
    frame = {'bbox_px': [100, 100, 400, 600], 'frame_px': [800, 800], 'visible': True, 'ok': True}
    hidden = {'report': {'frame': frame, 'visual_metrics': {'visible_fraction': .3}}}
    good = {'report': {'frame': frame, 'visual_metrics': {'visible_fraction': .8}}}
    unknown = {'report': {'frame': frame}}
    shifted = C.derived([cand], [hidden], CFG, 1, [], None, cand)
    assert len(shifted) == len(CFG['frame_shifts_m'])
    assert np.allclose([shifted[1]['body_dx'], shifted[1]['body_dy']], CFG['frame_shifts_m'][1])
    assert C.derived([cand, cand], [hidden, hidden], CFG, 2, [], None, cand)
    assert C.derived([cand, cand], [hidden, good], CFG, 2, [], None, cand) == []
    assert C.derived([cand, cand], [hidden, unknown], CFG, 2, [], None, cand) == []
    assert C.derived([cand, cand], [hidden], CFG, 2, [], None, cand) == []
    threshold = {'report': {'visual_metrics': {'visible_fraction': CFG['visibility']['min_visible_frac']}}}
    assert C.derived([cand], [threshold], CFG, 1, [], None, cand) == []


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
    # E1c: отчёт без измерения «в кадре» — ошибка измерения (screen_measured 0), не идеал
    assert set(comp) == {'status', 'gap', 'penetration', 'pose_match', 'support_class', 'visibility',
                         'frame_height', 'crop', 'depth', 'screen_factor', 'screen_measured'}
    assert comp['screen_measured'] == 0.0 and comp['screen_factor'] == CFG['screen']['measure_error_factor']
    def visual(height, edges, visible=1.0):
        return {**rep('ok'), 'visual_metrics': {'height_fraction': height, 'cropped_edges': edges,
                                                'visible_fraction': visible}}
    _, oversized = C.score(sofa, 'ok', visual(.96, ['top']), 'sit_edge', CFG)
    assert oversized['frame_height'] == 0.0 and oversized['crop'] == 0.0
    _, closeup = C.score(sofa, 'ok', visual(.86, []), 'sit_edge', CFG)
    assert closeup['frame_height'] == 0.0 and closeup['crop'] == 1.0
    _, lower_edge = C.score(sofa, 'ok', visual(1.0, ['bottom']), 'sit_edge', CFG)
    assert lower_edge['crop'] == 1.0
    _, small_bottom_crop = C.score(sofa, 'ok', visual(.3, ['bottom']), 'sit_edge', CFG)
    assert small_bottom_crop['crop'] == 0.0
    _, mostly_hidden = C.score(sofa, 'ok', visual(.5, [], .3), 'sit_edge', CFG)
    assert mostly_hidden['visibility'] == 0.5
    _, unknown_visual = C.score(sofa, 'ok', rep('ok'), 'sit_edge', CFG)
    assert unknown_visual['visibility'] == CFG['visual_unknown_score']
    assert unknown_visual['frame_height'] == CFG['visual_unknown_score']
    assert unknown_visual['crop'] == CFG['visual_unknown_score']
    _, shallow = C.score(sofa, 'ok', {**rep('ok'), 'support_depth_m': 1.2}, 'sit_edge', CFG)
    assert shallow['depth'] == 0.0
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
        calls.append(dict(target_object=target_object, instance_hint=instance_hint, body_dyaw=body_dyaw,
                          body_dpitch=body_dpitch, body_droll=body_droll))
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


def test_node_passes_derived_pitch_and_roll(monkeypatch):
    N = load_nodes()
    monkeypatch.delenv('THREEDA_CAND_CACHE', raising=False)
    monkeypatch.setattr(N, 'candidates_helpers', lambda: C)
    cand = {'variant': 'recline_pitch', 'target_object': 'bed', 'instance_hint': 'surf_4', 'support_id': 'surf_4',
            'label': 'bed', 'support_parts': 'pelvis,torso', 'secondary_contacts': '', 'height_m': 2.3,
            'body_dx': 0., 'body_dy': 0., 'body_dz': 0., 'body_dyaw': 0., 'body_dpitch': 20., 'body_droll': -3.}
    monkeypatch.setattr(C, 'plan', lambda *args, **kwargs: [cand])
    execute, calls = fake_execute({'surf_4': 'ok'})
    monkeypatch.setattr(N.PlacePose, 'execute', execute)
    N.PlaceCandidates().execute('missing/profile.json', torch.zeros((1, 8, 8, 3)), 'recline', '', '', 2.30, '',
                                 'grey', 1., 101, json.dumps(room()), n_candidates=1)
    assert calls[0]['body_dpitch'] == 20. and calls[0]['body_droll'] == -3.


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
    # с классом позы — и опорные части из zone_relax_support_parts
    out = C.retries(sofa, 'no_target', ['zone_too_small'], CFG, 'fallback', {}, 'sit_edge')
    assert out[0]['support_parts'] == ','.join(CFG['zone_relax_support_parts']['sit_edge'])
    assert 'support_parts' not in C.retries(sofa, 'no_target', ['zone_too_small'], CFG, 'fallback', {}, 'recline')[0]


def test_small_seat_fallback_only():
    """Маленький верх на высоте сиденья — «seat»: опора сидя только если другой мебели нет; лёжа — нет."""
    seat = {'id': 'surf_9', 'type': 'other', 'height_m': 0.55, 'area_m2': 0.16, 'cells_xy': cells(3, 3, 3.5, 3.4)}
    tall = {'id': 'surf_8', 'type': 'other', 'height_m': 0.73, 'area_m2': 0.18, 'cells_xy': cells(-3, 3, -2.5, 3.4)}
    floor = room()[0]
    objs = [floor, seat, tall]
    lab = C.classify(objs, CFG)
    assert lab['surf_9']['label'] == 'seat' and lab['surf_8']['label'] == 'nightstand'
    assert [s['id'] for s in C.supports_for('sit_edge', objs, lab, {}, CFG)] == ['surf_9']
    assert C.supports_for('lie', objs, lab, {}, CFG)[0]['label'] == 'floor'
    assert all(s['label'] != 'seat' for s in C.supports_for('lie', objs, lab, {}, CFG))
    objs = room() + [seat]                                   # есть диван/кровать — «seat» не нужен
    lab = C.classify(objs, CFG)
    assert all(s['label'] != 'seat' for s in C.supports_for('sit_edge', objs, lab, {}, CFG))
    c = {'label': 'seat', 'body_dyaw': 0.0}
    assert C.score(c, 'ok', {}, 'sit_edge', CFG)[1]['support_class'] == CFG['score']['support_tier2']


def sup_rep(*gaps_cm):
    return {'contacts': {f'p{i}': {'role': 'support', 'gap_cm': g} for i, g in enumerate(gaps_cm)} |
            {'foot.L': {'role': 'secondary', 'gap_cm': 30.0}}}


def test_settle_offset():
    s = CFG['settle']
    # тело над сиденьем на 4.3–4.6 см: дальняя часть — до target_m
    assert abs(C.settle_offset('wrong_surface', sup_rep(4.6, 4.3, 4.4), CFG) + (0.046 - s['target_m'])) < 1e-6
    # ближняя часть почти на опоре — сдвиг ограничен sink_m
    assert abs(C.settle_offset('contact_fail', sup_rep(0.5, 9.0), CFG) + (0.005 + s['sink_m'])) < 1e-6
    assert C.settle_offset('contact_fail', sup_rep(30.0, 30.0), CFG) == -s['max_m']
    assert C.settle_offset('ok', sup_rep(4.6), CFG) is None                     # статус и так годный
    assert C.settle_offset('wrong_surface', sup_rep(-1.0, 3.0), CFG) is None    # уже в опоре — не опускать
    assert C.settle_offset('wrong_surface', sup_rep(2.2), CFG) is None          # меньше min_m
    assert C.settle_offset('wrong_surface', {}, CFG) is None


def test_retries_settle_rounds():
    bed = {'variant': 'support', 'label': 'bed', 'support_id': 'surf_4', 'body_dz': 0.0}
    out = C.retries(bed, 'wrong_surface', [], CFG, 'wrong_surface', sup_rep(4.6, 4.3), None)
    assert len(out) == 1 and out[0]['variant'] == 'support+settle' and out[0]['body_dz'] < 0
    c = out[0]
    for _ in range(CFG['settle']['rounds'] - 1):
        c = C.retries(c, 'wrong_surface', [], CFG, 'wrong_surface', sup_rep(4.6), None)[0]
    assert C.retries(c, 'wrong_surface', [], CFG, 'wrong_surface', sup_rep(4.6), None) == []   # лимит rounds
    assert C.retries(dict(bed, label='floor'), 'wrong_surface', [], CFG, 'wrong_surface', sup_rep(4.6), None) == []


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
    monkeypatch.setattr(place, 'ATTEMPTS', place.ATTEMPTS)
    cc.patch({'e1_zone_min_frac': 0.5, 'e1_attempts': 2}, {'farm_root': '.', 'comfy': '.'}, {})
    assert place.ATTEMPTS == 2
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
    z = CFG['zone_relax_frac']
    assert [(s[1], s[2]) for s in seen] == [('surf_1', None), ('surf_1', z), ('surf_4', None), ('surf_4', z), ('', None)]
    summary = json.loads(out[10])
    assert summary['candidates'][0]['place_status'] == 'no_target'
    assert summary['candidates'][0]['zone']['zone_mask_area_m2'] == 0.2
    assert N.PlacePose().runner.__func__ is N.PlacePose.runner                 # у старого узла — прежний invoke


def test_small_seat_with_back_is_sofa_not_nightstand():
    # виден только средний кусок сиденья дивана (0.17 м²) и верх спинки — это диван, не тумба
    objs = [{'id': 'floor_1', 'type': 'floor', 'height_m': 0.0, 'area_m2': 3.0, 'cells_xy': cells(-2, -2, 2, 0, .25)},
            {'id': 'surf_4', 'type': 'other', 'height_m': 0.49, 'area_m2': 0.17, 'cells_xy': cells(0, 1, .45, 1.4)},
            {'id': 'surf_5', 'type': 'other', 'height_m': 0.93, 'area_m2': 0.42, 'cells_xy': cells(-1, 1.6, 1.4, 1.8)}]
    lab = labels_of(objs)
    assert lab['surf_4'] == 'sofa' and lab['surf_5'] == 'backrest'


def test_floor_choice_without_profile_floor():
    objs = [{'id': 'surf_1', 'type': 'other', 'height_m': -0.05, 'area_m2': 0.7},
            {'id': 'surf_2', 'type': 'other', 'height_m': 0.01, 'area_m2': 6.4},
            {'id': 'surf_3', 'type': 'other', 'height_m': 0.52, 'area_m2': 1.0}]
    lab = C.classify(objs, CFG)
    assert C.floor_choice(objs, lab) == 'surf_2'
    stand = C.plan('stand', objs, lab, {}, 1, CFG, 2.30)[0]
    assert (stand['target_object'], stand['instance_hint'], stand['floor_id']) == ('floor', '', 'surf_2')
    sit = C.plan('sit_edge', objs, lab, {}, 4, CFG, 2.30)
    assert sit[-1]['variant'] == 'baseline' and sit[-1]['floor_id'] == 'surf_2'
    assert C.floor_choice(room(), C.classify(room(), CFG)) is None          # свой floor_1 — штатно
    # куски пола одной опорой — кроме stand; кусок дальше merge_dh_m по высоте — отдельно
    assert stand.get('merge_ids') is None
    lie = C.plan('lie', objs + [{'id': 'surf_4', 'type': 'other', 'height_m': -0.10, 'area_m2': 0.3}],
                 C.classify(objs + [{'id': 'surf_4', 'type': 'other', 'height_m': -0.10, 'area_m2': 0.3}], CFG),
                 {}, 4, CFG, 2.30)
    floor = next(c for c in lie if c['label'] == 'floor')
    assert floor['floor_id'] == 'surf_2' and floor['merge_ids'] == 'surf_2,surf_1'
    assert C.floor_merge_ids(None, objs, lab, CFG) is None


def test_low_long_narrow_is_bench():
    objs = [{'id': 'floor_1', 'type': 'floor', 'height_m': 0.0, 'area_m2': 6.0},
            {'id': 'surf_1', 'type': 'other', 'height_m': 0.17, 'area_m2': 0.9, 'cells_xy': cells(0, 0, 2.8, .45, .05)},
            {'id': 'surf_2', 'type': 'other', 'height_m': 0.17, 'area_m2': 2.0, 'cells_xy': cells(3, 3, 5, 4, .05)}]
    lab = labels_of(objs)
    assert lab['surf_1'] == 'bench' and lab['surf_2'] != 'bench'           # широкий низкий верх — не скамья


def test_visibility_and_yaw_penalty():
    assert C.visibility_fraction({'visual_metrics': {'visible_fraction': .42}}) == .42
    assert C.visibility_fraction({'frame': {'ok': True}, 'keypoints_body25_px': [[0, 0, 1]]}) is None
    base, _ = C.score({'label': 'floor'}, 'ok', rep('ok'), 'stand', CFG)
    turned, _ = C.score({'label': 'floor', 'body_dyaw': 25.0}, 'ok', rep('ok'), 'stand', CFG)
    tilted, _ = C.score({'label': 'floor', 'body_dpitch': 20.0, 'body_droll': -5.0}, 'ok', rep('ok'), 'stand', CFG)
    assert turned < base and tilted < base                                    # ручной поворот/наклон — отход от референса


def test_cli_floor_id(monkeypatch, tmp_path):
    import candidates_cli as cc
    import manual_pose
    import profiles
    for mod, name in ((profiles.LocationProfile, '__init__'), (manual_pose, 'placement_key')):
        monkeypatch.setattr(mod, name, getattr(mod, name))
    import place
    import reference_place
    monkeypatch.setattr(place, 'place', place.place)
    monkeypatch.setattr(reference_place, 'place', reference_place.place)
    monkeypatch.setattr(manual_pose, 'placement_key', lambda *a, **k: 'abc')
    data = {'loc_id': 'x', 'version': 2, 'reviewed': False, 'scale': {}, 'anchors': [], 'substitutes': {},
            'auto_surfaces': {'floor_status': 'floor_ambiguous'},
            'objects': [{'id': 'surf_1', 'name': 'other', 'type': 'other', 'top_grid_npz': 'a.npz'},
                        {'id': 'surf_2', 'name': 'other', 'type': 'other', 'top_grid_npz': 'b.npz'}]}
    (tmp_path / 'profile.json').write_text(json.dumps(data), encoding='utf-8')
    cc.patch({'e1_floor_id': 'surf_2'}, {'farm_root': '.', 'comfy': '.'}, {})
    prof = profiles.LocationProfile(tmp_path / 'profile.json')
    assert [o['type'] for o in prof.data['objects']] == ['other', 'floor']
    assert prof.data['auto_surfaces']['floor_status'] == 'ok'
    assert prof.find_object('floor')[0]['id'] == 'surf_2'
    assert manual_pose.placement_key() != 'abc'


def test_merge_ids_for_bed_pieces():
    objs = room()
    lab = C.classify(objs, CFG)
    assert C.merge_ids('surf_4', objs, lab, CFG) == 'surf_4,surf_5'          # 0.52 и 0.55 — одна кровать
    assert C.merge_ids('surf_1', objs, lab, CFG) is None                      # спинка 0.92 — не та высота
    sit = C.plan('sit_edge', objs, lab, {}, 4, CFG, 2.30)
    assert sit[1]['merge_ids'] == 'surf_4,surf_5' and sit[0].get('merge_ids') is None
    assert sit[0]['attempts'] == CFG['furniture_attempts'] and sit[-1].get('attempts') is None   # пол — штатно


def test_cli_merge_support(monkeypatch, tmp_path):
    import candidates_cli as cc
    import manual_pose
    import place
    import profiles
    import reference_place
    import top_region as tr
    for mod, name in ((profiles.LocationProfile, '__init__'), (profiles.LocationProfile, 'region'),
                      (place, 'match_auto_support'), (place, 'place'), (reference_place, 'place'),
                      (manual_pose, 'placement_key')):
        monkeypatch.setattr(mod, name, getattr(mod, name))
    shape, origin = (40, 40), np.array([0.0, 0.0])
    objects = []
    for k, (rows, h) in enumerate(((slice(0, 20), .52), (slice(20, 40), .55), (slice(0, 5), .29)), 1):
        mask = np.zeros(shape, bool)
        mask[rows, 5:35] = True
        reg = tr.TopRegion('other', 'auto', origin, shape, np.where(mask, h, np.nan), mask, np.zeros(shape), h, (1, 6),
                           np.zeros(2), object_mask=mask.copy())
        tr.save_npz(reg, tmp_path / f'surf_{k}_top.npz')
        objects.append({'id': f'surf_{k}', 'name': 'other', 'type': 'other', 'top_grid_npz': f'surf_{k}_top.npz'})
    data = {'loc_id': 'x', 'version': 2, 'reviewed': False, 'scale': {}, 'anchors': [], 'substitutes': {},
            'objects': objects}
    (tmp_path / 'profile.json').write_text(json.dumps(data), encoding='utf-8')
    cc.patch({'e1_merge_ids': 'surf_2,surf_1'}, {'farm_root': '.', 'comfy': '.'}, {})
    prof = profiles.LocationProfile(tmp_path / 'profile.json')
    assert [o['id'] for o in prof.data['objects']] == ['surf_2', 'surf_3']   # surf_1 влит в surf_2
    reg = prof.region(prof.data['objects'][0])
    assert reg.mask.sum() == 40 * 30 and reg.object_id == 'surf_2' and abs(reg.h0 - .535) < .02
    assert place.match_auto_support(None, reg, 2.3)[0] is None               # не перевыбирать один кусок
    assert prof.region(prof.data['objects'][1]).mask.sum() == 5 * 30          # чужие куски — как были


def test_long_narrow_seat_is_bench():
    objs = [{'id': 'floor_1', 'type': 'floor', 'height_m': 0.0, 'area_m2': 12.0},
            {'id': 'surf_1', 'type': 'other', 'height_m': 0.51, 'area_m2': 0.17, 'cells_xy': cells(0, 0, 1.3, .27, .05)},
            {'id': 'surf_2', 'type': 'other', 'height_m': 0.55, 'area_m2': 0.16, 'cells_xy': cells(3, 3, 3.45, 3.4, .05)}]
    lab = labels_of(objs)
    assert lab['surf_1'] == 'bench' and lab['surf_2'] == 'seat'      # маленький верх 0.55 — «seat», не тумба
    table = [{'id': 'floor_1', 'type': 'floor', 'height_m': 0.0},
             {'id': 'surf_6', 'type': 'other', 'height_m': 0.78, 'area_m2': 0.17, 'cells_xy': cells(0, 0, 1.6, .24, .05)}]
    assert labels_of(table)['surf_6'] == 'table'                                # массажный стол, не тумба
    assert C.support_rank('lie', 'table', CFG)[0] == 1 and C.support_rank('sit_edge', 'table', CFG) is None
