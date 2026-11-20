"""E1c: проверка «в кадре» (tools/scene3d/posev2/candidate_screen.py) и её место в оценке кандидатов, на синтетике.
Без GPU и сервера. Запуск: python -m pytest tests/test_candidate_screen.py -q"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools/scene3d/posev2')]
import candidate_screen as S  # noqa: E402
import candidates as C  # noqa: E402

CFG = S.load_config()
CAND = C.load_config()


class FakePlate:
    """Камера (0, 0, 1.5) смотрит вдоль +Y вниз на pitch°, плоский пол z = 0; контракт pose_fit.Plate."""
    Wp, Hp = 320, 240

    def __init__(self, pitch_deg=20.0, floor_z=0.0, hfov_deg=70.0):
        p = math.radians(pitch_deg)
        forward = np.array([0.0, math.cos(p), -math.sin(p)])
        right = np.array([1.0, 0.0, 0.0])
        down = np.cross(forward, right)          # cv y вниз
        self.R = np.column_stack([right, down, forward])
        self.C = np.array([0.0, 0.0, 1.5])
        self.fx = (self.Wp / 2) / math.tan(math.radians(hfov_deg) / 2)
        self.cx, self.cy = self.Wp / 2, self.Hp / 2
        u, v = np.meshgrid(np.arange(self.Wp), np.arange(self.Hp))
        rays = np.stack([(u - self.cx) / self.fx, (v - self.cy) / self.fx, np.ones_like(u, float)], -1) @ self.R.T
        with np.errstate(divide='ignore', invalid='ignore'):
            t = (floor_z - self.C[2]) / rays[..., 2]
        self.Z = np.where(rays[..., 2] < 0, t, np.inf)

    def project(self, X):
        c = (np.asarray(X, float) - self.C) @ self.R
        z = c[:, 2]
        with np.errstate(divide='ignore', invalid='ignore'):
            return self.fx * c[:, 0] / z + self.cx, self.fx * c[:, 1] / z + self.cy, z


def patch(cx, cy, z, half=0.15, n=6):
    g = np.linspace(-half, half, n)
    return np.array([[cx + a, cy + b, z] for a in g for b in g])


def lying_body(z=0.05, cy=3.0, shift_x=0.0):
    """Тело лёжа: таз и торс на высоте z (контакт), голова чуть выше; регионы как у body.regions()."""
    parts = {'pelvis': patch(shift_x, cy, z), 'torso': patch(shift_x, cy + 0.5, z),
             'head': patch(shift_x, cy + 1.0, z + 0.1, 0.08, 4), 'thigh.L': patch(shift_x, cy - 0.5, z + 0.02)}
    X = np.vstack(list(parts.values()))
    reg = np.concatenate([[name] * len(points) for name, points in parts.items()])
    return X, reg


def test_contact_on_surface_is_ok():
    X, reg = lying_body(z=0.0)
    out = S.measure(FakePlate(), X, reg, 'lie', ['pelvis', 'torso'], CFG)
    assert out['measure_error'] == [] and out['frame_ok'] is True and out['in_frame_frac'] == 1.0
    assert out['contact_status'] == 'contact' and abs(out['screen_gap_m']) < 0.02
    assert out['auto_flag'] == 'ok' and set(out['parts']) == {'pelvis', 'torso'}


def test_floating_gap_and_vertical_settle():
    plate = FakePlate()
    X, reg = lying_body(z=0.15)
    out = S.measure(plate, X, reg, 'lie', ['pelvis', 'torso'], CFG)
    assert out['contact_status'] == 'floating' and out['screen_gap_m'] > CFG['floating_gap_m']
    assert out['vertical_gap_m'] == pytest.approx(0.15, abs=0.01)      # высота над поверхностью, которую видит камера
    assert out['auto_flag'] == 'floating' and 'floating' in out['auto_flag']
    dz = S.settle_dz(out, CFG)
    assert dz == pytest.approx(-(0.15 - CFG['settle']['target_m']), abs=0.01)
    # после сдвига на dz контакт проходит
    X2 = X + np.array([0.0, 0.0, dz])
    again = S.measure(plate, X2, reg, 'lie', ['pelvis', 'torso'], CFG)
    assert again['contact_status'] == 'contact'


def test_penetration_and_upward_settle():
    plate = FakePlate()
    X, reg = lying_body(z=-0.12)
    out = S.measure(plate, X, reg, 'lie', [], CFG)                       # части класса из YAML
    assert out['contact_status'] == 'penetrating' and out['screen_gap_m'] < -CFG['penetrating_gap_m']
    assert out['auto_flag'] == 'penetrating'
    dz = S.settle_dz(out, CFG)
    assert dz is not None and 0 < dz <= CFG['settle']['max_up_m']


def test_small_gap_below_threshold_is_not_floating_and_no_settle():
    out = S.measure(FakePlate(), *lying_body(z=0.02), 'lie', ['pelvis', 'torso'], CFG)
    assert out['contact_status'] == 'contact' and S.settle_dz(out, CFG) is None


def test_gap_scaled_by_room_scale_and_recorded():
    plate = FakePlate()
    X, reg = lying_body(z=0.012)
    base = S.measure(plate, X, reg, 'lie', [], CFG)
    scaled = S.measure(plate, X, reg, 'lie', [], CFG, room_scale=3.0)
    assert base['scale_source'] == 'plate_units' and scaled['scale_source'] == 'room_scale'
    for name in base['parts']:
        assert scaled['parts'][name]['screen_gap_m'] == pytest.approx(3 * base['parts'][name]['screen_gap_m'], rel=2e-2)
    assert scaled['contact_status'] == 'floating' and base['contact_status'] == 'contact'


def test_worst_part_decides_not_pooled_median():
    X, reg = lying_body(z=0.0)
    X = X.copy()
    X[reg == 'torso', 2] += 0.2                                         # торс висит, таз на опоре
    out = S.measure(FakePlate(), X, reg, 'lie', ['pelvis', 'torso'], CFG)
    assert out['contact_status'] == 'floating'
    assert out['parts']['pelvis']['screen_gap_m'] < CFG['floating_gap_m'] < out['parts']['torso']['screen_gap_m']


def test_stand_uses_lowest_foot_only():
    left, right = patch(-0.2, 3.0, 0.0, 0.05, 3), patch(0.2, 3.0, 0.4, 0.05, 3)    # правая нога поднята
    X = np.vstack([left, right, patch(0.0, 3.0, 1.7, 0.05, 3)])
    reg = np.array(['foot.L'] * len(left) + ['foot.R'] * len(right) + ['head'] * 9)
    out = S.measure(FakePlate(), X, reg, 'stand', [], CFG)
    assert list(out['parts']) == ['foot.L'] and out['contact_status'] == 'contact'


def test_frame_crop_and_minimal_shift_back_into_frame():
    plate = FakePlate()
    X, reg = lying_body(z=0.0, cy=1.2)                                  # близко к камере: низ тела режется кадром
    out = S.measure(plate, X, reg, 'lie', ['pelvis'], CFG)
    assert out['in_frame_frac'] < CFG['min_in_frame_frac'] and out['frame_ok'] is False
    assert 'cropped' in out['auto_flag']
    fix = out['frame_fix_dxdy_m']
    assert fix is not None and fix[1] > 0                               # «дальше от камеры»
    shifted = X + S.location_basis(plate) @ np.array([fix[0], fix[1], 0.0])
    after = S.measure(plate, shifted, reg, 'lie', ['pelvis'], CFG)
    assert after['frame_ok'] is True and after['in_frame_frac'] >= CFG['min_in_frame_frac']
    # масштаб не менялся: сдвиг жёсткий
    assert np.allclose(np.ptp(shifted, axis=0), np.ptp(X, axis=0))


def test_head_outside_frame_fails_even_when_body_fraction_is_high():
    plate = FakePlate()
    X = np.vstack([patch(0.0, 3.0, 0.0, 0.3, 60), patch(6.0, 3.0, 1.7, 0.05, 3)])   # много точек в кадре, голова за краем
    reg = np.array(['pelvis'] * 3600 + ['head'] * 9)
    u, v, z = plate.project(X)
    out = S.measure(plate, X, reg, 'sit_edge', [], CFG)
    assert out['in_frame_frac'] > 0.99                                   # доля тела высокая, но голова вне кадра
    assert out['head_in_frame'] is False and out['frame_ok'] is False


def test_measurement_errors_are_explicit_and_penalised():
    plate = FakePlate()
    X, reg = lying_body(z=0.0)
    broken = FakePlate()
    broken.Z = np.full_like(broken.Z, np.inf)                            # карты глубины нет
    out = S.measure(broken, X, reg, 'lie', [], CFG)
    assert out['contact_status'] == 'unknown' and out['measure_error'] and out['screen_gap_m'] is None
    assert out['auto_flag'] == 'measure_error'
    factor, comp = S.score_factor(out, CFG)
    assert factor == CFG['measure_error_factor'] and comp['screen_measured'] == 1.0
    # нет нужной части тела — ошибка, а не «контакт есть»
    no_pelvis = S.measure(plate, X[reg != 'pelvis'], reg[reg != 'pelvis'], 'sit_edge', [], CFG)
    assert any('pelvis' in e for e in no_pelvis['measure_error']) and no_pelvis['contact_status'] == 'unknown'
    # нечисловая вершина
    bad = X.copy()
    bad[0, 0] = np.nan
    assert S.measure(plate, bad, reg, 'lie', [], CFG)['measure_error']
    # сломанный room_scale
    negative = S.measure(plate, X, reg, 'lie', [], CFG, room_scale=-1.0)
    assert negative['contact_status'] == 'unknown' and any('room_scale' in e for e in negative['measure_error'])
    nan_scale = S.measure(plate, X, reg, 'lie', [], CFG, room_scale=float('nan'))
    assert nan_scale['measure_error'] or nan_scale['contact_status'] == 'unknown'
    # измерения нет вообще
    assert S.score_factor(None, CFG)[0] == CFG['measure_error_factor']
    assert S.auto_flag(None) == 'measure_error'


def test_pose_without_contact_parts_measures_frame_only():
    X, reg = lying_body(z=0.0)
    out = S.measure(FakePlate(), X, reg, 'auto', [], CFG)
    assert out['contact_status'] == 'n/a' and out['measure_error'] == [] and out['frame_ok'] is True


# ------------------------------------------------------------------------------------------------ оценка и retries
def rep_with(screen):
    return {'status': 'ok', 'support_gap_cm': 1.0, 'max_penetration_cm': 1.0, 'screen': screen,
            'orient': {'total_change_deg': 5.0}, 'angle_to_camera_deg': {'ref': 10.0, 'plate': 20.0}}


def good_screen(**kw):
    return {'frame_ok': True, 'in_frame_frac': 1.0, 'contact_status': 'contact', 'measure_error': [],
            'screen_gap_m': 0.0, 'vertical_gap_m': 0.0, **kw}


def test_score_penalises_floating_cropped_and_unmeasured():
    cand = {'label': 'bed'}
    ok, comp = C.score(cand, 'ok', rep_with(good_screen()), 'lie', CAND)
    floating, _ = C.score(cand, 'ok', rep_with(good_screen(contact_status='floating', screen_gap_m=.2)), 'lie', CAND)
    cropped, _ = C.score(cand, 'ok', rep_with(good_screen(frame_ok=False, in_frame_frac=.8)), 'lie', CAND)
    unmeasured, comp_u = C.score(cand, 'ok', rep_with(good_screen(measure_error=['нет карты'])), 'lie', CAND)
    assert comp['screen_factor'] == 1.0
    assert ok > floating and ok > cropped and ok > unmeasured > 0
    assert floating == pytest.approx(ok * CAND['screen']['floating_factor'], abs=1e-3)
    assert cropped == pytest.approx(ok * CAND['screen']['frame_fail_factor'], abs=1e-3)
    assert comp_u['screen_factor'] == CAND['screen']['measure_error_factor']
    # без постановки итог остаётся 0 — фактор его не поднимает
    assert C.score(cand, 'fallback', {}, 'lie', CAND)[0] == 0.0


def test_hinted_support_beats_neighbour_object():
    hinted = {'label': 'sofa', 'hinted': True, 'hint_available': True}
    neighbour = {'label': 'sofa', 'hinted': False, 'hint_available': True}
    no_hint = {'label': 'sofa', 'hinted': False, 'hint_available': False}
    rep = rep_with(good_screen())
    a, _ = C.score(hinted, 'ok', rep, 'sit_edge', CAND)
    b, comp = C.score(neighbour, 'ok', rep, 'sit_edge', CAND)
    c, _ = C.score(no_hint, 'ok', rep, 'sit_edge', CAND)
    assert a == c and b == pytest.approx(a * CAND['screen']['hint_mismatch_factor'], abs=1e-3)
    assert comp['hint_mismatch'] == 1.0


def test_plan_marks_hinted_support():
    objs = [{'id': 'floor_1', 'type': 'floor', 'height_m': 0.0, 'area_m2': 8.0},
            {'id': 'surf_1', 'type': 'sofa', 'height_m': 0.47, 'area_m2': 0.9, 'center_m': [0, 1.2, .47]},
            {'id': 'surf_2', 'type': 'bench', 'height_m': 0.45, 'area_m2': 1.5, 'center_m': [2, 1.2, .45]}]
    labels = C.classify(objs, CAND)
    plan = C.plan('sit_edge', objs, labels, {'support': 'sofa'}, 4, CAND, 1.6)
    assert plan and all(c['hint_available'] for c in plan)
    assert [c['hinted'] for c in plan if c['label'] == 'sofa'] == [True]
    assert not any(c['hinted'] for c in plan if c['label'] != 'sofa')
    nohint = C.plan('sit_edge', objs, labels, {}, 4, CAND, 1.6)
    assert not any(c['hint_available'] for c in nohint)


def test_retries_settle_by_depth_map_also_for_floor_and_limited():
    cand = {'variant': 'support', 'label': 'floor', 'body_dz': 0.0}
    screen = good_screen(contact_status='floating', screen_gap_m=.2, vertical_gap_m=.12,
                         parts={'pelvis': {'screen_gap_m': .2, 'vertical_gap_m': .12, 'measure_error': None}})
    first = C.retries(cand, 'ok', [], CAND, 'ok', rep_with(screen), 'lie')
    assert len(first) == 1 and first[0]['variant'].endswith('+screen') and first[0]['origin'] == 'retry'
    assert first[0]['body_dz'] == pytest.approx(-(.12 - CAND['screen']['settle']['target_m']))
    twice = C.retries(first[0], 'ok', [], CAND, 'ok', rep_with(screen), 'lie')
    assert len(twice) == 1 and twice[0]['screen_settled'] == 2
    assert C.retries(twice[0], 'ok', [], CAND, 'ok', rep_with(screen), 'lie') == []          # лимит rounds
    assert C.retries(cand, 'ok', [], CAND, 'ok', rep_with(good_screen()), 'lie') == []
    assert C.retries(cand, 'ok', [], CAND, 'ok', rep_with(good_screen(measure_error=['x'])), 'lie') == []
    assert C.retries(cand, 'ok', [], CAND, 'ok', {'status': 'ok'}, 'lie') == []              # нет измерения — нет сдвига


def test_derived_adds_minimal_frame_shift_for_cropped_candidates():
    cand = {'variant': 'floor', 'label': 'floor', 'body_dx': 0.1, 'body_dy': 0.2, 'height_m': 2.3}
    cropped = {'report': {'screen': good_screen(frame_ok=False, in_frame_frac=.8, frame_fix_dxdy_m=[0.25, 0.6])}}
    inside = {'report': {'screen': good_screen()}}
    out = C.derived([cand, dict(cand, variant='yaw+25')], [cropped, inside], CAND, 2, [], None, cand)
    shifts = [c for c in out if 'screen_frame' in c['variant']]
    assert len(shifts) == 1 and shifts[0]['origin'] == 'derived'
    assert shifts[0]['body_dx'] == pytest.approx(0.35) and shifts[0]['body_dy'] == pytest.approx(0.8)
    assert shifts[0]['height_m'] == 2.3                                  # масштаб фигуры не уменьшается
    assert not [c for c in C.derived([cand], [inside], CAND, 1, [], None, cand) if 'screen_frame' in c['variant']]


def test_mixed_parts_no_vertical_settle_but_tilt_variants():
    """Таз лежит, торс висит: вертикальный сдвиг утопил бы таз — вместо него наклон (recline_pitch)."""
    X, reg = lying_body(z=0.0)
    X = X.copy()
    X[reg == 'torso', 2] += 0.2
    out = S.measure(FakePlate(), X, reg, 'lie', ['pelvis', 'torso'], CFG)
    assert out['contact_status'] == 'floating' and S.settle_dz(out, CFG) is None
    assert S.floating_parts(out, CFG) == ['torso'] and S.torso_floats_while_others_rest(out, CFG)
    cand = {'variant': 'support', 'label': 'floor', 'support_parts': 'pelvis,torso', 'body_dpitch': 0.0}
    rep = {'screen': out}
    derived = C.derived([cand], [{'report': rep}], CAND, 1, [], None, cand)
    assert [c['body_dpitch'] for c in derived if c['variant'].startswith('recline_pitch')] == CAND['recline_pitch_shifts_deg']
    assert C.retries(cand, 'ok', [], CAND, 'ok', rep, 'lie') == []
    whole = S.measure(FakePlate(), *lying_body(z=0.15), 'lie', ['pelvis', 'torso'], CFG)
    assert not S.torso_floats_while_others_rest(whole, CFG)
    assert not [c for c in C.derived([cand], [{'report': {'screen': whole}}], CAND, 1, [], None, cand)
                if c['variant'].startswith('recline_pitch')]
