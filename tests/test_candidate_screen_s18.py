"""E1c: проверка «в кадре» на реальных данных S18 (лёжа на мате): карта глубины, камера и результат постановки,
полученные прогоном 3DA (локально на ПК, тот же код и те же узлы, что на сервере). Данные лежат в data/e1c_s18 проекта
AIO_auto_photo, не в репозитории: нет каталога — тесты пропускаются. Без GPU. Запуск: python -m pytest tests/test_candidate_screen_s18.py -q"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parents[1] / 'data' / 'e1c_s18'
pytestmark = pytest.mark.skipif(not (DATA / 'plate' / 'depth_loc.png').is_file(), reason='нет data/e1c_s18 проекта')
sys.path[:0] = [str(ROOT / 'tools/scene3d'), str(ROOT / 'tools/scene3d/posev2')]

import candidate_screen as S  # noqa: E402

CFG = S.load_config()
PARTS = ['pelvis', 'torso']


@pytest.fixture(scope='module')
def plate():
    import pose_fit
    return pose_fit.Plate(DATA / 'plate')


def load(name):
    import body
    X = S.read_obj_vertices(DATA / name / 'fit' / 'body.obj')
    kp = np.load(DATA / name / 'fit' / 'label_joints.npz')['kp']
    return X, body.regions(X, kp)


def test_floating_torso_found_where_passport_geometry_says_contact(plate):
    X, reg = load('floating')
    out = S.measure(plate, X, reg, 'recline', PARTS, CFG)
    assert out['measure_error'] == [] and out['frame_ok'] is True and out['in_frame_frac'] >= 0.97
    assert out['contact_status'] == 'floating' and out['auto_flag'] == 'floating'
    assert out['parts']['pelvis']['screen_gap_m'] < CFG['floating_gap_m']          # таз лежит
    assert out['parts']['torso']['screen_gap_m'] > CFG['floating_gap_m']           # торс висит (~0,19 м по лучу, ~0,10 м по высоте)
    assert out['parts']['torso']['vertical_gap_m'] == pytest.approx(0.096, abs=0.02)
    assert S.floating_parts(out, CFG) == ['torso'] and S.torso_floats_while_others_rest(out, CFG)
    assert S.settle_dz(out, CFG) is None                                           # вертикальный сдвиг утопил бы таз


def test_tilted_candidate_rests_on_the_mat(plate):
    X, reg = load('contact')
    out = S.measure(plate, X, reg, 'recline', PARTS, CFG)
    assert out['contact_status'] == 'contact' and out['auto_flag'] == 'ok' and out['measure_error'] == []
    assert abs(out['parts']['pelvis']['screen_gap_m']) < CFG['floating_gap_m']


def test_whole_body_hover_is_settled_down_by_depth_map(plate):
    X, reg = load('contact')
    hover = X + np.array([0.0, 0.0, 0.2])
    out = S.measure(plate, hover, reg, 'recline', PARTS, CFG)
    assert out['contact_status'] == 'floating' and len(S.floating_parts(out, CFG)) == 2
    dz = S.settle_dz(out, CFG)
    assert dz is not None and -CFG['settle']['max_down_m'] <= dz < 0
    again = S.measure(plate, hover + np.array([0.0, 0.0, dz]), reg, 'recline', PARTS, CFG)
    assert again['contact_status'] == 'contact'


def test_body_pushed_into_the_floor_is_penetrating(plate):
    X, reg = load('contact')
    out = S.measure(plate, X - np.array([0.0, 0.0, 0.2]), reg, 'recline', PARTS, CFG)
    assert out['contact_status'] == 'penetrating' and S.settle_dz(out, CFG) > 0


def test_body_cropped_by_the_camera_is_found_and_moved_back_into_frame(plate):
    X, reg = load('contact')
    basis = S.location_basis(plate)
    near = X - basis[:, 1] * 2.2                                                   # к камере: низ тела режется кадром
    out = S.measure(plate, near, reg, 'recline', PARTS, CFG)
    assert out['frame_ok'] is False and out['in_frame_frac'] < CFG['min_in_frame_frac'] and 'cropped' in out['auto_flag']
    fix = out['frame_fix_dxdy_m']
    assert fix is not None and fix[1] > 0
    shifted = near + basis @ np.array([fix[0], fix[1], 0.0])
    after = S.measure(plate, shifted, reg, 'recline', PARTS, CFG)
    assert after['frame_ok'] is True
    assert np.allclose(np.ptp(shifted, axis=0), np.ptp(near, axis=0))              # масштаб фигуры тот же
