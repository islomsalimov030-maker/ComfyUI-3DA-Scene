"""E15: манекен в рендере не опускается относительно 3D-окна — остатки после E14: правки ручек позы, масштаб гизмо, «нет данных» у базы.
CPU, без Blender: решение — в manual_contact (вектор сдвига и масштаб), Blender-часть только подставляет их."""
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools/scene3d'), str(ROOT / 'tools/scene3d/posev2')]
import manual_contact as mc  # noqa: E402
import manual_pose  # noqa: E402

KP = np.zeros((70, 3))
KP[9], KP[10] = [0.1, 0.5, 1.0], [-0.1, 0.5, 1.0]
PIVOT = (KP[9] + KP[10]) / 2
BASE = {'contact_dz_exact_m': -0.0882, 'scale': 1.15}              # S11: посадка −8,8 см, масштаб манекена у предела 1,15


def plate():
    return SimpleNamespace(R=np.array([[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]]).T)


def gizmo(scale=1., yaw=0., t=(0., 0., 0.)):
    c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    m = np.eye(4)
    m[:3, :3] = scale * np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    m[:3, 3] = PIVOT + np.array(t) - m[:3, :3] @ PIVOT
    return lambda X, K: manual_pose.transform(X, K, plate(), {'body_matrix': m.tolist()})[0]


def test_base_scale_is_saved(tmp_path):
    assert mc.save_base_contact(tmp_path, BASE)
    assert json.loads((tmp_path / mc.NAME).read_text())['scale'] == 1.15


def test_carry_scale_is_base_times_gizmo_without_limit(tmp_path):
    mc.save_base_contact(tmp_path, BASE)
    # окно масштабирует тот же манекен: 1,15 × 1,1 = 1,265 — выше предела 1,15 старого пересчёта
    assert math.isclose(mc.carry_scale(tmp_path, gizmo(1.1), KP), 1.15 * 1.1, rel_tol=1e-12)
    assert math.isclose(mc.carry_scale(tmp_path, gizmo(0.8, yaw=30, t=(0.3, 0., 0.)), KP), 1.15 * 0.8, rel_tol=1e-12)
    assert math.isclose(mc.carry_scale(tmp_path, gizmo(), KP), 1.15, rel_tol=1e-12)    # без масштаба — базовый, а не новый пересчёт


def test_carry_scale_absent_for_old_placement(tmp_path):
    assert mc.carry_scale(tmp_path, gizmo(1.1), KP) is None                              # нет файла
    mc.save_base_contact(tmp_path, {'contact_dz_exact_m': -0.2})                         # файл E14 без масштаба
    assert mc.carry_scale(tmp_path, gizmo(1.1), KP) is None


def test_carry_vector_scales_with_gizmo(tmp_path):
    mc.save_base_contact(tmp_path, BASE)
    assert np.allclose(mc.carry_vector(tmp_path, gizmo(1.3), KP), [0, 0, -0.0882 * 1.3], atol=1e-12)


def test_render_args_for_moved_and_edited_body(tmp_path):
    """Те же ключи для сдвига, масштаба и правок ручек: правки ручек — тот же жёсткий сдвиг базовой постановки."""
    mc.save_base_contact(tmp_path, BASE)
    args, info = mc.render_args(tmp_path, gizmo(1.1, yaw=20, t=(0.3, 0., 0.)), KP)
    assert args[0].startswith('--contact-carry=') and args[1].startswith('--body-scale=')
    vec = [float(x) for x in args[0].split('=')[1].split(',')]
    assert np.allclose(vec, [0, 0, -0.0882 * 1.1], atol=1e-12) and math.isclose(float(args[1].split('=')[1]), 1.265, rel_tol=1e-12)
    assert info == {'contact_carry_m': [0.0, 0.0, round(-0.0882 * 1.1, 5)], 'body_scale': 1.265}


def test_render_args_legacy_without_saved_contact(tmp_path):
    assert mc.render_args(tmp_path, gizmo(1.1), KP) == ([], {})
    mc.save_base_contact(tmp_path, {'contact_dz_exact_m': -0.2})
    args, info = mc.render_args(tmp_path, gizmo(), KP)
    assert len(args) == 1 and 'body_scale' not in info


def test_no_data_base_contact_is_zero_not_recomputed(tmp_path):
    """Под телом базовой постановки нет комнаты (нет данных) — посадка 0 запоминается; после сдвига рендер её не пересчитывает
    (раньше луч вниз находил другую поверхность — манекен проваливался)."""
    assert mc.save_base_contact(tmp_path, {'contact_dz_exact_m': 0., 'contact_points': 0, 'scale': 1.0})
    args, info = mc.render_args(tmp_path, gizmo(1., yaw=15, t=(0.5, 0., 0.)), KP)
    assert args[0] == '--contact-carry=0.0,0.0,0.0'
    vec, dz, source = mc.contact_translation((0., 0., 0.), lambda: 9.)
    assert (vec, dz, source) == ((0., 0., 0.), 0., 'base')


def test_window_equals_render_with_pose_edits_and_scale(tmp_path):
    """Модель: окно = (поза с правками ручек + базовая посадка) × гизмо; рендер = (поза, посаженная на MHR) × гизмо + перенесённая посадка
    и масштаб. Раньше при правках ручек и масштабе брался новый пересчёт: S11 грудь −3,4 см, масштаб 1,1 → до 6,8 см."""
    rng = np.random.default_rng(3)
    base_pose = rng.normal(size=(400, 3)) * [0.2, 0.15, 0.5] + [0, 0.5, 1.0]
    edited = base_pose + np.where(base_pose[:, 2:3] > 1.0, [0.0, -0.05, 0.03], 0.0)     # ручки: верх тела сдвинут
    mc.save_base_contact(tmp_path, BASE)
    dz0 = BASE['contact_dz_exact_m']
    for g, new_dz in ((1.0, -0.1217), (1.1, -0.0808), (0.8, -0.055)):
        m_tr = gizmo(g, yaw=20, t=(0.3, 0., 0.))
        R = np.array([m_tr(np.array([[0., 0, 0], [1., 0, 0]]), KP)[1] - m_tr(np.array([[0., 0, 0], [1., 0, 0]]), KP)[0],
                      m_tr(np.array([[0., 0, 0], [0., 1, 0]]), KP)[1] - m_tr(np.array([[0., 0, 0], [0., 1, 0]]), KP)[0],
                      m_tr(np.array([[0., 0, 0], [0., 0, 1]]), KP)[1] - m_tr(np.array([[0., 0, 0], [0., 0, 1]]), KP)[0]]).T
        move = lambda P: m_tr(P, KP)                                                      # noqa: E731
        window = move(edited + [0, 0, dz0])
        base_render = move(edited)
        args, _ = mc.render_args(tmp_path, m_tr, KP)
        carry = np.array([float(x) for x in args[0].split('=')[1].split(',')])
        assert np.abs(base_render + carry - window).max() < 1e-12
        legacy = base_render + [0, 0, new_dz]
        assert np.abs(legacy - window).max() > 5e-4
        assert np.allclose(R, g * np.array([[math.cos(math.radians(20)), -math.sin(math.radians(20)), 0],
                                            [math.sin(math.radians(20)), math.cos(math.radians(20)), 0], [0, 0, 1]]), atol=1e-12)


def test_pack_cli_carries_contact_with_pose_edits():
    """В pack_cli ключи переноса не зависят от правок ручек: условие — только «тело двигали» и «манекен MPFB»."""
    src = (ROOT / 'tools/scene3d/pack_cli.py').read_text(encoding='utf-8')
    assert 'if moved and body_model == "mpfb":' in src and 'manual_contact.render_args(' in src
    assert 'not pose_edits and body_model' not in src
