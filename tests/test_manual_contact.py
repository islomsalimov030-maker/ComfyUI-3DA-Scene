"""E14: ручной сдвиг манекена в 3D-окне — рендер на той же высоте, что окно (перенос контактной посадки).
CPU, без Blender: Blender-часть (pose_mannequin_to_mhr) только подставляет вектор, решение — в manual_contact."""
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


def plate():
    # камера смотрит вдоль +Y мира, Z вверх: столбцы R — оси камеры в мире (x вправо, y вниз, z вглубь)
    return SimpleNamespace(R=np.array([[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]]).T)


def rot(axis, deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return {'x': np.array([[1, 0, 0], [0, c, -s], [0, s, c]]), 'z': np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[axis]


def delta_matrix(R, t, pivot, scale=1.):
    m = np.eye(4)
    m[:3, :3] = scale * R
    m[:3, 3] = pivot + t - scale * R @ pivot
    return m


KP = np.zeros((70, 3))
KP[9], KP[10] = [0.1, 0.5, 1.0], [-0.1, 0.5, 1.0]
PIVOT = (KP[9] + KP[10]) / 2
BASE_DZ = -0.2317                                    # контактный сдвиг манекена базовой постановки (S04, м)


def world_transform(matrix):
    return lambda X, K: manual_pose.transform(X, K, plate(), {'body_matrix': matrix.tolist()})[0]


def test_carry_vector_follows_gizmo_rotation(tmp_path):
    mc.save_base_contact(tmp_path, {'contact_dz_exact_m': BASE_DZ})
    m = delta_matrix(rot('x', 8), np.array([0.3, -0.1, 0.05]), PIVOT)
    vec = mc.carry_vector(tmp_path, world_transform(m), KP)
    assert np.allclose(vec, m[:3, :3] @ [0, 0, BASE_DZ], atol=1e-12)
    # при наклоне 8° вектор отклоняется от вертикали: боковая составляющая |dz|·sin 8° ≈ 3,2 см
    assert abs(abs(vec[1]) - abs(BASE_DZ) * math.sin(math.radians(8))) < 1e-9


def test_carry_vector_translation_and_yaw_stay_vertical(tmp_path):
    mc.save_base_contact(tmp_path, {'contact_dz_exact_m': BASE_DZ})
    for m in (delta_matrix(np.eye(3), np.array([0.3, 0.2, 0.0]), PIVOT), delta_matrix(rot('z', 30), np.zeros(3), PIVOT)):
        assert np.allclose(mc.carry_vector(tmp_path, world_transform(m), KP), [0, 0, BASE_DZ], atol=1e-12)


def test_carry_vector_widget_values_path(tmp_path):
    """Сдвиг из полей body_dx… (не гизмо): тот же перенос."""
    mc.save_base_contact(tmp_path, {'contact_dz_exact_m': BASE_DZ})
    values = {'body_dx': 0.2, 'body_dy': -0.1, 'body_dz': 0.0, 'body_dyaw': 25., 'body_dpitch': 0., 'body_droll': 0.}
    vec = mc.carry_vector(tmp_path, lambda X, K: manual_pose.transform(X, K, plate(), values)[0], KP)
    assert np.allclose(vec, [0, 0, BASE_DZ], atol=1e-12)


def test_no_saved_contact_means_legacy(tmp_path):
    assert mc.carry_vector(tmp_path, world_transform(np.eye(4)), KP) is None
    assert not mc.save_base_contact(tmp_path, {'warning': 'x'}) and not (tmp_path / mc.NAME).exists()
    assert not mc.save_base_contact(tmp_path, None)


def test_contact_translation_choice():
    called = []
    vec, dz, source = mc.contact_translation((0.0, 0.03, -0.2), lambda: called.append(1) or 9.)
    assert (vec, dz, source) == ((0.0, 0.03, -0.2), -0.2, 'base') and not called
    vec, dz, source = mc.contact_translation(None, lambda: -0.15)
    assert (vec, dz, source) == ((0., 0., -0.15), -0.15, 'scene')
    assert mc.contact_translation(None, lambda: 0.)[0] is None            # нет данных — без сдвига, как раньше


def test_window_equals_render_height_after_move(tmp_path):
    """Модель сцены: поза манекена эквивариантна жёсткому преобразованию (измерено на сервере: после гизмо
    R − окно = только разница контактных сдвигов). Раньше сдвинутый рендер брал новый контактный сдвиг
    (S04: dy +30 см → +0,7 см; S09: dx +30 см → +4 см), теперь — перенесённый базовый: Δ = 0."""
    rng = np.random.default_rng(1)
    pose0 = rng.normal(size=(500, 3)) * [0.2, 0.15, 0.5] + [0, 0.5, 1.0]
    window0 = pose0 + [0, 0, BASE_DZ]                                    # то, что показывает окно до сдвига
    for m, new_dz in ((delta_matrix(np.eye(3), np.array([0., 0.3, 0.]), PIVOT), BASE_DZ + 0.0074),
                      (delta_matrix(rot('z', 30), np.array([0.3, 0., 0.]), PIVOT), BASE_DZ + 0.0407),
                      (delta_matrix(rot('x', 8), np.zeros(3), PIVOT), BASE_DZ + 0.0014)):
        window = window0 @ m[:3, :3].T + m[:3, 3]                       # окно после гизмо
        posed = pose0 @ m[:3, :3].T + m[:3, 3]                          # манекен, посаженный на сдвинутый MHR, без контакта
        legacy = posed + [0, 0, new_dz]
        mc.save_base_contact(tmp_path, {'contact_dz_exact_m': BASE_DZ})
        carry = mc.carry_vector(tmp_path, world_transform(m), KP)
        fixed = posed + carry
        assert np.abs(fixed - window).max() < 1e-12
        assert np.allclose(legacy - window, np.array([0, 0, new_dz]) - m[:3, :3] @ [0, 0, BASE_DZ], atol=1e-12)
        assert np.abs(legacy - window).max() > 1e-3
    assert json.loads((tmp_path / mc.NAME).read_text())['contact_dz_m'] == BASE_DZ
