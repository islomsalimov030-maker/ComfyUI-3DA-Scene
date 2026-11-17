"""E10: правка позы ручками 3D-окна — IK рук/ног, изгиб груди, LBS, разбор pose_edits, путь без ключа.
Без GPU и сервера, на синтетическом скелете MHR70 и трубках-конечностях. Запуск: python -m pytest tests/test_pose_ik.py -q
"""
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools/scene3d'), str(ROOT / 'tools/scene3d/posev2')]
import pose_ik as ik  # noqa: E402
import scene_edit  # noqa: E402

CONF = ik.cfg()


def skeleton():
    """Стоящий человек, оси постановки: x — влево тела, y — назад, z — вверх (вперёд = -y)."""
    K = np.zeros((70, 3))
    for s, x in (('l', 1.), ('r', -1.)):
        i = 0 if s == 'l' else 1
        K[9 + i] = [.1 * x, 0, .9]                      # бедро
        K[11 + i] = [.1 * x, -.08, .5]                  # колено согнуто вперёд
        K[13 + i] = [.1 * x, 0, .08]                    # лодыжка
        K[5 + i] = [.18 * x, 0, 1.42]                   # плечо
        K[7 + i] = [.22 * x, .06, 1.15]                 # локоть чуть назад
        wrist = np.array([.24 * x, 0, .9])
        block, w = (42, 62) if s == 'l' else (21, 41)
        K[w] = wrist
        for f in range(5):
            for j in range(4):
                K[block + 4 * f + j] = wrist + [.015 * (f - 2) * x, -.01, -.03 - .025 * (3 - j)]
        foot = 15 if s == 'l' else 18
        K[foot], K[foot + 1], K[foot + 2] = [.1 * x, -.15, .02], [.13 * x, -.13, .02], [.1 * x, .04, .02]
        K[63 + i], K[65 + i], K[67 + i] = K[7 + i] + [0, .02, 0], K[7 + i] - [0, .02, 0], K[5 + i] + [0, 0, .02]
    K[69] = [0, 0, 1.45]
    K[0], K[1], K[2], K[3], K[4] = [0, -.1, 1.6], [.03, -.09, 1.63], [-.03, -.09, 1.63], [.07, 0, 1.6], [-.07, 0, 1.6]
    return K


def tube(points, radius, ring=10, step=.01):
    """Трубка вдоль ломаной (одна связная поверхность): вершины, треугольники."""
    pts = [np.asarray(p, float) for p in points]
    path = [pts[0]]
    for a, b in zip(pts, pts[1:]):
        n = max(1, int(np.linalg.norm(b - a) / step))
        path += [a + (b - a) * t / n for t in range(1, n + 1)]
    path = np.array(path)
    V, F = [], []
    for k, p in enumerate(path):
        t = path[min(k + 1, len(path) - 1)] - path[max(k - 1, 0)]
        t /= np.linalg.norm(t)
        u = np.cross(t, [0, 0, 1.]) if abs(t[2]) < .9 else np.cross(t, [1., 0, 0])
        u /= np.linalg.norm(u)
        v = np.cross(t, u)
        # неровная трубка (как живой меш): без равноудалённых от оси вершин и ничьих в затравках
        V += [p + radius * (1 + .08 * math.sin(3 * a + .7 * k)) * (math.cos(a) * u + math.sin(a) * v)
              for a in np.linspace(0, 2 * math.pi, ring, endpoint=False) + .37 * k]
    for k in range(len(path) - 1):
        for j in range(ring):
            a, b = k * ring + j, k * ring + (j + 1) % ring
            F += [[a, b + ring, b], [a, a + ring, b + ring]]
    return np.array(V), np.array(F)


def body_mesh(K):
    J = ik.joints(K)
    parts = [tube([J['j0'], J['neck']], .12), tube([J['neck'], J['head']], .06)]
    for s in 'lr':
        parts.append(tube([J[f'sh_{s}'], J[f'el_{s}'], J[f'wr_{s}'], J[f'hd_{s}']], .04))
        parts.append(tube([J[f'hip_{s}'], J[f'kn_{s}'], J[f'an_{s}'], J[f'toe_{s}']], .06))
    V, F, off = [], [], 0
    for v, f in parts:
        V.append(v)
        F.append(f + off)
        off += len(v)
    return np.vstack(V), np.vstack(F)


# ---------------------------------------------------------------------------------------------- двухзвенная IK
ARM = (np.array([.18, 0, 1.42]), np.array([.22, .06, 1.15]), np.array([.24, 0, .9]))


def test_two_bone_reachable_keeps_lengths_and_hits_target():
    a, b, c = ARM
    target = c + [0, -.05, .15]                         # кисть на 15 см вверх и чуть вперёд
    b2, c2, R1, R2, info = ik.two_bone(a, b, c, target, [0, 1., 0], CONF['reach_eps'])
    assert info['reached'] and not info['straightened']
    assert np.allclose(c2, target, atol=1e-9)
    assert abs(np.linalg.norm(b2 - a) - np.linalg.norm(b - a)) < 1e-9
    assert abs(np.linalg.norm(c2 - b2) - np.linalg.norm(c - b)) < 1e-9
    # повороты костей — вращения, переводят исходные кости в новые
    for R in (R1, R2):
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-9) and abs(np.linalg.det(R) - 1) < 1e-9
    assert np.allclose(R1 @ (b - a), b2 - a, atol=1e-9)
    assert np.allclose(R2 @ (c - b), c2 - b2, atol=1e-9)


def test_two_bone_same_target_is_identity():
    a, b, c = ARM
    b2, c2, R1, R2, _ = ik.two_bone(a, b, c, c, [0, 1., 0], CONF['reach_eps'])
    assert np.allclose(b2, b, atol=1e-9) and np.allclose(c2, c, atol=1e-9)
    assert np.allclose(R1, np.eye(3), atol=1e-9) and np.allclose(R2, np.eye(3), atol=1e-9)


def test_two_bone_unreachable_straightens_toward_target():
    a, b, c = ARM
    l1, l2 = np.linalg.norm(b - a), np.linalg.norm(c - b)
    target = a + np.array([.3, -1., -.2])              # дальше длины руки
    b2, c2, _, _, info = ik.two_bone(a, b, c, target, [0, 1., 0], CONF['reach_eps'])
    assert info['straightened'] and not info['reached']
    direction = (target - a) / np.linalg.norm(target - a)
    assert np.allclose((c2 - a) / np.linalg.norm(c2 - a), direction, atol=1e-9)
    assert abs(np.linalg.norm(b2 - a) - l1) < 1e-9 and abs(np.linalg.norm(c2 - b2) - l2) < 1e-9
    assert info['bend_deg'] > 179.                      # рука прямая
    assert np.linalg.norm(c2 - a) > (l1 + l2) * (1 - 1e-5)


def test_two_bone_too_close_target_stays_folded_with_lengths():
    a, b, c = ARM
    b2, c2, _, _, info = ik.two_bone(a, b, c, a + [0, 0, -.001], [0, 1., 0], CONF['reach_eps'])
    assert not info['reached']
    assert abs(np.linalg.norm(b2 - a) - np.linalg.norm(b - a)) < 1e-9
    assert abs(np.linalg.norm(c2 - b2) - np.linalg.norm(c - b)) < 1e-9


@pytest.mark.parametrize('shift', [[0, 0, .15], [0, -.2, .1], [.1, .2, .2], [-.15, 0, .3], [0, .25, 0]])
def test_two_bone_bend_plane_and_no_reverse_bend(shift):
    a, b, c = ARM
    u = (c - a) / np.linalg.norm(c - a)
    pole = (b - a) - ((b - a) @ u) * u                   # исходное направление локтя
    target = c + np.array(shift)
    b2, c2, _, _, _ = ik.two_bone(a, b, c, target, [0, 1., 0], CONF['reach_eps'])
    d = (c2 - a) / np.linalg.norm(c2 - a)
    side = pole - (pole @ d) * d
    normal = np.cross(d, side / np.linalg.norm(side))
    assert abs((b2 - a) @ normal) < 1e-9               # локоть в плоскости «цель + исходный полюс»
    offset = (b2 - a) - ((b2 - a) @ d) * d
    assert offset @ pole > 0                             # сустав не выгнут в обратную сторону


def test_two_bone_target_in_original_plane_keeps_plane():
    a, b, c = ARM
    n = np.cross(b - a, c - a)
    n /= np.linalg.norm(n)
    target = c + .1 * (b - a) / np.linalg.norm(b - a) + .05 * np.cross(n, (c - a) / np.linalg.norm(c - a))
    b2, c2, _, _, _ = ik.two_bone(a, b, c, target, [0, 1., 0], CONF['reach_eps'])
    assert abs((b2 - a) @ n) < 1e-9 and abs((c2 - a) @ n) < 1e-9


def test_two_bone_straight_limb_uses_pole_hint():
    a, b, c = np.array([0, 0, 1.]), np.array([0, 0, .6]), np.array([0, 0, .2])
    b2, c2, _, _, _ = ik.two_bone(a, b, c, np.array([0, 0, .4]), np.array([0, -1., 0]), CONF['reach_eps'])
    assert np.allclose(c2, [0, 0, .4]) and b2[1] < -.1  # колено вперёд (подсказка полюса)


# ---------------------------------------------------------------------------------------------- грудь
def test_chest_bend_distributed_pelvis_fixed():
    K = skeleton()
    J = ik.joints(K)
    target = J['chest'] + [0, -.15, 0]                  # грудь на 15 см вперёд
    T, info = ik.spine_bend(J, target, CONF)
    assert np.allclose(ik.apply(T['spine1'], J['j0']), J['j0'])               # таз на месте
    p = [J['j0']] + [ik.apply(T[b], J[n]) for b, n in (('spine1', 'j1'), ('spine2', 'j2'), ('chest', 'neck'))]
    old = [J['j0'], J['j1'], J['j2'], J['neck']]
    for k in range(3):                                                         # длины позвонков сохранены
        assert abs(np.linalg.norm(p[k + 1] - p[k]) - np.linalg.norm(old[k + 1] - old[k])) < 1e-9
    angles = []
    for k, (b, prev) in enumerate((('spine1', None), ('spine2', 'spine1'), ('chest', 'spine2'))):
        R = T[b][0] if prev is None else T[b][0] @ T[prev][0].T
        angles.append(math.degrees(math.acos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
    assert max(angles) - min(angles) < 1e-6 and angles[0] > 1.                # поровну на 3 позвонка
    chest = ik.apply(T['chest'], J['chest'])
    v, w = chest - J['j0'], target - J['j0']
    assert math.degrees(math.acos(v @ w / np.linalg.norm(v) / np.linalg.norm(w))) < .05   # направление на цель
    assert chest[1] < J['chest'][1] - .1 and not info['limited']


def test_chest_bend_limited():
    K = skeleton()
    J = ik.joints(K)
    T, info = ik.spine_bend(J, J['j0'] + [0, -.5, -.1], CONF)                  # цель почти под углом 100°
    assert info['limited'] and abs(info['bend_deg'] - CONF['spine_max_deg']) < 1e-6


# ---------------------------------------------------------------------------------------------- pose_edits
def test_parse_pose_edits():
    assert ik.parse(None) == {}
    out = ik.parse({'hand_r': [0, 0, .15], 'foot_l': [0, -.1, 0], 'chest': [0, 0, 0]})
    assert set(out) == {'hand_r', 'foot_l'} and np.allclose(out['hand_r'], [0, 0, .15])
    for bad in ({'knee_l': [0, 0, .1]}, {'hand_r': [0, .1]}, {'hand_r': [0, float('nan'), 0]},
                {'hand_r': 'up'}, [1, 2, 3], {'hand_r': [0, 0, 5.]}):
        with pytest.raises(ValueError):
            ik.parse(bad)


def test_edits_from_body_transform_without_key_is_empty():
    old = json.dumps(dict(placement_key='k', matrix=np.eye(4).tolist()))
    assert ik.edits_from(old) == {}
    new = json.dumps(dict(placement_key='k', matrix=np.eye(4).tolist(), pose_edits={'hand_r': [0, 0, .15]}))
    assert set(ik.edits_from(new)) == {'hand_r'}


def manifest():
    return dict(placement_key='key1', entities=[dict(id='body', kind='body', matrix_world=np.eye(4).tolist())])


def model(**extra):
    return [dict(entity_id='body', position=dict(x=.1, y=0, z=0), quaternion=dict(x=0, y=0, z=0, w=1),
                 scale=dict(x=1, y=1, z=1), **extra)]


def test_scene_edits_without_pose_edits_unchanged():
    body, props = scene_edit.edits(manifest(), model())
    data = json.loads(body)
    assert set(data) == {'placement_key', 'matrix'} and props == '[]'
    expected = np.eye(4)
    expected[0, 3] = .1
    assert body == json.dumps(dict(placement_key='key1', matrix=expected.tolist()))   # строка как до E10
    body0, _ = scene_edit.edits(manifest(), model(pose_edits={'hand_r': [0, 0, 0]}))
    assert body0 == body                                                       # нулевые правки — тоже без ключа


def test_scene_edits_pose_edits_three_to_world_axes():
    body, _ = scene_edit.edits(manifest(), model(pose_edits={'hand_r': [0, .15, 0], 'foot_l': [0, 0, -.1]}))
    data = json.loads(body)
    assert np.allclose(data['pose_edits']['hand_r'], [0, 0, .15])             # three Y вверх -> мир Z вверх
    assert np.allclose(data['pose_edits']['foot_l'], [0, .1, 0])              # three -Z -> мир +Y
    with pytest.raises(ValueError):
        scene_edit.edits(manifest(), model(pose_edits={'nose': [0, .1, 0]}))


# ---------------------------------------------------------------------------------------------- LBS и тело целиком
@pytest.fixture(scope='module')
def body():
    K = skeleton()
    X, F = body_mesh(K)
    idx, w = ik.skin_weights(X, F, K, CONF)
    return K, X, F, idx, w


def test_weights_normalized_and_identity(body):
    K, X, F, idx, w = body
    assert np.allclose(w.sum(1), 1) and (w >= 0).all()
    T = {b: ik.IDENTITY for b in ik.BONES}
    assert np.allclose(ik.deform(X, idx, w, T), X)
    main = np.array(ik.BONES)[idx[:, 0]]
    J = ik.joints(K)
    near_wrist = np.linalg.norm(X - (J['el_r'] + J['wr_r']) / 2, axis=1) < .045
    assert (main[near_wrist] == 'forearm_r').all()                             # середина предплечья — предплечье


def test_no_edits_returns_same_body(body):
    K, X, F, _, _ = body
    X2, K2, info = ik.apply_edits(X, K, F, {})
    assert np.array_equal(X2, X) and np.array_equal(K2, K) and info == {}


def test_hand_and_foot_edit_moves_only_their_limbs(body):
    K, X, F, _, _ = body
    J = ik.joints(K)
    edits = {'hand_r': np.array([0, 0, .15]), 'foot_l': np.array([0, -.1, 0])}
    X2, K2, info = ik.apply_edits(X, K, F, edits)
    assert np.allclose(K2[41], K[41] + [0, 0, .15], atol=1e-9)                 # правое запястье у цели
    assert np.allclose(K2[13], K[13] + [0, -.1, 0], atol=1e-9)                 # левая лодыжка у цели
    for i in (0, 5, 6, 9, 10, 62, 7, 14, 12, 69):                              # остальное тело на месте
        assert np.allclose(K2[i], K[i]), i
    before, after = ik.bone_lengths(K), ik.bone_lengths(K2)
    assert max(abs(after[k] - before[k]) for k in before) < 1e-9
    moved = np.linalg.norm(X2 - X, axis=1) > 1e-6
    far = np.minimum(np.linalg.norm(X - J['wr_r'], axis=1), np.linalg.norm(X - J['an_l'], axis=1))
    away = np.linalg.norm(X - J['sh_r'], axis=1) > .45
    away &= np.linalg.norm(X - J['hip_l'], axis=1) > .6
    assert not moved[away & (far > .7)].any()                                  # голова, другая рука, другая нога
    # меш не рвётся: рёбра растягиваются не больше чем вдвое
    e = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    ratio = np.linalg.norm(X2[e[:, 0]] - X2[e[:, 1]], axis=1) / np.linalg.norm(X[e[:, 0]] - X[e[:, 1]], axis=1)
    assert ratio.max() < 2. and ratio.min() > .3
    assert info['hand_r']['reached'] and info['foot_l']['reached']


def test_chest_edit_moves_upper_body_pelvis_fixed(body):
    K, X, F, _, _ = body
    X2, K2, info = ik.apply_edits(X, K, F, {'chest': np.array([0, -.12, 0])})
    for i in (9, 10, 11, 12, 13, 14):
        assert np.allclose(K2[i], K[i])
    assert K2[69][1] < K[69][1] - .08 and K2[5][1] < K[5][1] - .05 and K2[6][1] < K[6][1] - .05  # шея, плечи вперёд
    J = ik.joints(K)
    legs = X[:, 2] < J['j0'][2] - .15
    assert np.allclose(X2[legs], X[legs])


def test_edit_base_equivariant_to_body_move(body):
    """Правки в мире после сдвига/поворота тела = те же правки в осях тела до сдвига."""
    K, X, F, _, _ = body
    R = ik.rot_axis(np.array([0, 0, 1.]), math.radians(40))
    t = np.array([.5, -.2, .1])
    transform = lambda A, KP: (A @ R.T + t, KP @ R.T + t)   # noqa: E731
    edits = {'hand_r': np.array([0, 0, .15])}
    base = dict(X=X, KP=K, groups=[], reg=None, report={})
    edited, report = ik.edit_base(base, F, edits, transform)
    Xw, Kw = transform(edited['X'], edited['KP'])
    Xd, Kd, _ = ik.apply_edits(*transform(X, K), F, edits)
    assert np.allclose(Kw, Kd, atol=1e-9) and np.allclose(Xw, Xd, atol=1e-6)
    assert report['max_bone_length_change_m'] < 1e-9 and report['handles']['hand_r'] == [0, 0, .15]


# ---------------------------------------------------------------------------------------------- фронтенд = сервер
NODE = shutil.which('node')


@pytest.mark.skipif(NODE is None, reason='node не установлен')
def test_frontend_matches_backend(tmp_path, body):
    K, X, F, idx, w = body
    edits = {'hand_r': [0, 0, .15], 'foot_l': [0, -.1, 0], 'chest': [0, -.05, .02]}
    script = tmp_path / 'run.mjs'
    data = tmp_path / 'data.json'
    data.write_text(json.dumps(dict(K=K.tolist(), X=X.ravel().tolist(), F=F.ravel().tolist(), edits=edits, conf=CONF)))
    script.write_text(
        "import fs from 'fs';\n"
        f"import * as rig from {json.dumps((ROOT / 'web/poseRig.js').as_uri())};\n"
        f"const d = JSON.parse(fs.readFileSync({json.dumps(str(data))}, 'utf8'));\n"
        "const r = rig.buildWeights(Float32Array.from(d.X), Uint32Array.from(d.F), d.K, d.conf);\n"
        "const s = rig.solve(d.K, d.edits, d.conf);\n"
        "const out = rig.deform(r, s.T, new Float64Array(d.X.length));\n"
        "const H = rig.posedHandles(d.K, s.T);\n"
        "process.stdout.write(JSON.stringify({X: Array.from(out), H, iters: r.iters}));\n", encoding='utf-8')
    res = json.loads(subprocess.run([NODE, str(script)], capture_output=True, text=True, check=True).stdout)
    X2, K2, _ = ik.apply_edits(X, K, F, {k: np.array(v) for k, v in edits.items()})
    Xjs = np.array(res['X']).reshape(-1, 3)
    assert np.abs(Xjs - X2).max() < 2e-3                    # float32 во фронтенде, те же веса и кости
    H = ik.handle_points(K2)
    for h in ik.HANDLES:
        assert np.allclose(res['H'][h], H[h], atol=1e-5), h


@pytest.mark.skipif(NODE is None, reason='node не установлен')
def test_pose_handles_mesh_transforms_cpu(tmp_path):
    """Реальный poseHandles.js: матрицы частей, деформация, общая геометрия и сброс."""
    K = skeleton()
    X, F = body_mesh(K)
    data = tmp_path / 'pose_handles.json'
    data.write_text(json.dumps(dict(K=K.tolist(), X=X.ravel().tolist(), F=F.ravel().tolist(), conf=CONF)), encoding='utf-8')
    subprocess.run([NODE, str(ROOT / 'tests/pose_handles_cpu.mjs'), str(data)], check=True, timeout=120)


# ---------------------------------------------------------------------------------------------- круг 2: обратный сгиб
FWD = np.array([1., 0, 0])                               # «вперёд» в случаях из ревью
KNEE_CASES = {
    # почти прямая нога, колено на 2 мм назад (шум MHR) — раньше колено уходило назад на 0.206 м
    'leg_hyper_2mm': (np.array([0, 0, 1.0]), np.array([-.002, 0, .55]), np.array([0, 0, .1]), np.array([0, 0, .2])),
    # заметное переразгибание (угол < straight_deg), колено против подсказки — тоже вперёд (hint_side)
    'leg_hyper_8cm': (np.array([0, 0, 1.0]), np.array([-.08, 0, .55]), np.array([0, 0, .1]), np.array([0, 0, .25])),
}
ELBOW_CASES = {
    # почти прямая рука, локоть на 3 мм вперёд — раньше локоть уходил вперёд на 0.198 м
    'arm_hyper_3mm': (np.array([0, 0, 1.4]), np.array([.003, 0, 1.1]), np.array([0, 0, .8]), np.array([0, 0, .95])),
}


def elbow_hint(fwd=FWD, up=np.array([0, 0, 1.])):
    return -fwd - .5 * up                                 # = pose_ik.pole_hint для локтя


@pytest.mark.parametrize('name', sorted(KNEE_CASES))
def test_knee_never_bends_backward(name):
    a, b, c, target = KNEE_CASES[name]
    b2, c2, R1, R2, info = ik.two_bone(a, b, c, target, FWD, CONF['reach_eps'], CONF['straight_deg'], hint_side=True)
    assert info['pole'] == 'hint' and info['reached']
    assert b2[0] > .1                                     # колено вперёд
    assert np.allclose(c2, target, atol=1e-9)
    assert abs(np.linalg.norm(b2 - a) - np.linalg.norm(b - a)) < 1e-9
    assert abs(np.linalg.norm(c2 - b2) - np.linalg.norm(c - b)) < 1e-9
    for R in (R1, R2):                                    # при смене полюса повороты остаются поворотами
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-9) and abs(np.linalg.det(R) - 1) < 1e-9
    assert np.allclose(R1 @ (b - a), b2 - a, atol=1e-9) and np.allclose(R2 @ (c - b), c2 - b2, atol=1e-9)


@pytest.mark.parametrize('name', sorted(ELBOW_CASES))
def test_elbow_never_bends_forward_when_straight(name):
    a, b, c, target = ELBOW_CASES[name]
    b2, c2, _, _, info = ik.two_bone(a, b, c, target, elbow_hint(), CONF['reach_eps'], CONF['straight_deg'])
    assert info['pole'] == 'hint' and info['reached']
    assert b2[0] < -.1                                    # локоть назад
    assert np.allclose(c2, target, atol=1e-9)


def test_bent_elbow_keeps_its_own_pole():
    """Рука вперёд, предплечье вверх: локоть смотрит вниз-вперёд, против подсказки «назад-вниз» в осях корпуса.
    Это нормальный сгиб — полюс исходный, иначе рука выгнулась бы назад (поэтому hint_side только у колена)."""
    a, b, c = np.array([0, 0, 1.4]), np.array([.3, 0, 1.4]), np.array([.3, 0, 1.68])
    b2, c2, _, _, info = ik.two_bone(a, b, c, c + [0, 0, .05], elbow_hint(), CONF['reach_eps'], CONF['straight_deg'])
    assert info['pole'] == 'source' and info['bend_src_deg'] < 100
    u = (c2 - a) / np.linalg.norm(c2 - a)
    off = (b2 - a) - ((b2 - a) @ u) * u
    src = (b - a) - ((b - a) @ ((c - a) / np.linalg.norm(c - a))) * ((c - a) / np.linalg.norm(c - a))
    assert off @ src > 0                                  # сгиб в прежнюю сторону


def test_solve_hyperextended_leg_knee_toward_toes():
    """Целиком через solve: колено переразогнуто на 1 см назад (вперёд = к носку, -y) — правка стопы гнёт колено к носку."""
    K = skeleton()
    K[11] = [.1, .01, .5]                                 # левое колено чуть назад (+y)
    T, info = ik.solve(K, {'foot_l': np.array([0, -.05, .15])}, CONF)
    knee = ik.apply(T['shin_l'], K[11])
    assert info['foot_l']['pole'] == 'hint' and knee[1] < -.05


@pytest.mark.skipif(NODE is None, reason='node не установлен')
def test_frontend_pole_matches_backend(tmp_path):
    cases = {**{k: (v, FWD, True) for k, v in KNEE_CASES.items()},
             **{k: (v, elbow_hint(), False) for k, v in ELBOW_CASES.items()}}
    data = {k: dict(abc=[x.tolist() for x in v], hint=h.tolist(), knee=kn) for k, (v, h, kn) in cases.items()}
    script = tmp_path / 'pole.mjs'
    script.write_text(
        f"import * as rig from {json.dumps((ROOT / 'web/poseRig.js').as_uri())};\n"
        f"const d = {json.dumps(data)}, conf = {json.dumps(CONF)};\n"
        "const out = {};\n"
        "for (const [k, v] of Object.entries(d)) { const [a, b, c, t] = v.abc;\n"
        "  const r = rig.twoBone(a, b, c, t, v.hint, conf.reach_eps, conf.straight_deg, v.knee);\n"
        "  out[k] = {b2: r.b2, c2: r.c2, pole: r.info.pole}; }\n"
        "process.stdout.write(JSON.stringify(out));\n", encoding='utf-8')
    res = json.loads(subprocess.run([NODE, str(script)], capture_output=True, text=True, check=True).stdout)
    for k, (v, h, kn) in cases.items():
        b2, c2, _, _, info = ik.two_bone(*v, h, CONF['reach_eps'], CONF['straight_deg'], hint_side=kn)
        assert res[k]['pole'] == info['pole'] and np.allclose(res[k]['b2'], b2, atol=1e-9), k
        assert np.allclose(res[k]['c2'], c2, atol=1e-9), k
