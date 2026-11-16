"""E10: правка позы манекена ручками 3D-окна — IK рук и ног, изгиб корпуса, деформация меша (LBS).

pose_edits = {ручка: [dx, dy, dz]} — сдвиг ручки в метрах в мировых осях постановки (Z вверх) от положения ручки
на теле после сдвига/поворота тела гизмо. Ручки: hand_l/hand_r (запястья), foot_l/foot_r (лодыжки),
chest (середина плеч). Нет ключа или пустой словарь — тело не меняется.

Скелет — ключевые точки MHR70 тела постановки (KP); позвоночник — 3 звена между тазом и шеей.
Руки и ноги — аналитическая двухзвенная IK: длины костей не меняются, плоскость сгиба задаёт исходное
направление локтя/колена (полюс), сустав не выгибается в обратную сторону, недостижимая цель — конечность
выпрямляется к цели. Грудь — изгиб позвоночника поровну в 3 позвонках, таз на месте, угол ограничен.
Кисть/стопа сохраняют ориентацию (двигаются параллельно), голова и руки следуют за грудью.

Веса LBS: у меша постановки (MHR) и у манекена MPFB в 3D-окне готовых весов в кэше нет (ref.npz, manual_base.npz,
fit_state.npz хранят только вершины, KP и группы опоры), поэтому они строятся по скелету: метка кости — ближайшая
по поверхности меша (Дейкстра от затравок на осях костей; касающиеся части тела не слипаются), затем метки
сглаживаются по рёбрам меша у суставов (ширина blend_m). Тот же алгоритм — во фронтенде (web/poseRig.js).
Чистый numpy/scipy, без GPU.
"""
import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np

CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "pose_edits.yaml"

HANDLES = ("hand_l", "hand_r", "foot_l", "foot_r", "chest")
BONES = ("pelvis", "spine1", "spine2", "chest", "head",
         "upperarm_l", "forearm_l", "hand_l", "upperarm_r", "forearm_r", "hand_r",
         "thigh_l", "shin_l", "foot_l", "thigh_r", "shin_r", "foot_r")
PARENT = {"pelvis": None, "spine1": "pelvis", "spine2": "spine1", "chest": "spine2", "head": "chest",
          "upperarm_l": "chest", "forearm_l": "upperarm_l", "hand_l": "forearm_l",
          "upperarm_r": "chest", "forearm_r": "upperarm_r", "hand_r": "forearm_r",
          "thigh_l": "pelvis", "shin_l": "thigh_l", "foot_l": "shin_l",
          "thigh_r": "pelvis", "shin_r": "thigh_r", "foot_r": "shin_r"}
# Затравки меток: точки на осях костей (отрезок между суставами, доли длины).
SEGMENTS = {
    "pelvis": [("hip_l", "hip_r", (.25, .5, .75))],
    "spine1": [("j0", "j1", (.5, .75))],
    "spine2": [("j1", "j2", (.25, .5, .75))],
    "chest": [("j2", "neck", (.25, .5, .75)), ("neck", "sh_l", (.5,)), ("neck", "sh_r", (.5,))],
    "head": [("neck", "head", (.5, 1.))],
    **{f"{bone}_{s}": [seg] for s in "lr" for bone, seg in (
        ("upperarm", (f"sh_{s}", f"el_{s}", (.25, .5, .75))), ("forearm", (f"el_{s}", f"wr_{s}", (.25, .5, .75))),
        ("hand", (f"wr_{s}", f"hd_{s}", (.5, 1.))), ("thigh", (f"hip_{s}", f"kn_{s}", (.25, .5, .75))),
        ("shin", (f"kn_{s}", f"an_{s}", (.25, .5, .75))))},
    **{f"foot_{s}": [(f"an_{s}", f"toe_{s}", (.3, .6, .9)), (f"an_{s}", f"heel_{s}", (.5,))] for s in "lr"},
}
# Ключевые точки MHR70 -> кость (остальные, 63–68, — по ближайшей оси кости).
KP_BONE = {**{i: "head" for i in range(5)}, 5: "upperarm_l", 6: "upperarm_r", 7: "forearm_l", 8: "forearm_r",
           9: "thigh_l", 10: "thigh_r", 11: "shin_l", 12: "shin_r", 13: "foot_l", 14: "foot_r",
           **{i: "foot_l" for i in (15, 16, 17)}, **{i: "foot_r" for i in (18, 19, 20)},
           **{i: "hand_r" for i in range(21, 42)}, **{i: "hand_l" for i in range(42, 63)}, 69: "chest"}


@lru_cache(maxsize=1)
def cfg():
    """Пороги — config/pose_edits.yaml (не placement.yaml: тот входит в ключ кэша постановки)."""
    import yaml
    data = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{CONFIG_PATH}: ожидается mapping")
    return data


# ---------------------------------------------------------------------------------------------- разбор pose_edits
def parse(value, limit=None):
    """{ручка: [dx,dy,dz]} -> {ручка: np.array(3)}; нулевые сдвиги отброшены. Ошибка — ValueError по-русски."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("3DA: pose_edits должен быть словарём {ручка: [dx, dy, dz]}")
    limit = cfg()["max_offset_m"] if limit is None else limit
    out = {}
    for key, raw in value.items():
        if key not in HANDLES:
            raise ValueError(f"3DA: неизвестная ручка позы {key!r} (есть: {', '.join(HANDLES)})")
        try:
            vec = np.asarray(raw, dtype=float)
        except (TypeError, ValueError):
            raise ValueError(f"3DA: ручка {key}: нужен сдвиг [dx, dy, dz] в метрах") from None
        if vec.shape != (3,) or not np.isfinite(vec).all():
            raise ValueError(f"3DA: ручка {key}: нужен сдвиг [dx, dy, dz] в метрах")
        length = float(np.linalg.norm(vec))
        if length > limit:
            raise ValueError(f"3DA: ручка {key} сдвинута на {length:.2f} м — больше допустимых {limit} м")
        if length > 0:
            out[key] = vec
    return out


def edits_from(body_transform):
    """Ключ pose_edits строки body_transform (3DASceneEdits). Нет ключа — {} (поведение как до E10)."""
    data = json.loads(body_transform) if isinstance(body_transform, str) else body_transform
    if not isinstance(data, dict):
        return {}
    return parse(data.get("pose_edits"))


# ---------------------------------------------------------------------------------------------- скелет
def _unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else None


def joints(KP):
    K = np.asarray(KP, dtype=float)
    pelvis, neck = (K[9] + K[10]) / 2, K[69]
    return dict(j0=pelvis, j1=pelvis + (neck - pelvis) / 3, j2=pelvis + 2 * (neck - pelvis) / 3, neck=neck,
                head=(K[3] + K[4]) / 2, chest=(K[5] + K[6]) / 2,
                sh_l=K[5], sh_r=K[6], el_l=K[7], el_r=K[8], wr_l=K[62], wr_r=K[41],
                hd_l=K[[49, 53, 57, 61]].mean(0), hd_r=K[[28, 32, 36, 40]].mean(0),
                hip_l=K[9], hip_r=K[10], kn_l=K[11], kn_r=K[12], an_l=K[13], an_r=K[14],
                toe_l=(K[15] + K[16]) / 2, toe_r=(K[18] + K[19]) / 2, heel_l=K[17], heel_r=K[20])


def handle_points(KP):
    J = joints(KP)
    return dict(hand_l=J["wr_l"], hand_r=J["wr_r"], foot_l=J["an_l"], foot_r=J["an_r"], chest=J["chest"])


def forward(J):
    """«Вперёд» тела: left × up (как body.facing); запасной полюс для прямой руки/ноги."""
    up = _unit(J["neck"] - J["j0"])
    left = (J["hip_l"] - J["hip_r"]) + (J["sh_l"] - J["sh_r"])
    if up is None:
        return np.array([1., 0., 0.])
    f = np.cross(left, up)
    f = _unit(f - (f @ up) * up)
    return f if f is not None else np.array([1., 0., 0.])


def segment_distance(points, a, b):
    ab = b - a
    t = np.clip(((points - a) @ ab) / max(float(ab @ ab), 1e-12), 0, 1)
    return np.linalg.norm(points - (a + t[:, None] * ab), axis=1)


def nearest_bone(points, J):
    """Индекс кости с ближайшей осью (запасной путь для вершин вне меша-графа и точек KP 63–68)."""
    best = np.full(len(points), np.inf)
    out = np.zeros(len(points), dtype=int)
    for bi, bone in enumerate(BONES):
        for a, b, _ in SEGMENTS[bone]:
            d = segment_distance(points, J[a], J[b])
            closer = d < best
            best[closer], out[closer] = d[closer], bi
    return out


# ---------------------------------------------------------------------------------------------- IK
def rot_axis(axis, angle):
    x, y, z = axis
    c, s = math.cos(angle), math.sin(angle)
    C = 1 - c
    return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                     [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                     [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])


def compose(outer, inner):
    """(R,t)∘(R,t): сначала inner, затем outer."""
    return outer[0] @ inner[0], outer[0] @ inner[1] + outer[1]


def about(R, point):
    point = np.asarray(point, dtype=float)
    return R, point - R @ point


def apply(T, p):
    return T[0] @ np.asarray(p, dtype=float) + T[1]


IDENTITY = (np.eye(3), np.zeros(3))


def _frame(e, n):
    return np.column_stack([e, n, np.cross(e, n)])


def two_bone(a, b, c, target, pole_hint, eps):
    """Аналитическая IK плечо(a)–локоть(b)–кисть(c) к цели. Полюс — исходное смещение локтя от линии a–c
    (для прямой конечности — pole_hint). Возвращает b', c', повороты костей R1, R2 и сведения."""
    a, b, c, target = (np.asarray(v, dtype=float) for v in (a, b, c, target))
    l1, l2 = float(np.linalg.norm(b - a)), float(np.linalg.norm(c - b))
    u = _unit(c - a)
    if u is None:
        u = _unit(b - a)
    pole = (b - a) - ((b - a) @ u) * u
    if np.linalg.norm(pole) < 1e-6 * max(l1, 1e-9):
        pole = np.asarray(pole_hint, dtype=float) - (np.asarray(pole_hint, dtype=float) @ u) * u
    pole = _unit(pole)
    if pole is None:                                   # pole_hint вдоль конечности — любой перпендикуляр
        pole = _unit(np.cross(u, [1., 0., 0.])) if abs(u[0]) < .9 else _unit(np.cross(u, [0., 1., 0.]))
    normal = _unit(np.cross(u, pole))
    reach = target - a
    dist = float(np.linalg.norm(reach))
    direction = reach / dist if dist > 1e-9 else u
    dmin, dmax = abs(l1 - l2) + eps * (l1 + l2), (l1 + l2) * (1 - eps)
    d = min(max(dist, dmin), dmax)
    side = pole - (pole @ direction) * direction       # плоскость сгиба: направление на цель + исходный полюс
    side = _unit(side)
    if side is None:
        side = _unit(np.cross(normal, direction))
    cos_a = np.clip((l1 * l1 + d * d - l2 * l2) / (2 * l1 * d), -1., 1.)
    sin_a = math.sqrt(max(0., 1 - cos_a * cos_a))      # sin ≥ 0: локоть только в сторону полюса
    b2 = a + l1 * (cos_a * direction + sin_a * side)
    c2 = a + d * direction
    normal2 = _unit(np.cross(direction, side))
    R1 = _frame((b2 - a) / l1, normal2) @ _frame((b - a) / l1, normal).T
    R2 = _frame((c2 - b2) / l2, normal2) @ _frame((c - b) / l2, normal).T
    info = dict(reached=bool(dmin <= dist <= dmax), straightened=bool(dist > dmax), error_m=float(np.linalg.norm(c2 - target)),
                lengths_m=[round(l1, 5), round(l2, 5)],
                bend_deg=round(math.degrees(math.acos(np.clip((a - b2) @ (c2 - b2) / (l1 * l2), -1, 1))), 2))
    return b2, c2, R1, R2, info


def spine_bend(J, target, conf):
    """Изгиб корпуса: поворот поровну в 3 позвонках (j0, j1, j2) вокруг одной оси, таз на месте.
    Угол подбирается, чтобы направление таз→грудь совпало с направлением на цель; предел spine_max_deg."""
    P, C = J["j0"], J["chest"]
    v0, v1 = C - P, np.asarray(target, dtype=float) - P
    axis = np.cross(v0, v1)
    s = float(np.linalg.norm(axis))
    limit = math.radians(conf["spine_max_deg"])
    if s < 1e-9:
        return {k: IDENTITY for k in ("spine1", "spine2", "chest")}, dict(bend_deg=0., limited=False, error_m=float(
            np.linalg.norm(C - target)))
    axis /= s
    want = math.atan2(s, float(v0 @ v1))
    limited = want > limit
    want = min(want, limit)

    def chain(theta):
        R = rot_axis(axis, theta / 3)
        t1 = about(R, J["j0"])
        t2 = compose(t1, about(R, J["j1"]))
        t3 = compose(t2, about(R, J["j2"]))
        return dict(spine1=t1, spine2=t2, chest=t3)

    def achieved(T):
        v = apply(T["chest"], C) - P
        return math.atan2(float(np.cross(v0, v) @ axis), float(v0 @ v))

    theta = want
    for _ in range(int(conf["spine_iters"])):
        got = achieved(chain(theta))
        if abs(got) < 1e-12:
            break
        theta = min(theta * want / got, limit)
    T = chain(theta)
    return T, dict(bend_deg=round(math.degrees(theta), 3), limited=bool(limited),
                   error_m=float(np.linalg.norm(apply(T["chest"], C) - target)))


def solve(KP, edits, conf=None):
    """Трансформации костей {кость: (R, t)} и сведения по ручкам. edits — результат parse()."""
    conf = conf or cfg()
    J = joints(KP)
    T = {b: IDENTITY for b in BONES}
    info = {}
    if "chest" in edits:
        spine, info["chest"] = spine_bend(J, J["chest"] + edits["chest"], conf)
        T.update(spine)
        for b in ("head", "upperarm_l", "forearm_l", "hand_l", "upperarm_r", "forearm_r", "hand_r"):
            T[b] = T["chest"]
    fwd = forward(J)
    for s in "lr":
        for handle, parent, names, bones, pole in (
                (f"hand_{s}", "chest", ("sh", "el", "wr"), ("upperarm", "forearm", "hand"), -fwd),
                (f"foot_{s}", "pelvis", ("hip", "kn", "an"), ("thigh", "shin", "foot"), fwd)):
            if handle not in edits:
                continue
            Tp = T[parent]
            a, b, c = (apply(Tp, J[f"{n}_{s}"]) for n in names)
            target = J[f"{names[2]}_{s}"] + edits[handle]
            b2, c2, R1, R2, inf = two_bone(a, b, c, target, pole, conf["reach_eps"])
            T[f"{bones[0]}_{s}"] = compose((R1, a - R1 @ a), Tp)
            T[f"{bones[1]}_{s}"] = compose((R2, b2 - R2 @ b), Tp)
            T[f"{bones[2]}_{s}"] = compose((np.eye(3), c2 - c), Tp)
            info[handle] = inf
    return T, info


def kp_bones(KP):
    K = np.asarray(KP, dtype=float)
    near = nearest_bone(K, joints(K))
    return np.array([BONES.index(KP_BONE[i]) if i in KP_BONE else near[i] for i in range(len(K))])


# ---------------------------------------------------------------------------------------------- веса LBS
def vertex_normals(X, F):
    n = np.zeros_like(X)
    fn = np.cross(X[F[:, 1]] - X[F[:, 0]], X[F[:, 2]] - X[F[:, 0]])
    for k in range(3):
        np.add.at(n, F[:, k], fn)
    length = np.linalg.norm(n, axis=1)
    return n / np.maximum(length, 1e-20)[:, None]


def skin_weights(X, F, KP, conf=None, top=4):
    """Веса LBS по скелету: (idx [nV×top] номер кости в BONES, w [nV×top]), сумма весов = 1."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra
    from scipy.spatial import cKDTree
    conf = conf or cfg()
    X = np.asarray(X, dtype=float)
    F = np.asarray(F, dtype=np.int64)
    nV = len(X)
    J = joints(KP)
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    e = np.unique(np.sort(e, axis=1), axis=0)
    e = e[e[:, 0] != e[:, 1]]
    length = np.linalg.norm(X[e[:, 0]] - X[e[:, 1]], axis=1)
    G = coo_matrix((np.concatenate([length, length]) + 1e-12, (np.concatenate([e[:, 0], e[:, 1]]),
                                                              np.concatenate([e[:, 1], e[:, 0]]))), shape=(nV, nV)).tocsr()
    N = vertex_normals(X, F)
    tree = cKDTree(X)
    k = min(int(conf["seed_candidates"]), nV)
    seed_bone = {}
    for bi, bone in enumerate(BONES):
        for a, b, ts in SEGMENTS[bone]:
            for t in ts:
                s = J[a] + t * (J[b] - J[a])
                _, cand = tree.query(s, k=k)
                cand = np.atleast_1d(cand)
                outward = [int(i) for i in cand if N[i] @ (X[i] - s) > 0]
                v = outward[0] if outward else int(cand[0])
                seed_bone.setdefault(v, bi)
    seeds = np.array(list(seed_bone), dtype=int)
    _, _, src = dijkstra(G, directed=False, indices=seeds, min_only=True, return_predecessors=True)
    label = nearest_bone(X, J)
    ok = src >= 0
    lookup = np.full(nV, -1, dtype=int)
    lookup[seeds] = [seed_bone[v] for v in seeds]
    label[ok] = lookup[src[ok]]
    W = np.zeros((nV, len(BONES)))
    W[np.arange(nV), label] = 1.
    A = coo_matrix((np.ones(2 * len(e)), (np.concatenate([e[:, 0], e[:, 1]]), np.concatenate([e[:, 1], e[:, 0]]))),
                   shape=(nV, nV)).tocsr()
    degree = np.maximum(np.asarray(A.sum(axis=1)).reshape(-1), 1.)
    iters = blend_iters(float(length.mean()) if len(length) else 1., conf)
    for _ in range(iters):
        W = .5 * W + .5 * (A @ W) / degree[:, None]
    return top_weights(W, conf["weight_min"], top)


def blend_iters(mean_edge, conf):
    return int(np.clip(round(2 * (conf["blend_m"] / max(mean_edge, 1e-6)) ** 2), conf["blend_iters_min"],
                       conf["blend_iters_max"]))


def top_weights(W, minimum, top=4):
    idx = np.argsort(-W, axis=1, kind="stable")[:, :top]
    w = np.take_along_axis(W, idx, axis=1)
    w[w < minimum] = 0.
    w /= np.maximum(w.sum(axis=1, keepdims=True), 1e-12)
    return idx, w


def deform(X, idx, w, T):
    X = np.asarray(X, dtype=float)
    Rs = np.stack([T[b][0] for b in BONES])
    ts = np.stack([T[b][1] for b in BONES])
    out = np.zeros_like(X)
    for k in range(idx.shape[1]):
        out += w[:, k, None] * (np.einsum("vij,vj->vi", Rs[idx[:, k]], X) + ts[idx[:, k]])
    return out


def apply_edits(X, KP, F, edits, conf=None):
    """Тело (вершины X, KP MHR70) после правок ручек; edits — сдвиги в тех же осях, что X."""
    conf = conf or cfg()
    if not edits:
        return np.asarray(X, dtype=float), np.asarray(KP, dtype=float), {}
    T, info = solve(KP, edits, conf)
    idx, w = skin_weights(X, F, KP, conf)
    X2 = deform(X, idx, w, T)
    K = np.asarray(KP, dtype=float)
    bones = kp_bones(K)
    K2 = np.array([apply(T[BONES[b]], p) for p, b in zip(K, bones)])
    return X2, K2, info


def edit_base(base, faces, edits, transform):
    """Правки ручек к закэшированной постановке до сдвига тела. Сдвиги ручек заданы в мире после сдвига тела
    (transform(X, KP) -> X', KP' — тот же сдвиг/поворот, что применит revalidate), поэтому переводятся в оси
    исходного тела обратной линейной частью сдвига (IK и LBS не зависят от жёсткого сдвига и поворота)."""
    probe = np.asarray(base["KP"], dtype=float).mean(0) + np.vstack([np.zeros(3), np.eye(3)])
    moved, _ = transform(probe, np.asarray(base["KP"], dtype=float))
    L = (np.asarray(moved[1:]) - np.asarray(moved[0])).T
    local = {k: np.linalg.solve(L, v) for k, v in edits.items()}
    X, KP, info = apply_edits(base["X"], base["KP"], faces, local)
    before, after = bone_lengths(base["KP"]), bone_lengths(KP)
    report = dict(handles={k: [round(float(x), 4) for x in v] for k, v in edits.items()},
                  solve=json.loads(json.dumps(info, default=float)),
                  max_bone_length_change_m=round(max(abs(after[k] - before[k]) for k in before), 6))
    return dict(base, X=X, KP=KP), report


def bone_lengths(KP):
    J = joints(KP)
    pairs = {f"{n}_{s}": (f"{a}_{s}", f"{b}_{s}") for s in "lr" for n, a, b in
             (("upperarm", "sh", "el"), ("forearm", "el", "wr"), ("thigh", "hip", "kn"), ("shin", "kn", "an"))}
    return {k: float(np.linalg.norm(J[a] - J[b])) for k, (a, b) in pairs.items()}
