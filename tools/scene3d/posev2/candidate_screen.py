"""E1c: числовая проверка постановленного тела «в кадре» — только для 3DAPlaceCandidates (candidates_cli, candidates).

(а) кадр: вершины тела проецируются камерой плиты; in_frame_frac — доля внутри картинки с полем frame_pad, голова
    обязательна;
(б) «висит/утоплен»: точки контакта обязательных частей проецируются в пиксели, их глубина от камеры сравнивается с
    картой глубины локации (Plate.Z) в тех же пикселях. Зазор по лучу камеры (положительный — тело ближе к камере,
    чем поверхность): screen_gap_m = (Z_карты − z_тела) × room_scale; vertical_gap_m — то же по вертикали
    (высота точки тела над точкой поверхности, которую видит камера в этом пикселе) — для settle;
(в) любая ошибка измерения пишется в measure_error и штрафуется, идеалом не считается.

Зазор по лучу — не нормаль: загораживающий предмет перед телом даёт ложное «утоплено», ошибки карты глубины —
ложный зазор; при взгляде вскользь (камера почти вровень с поверхностью) разница глубин в 5–7 раз больше высоты, поэтому
статус (висит/утоплено) и settle берутся по vertical_gap_m (cfg.gap_basis = vertical), а screen_gap_m — как есть, для
отчёта и сравнения (gap_basis = depth возвращает решение по глубине). Пороги и факторы — config/placement_screen.yaml;
place.py и 3DAPlacePose не затрагиваются.
Plate — любой объект с R (мир←камера CV), C, fx, cx, cy, Wp, Hp, Z и project(X) -> (u, v, z) (pose_fit.Plate).
"""
import math
from pathlib import Path

import numpy as np

CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "placement_screen.yaml"


def load_config(path=None):
    import yaml
    return yaml.safe_load(Path(path or CONFIG_PATH).read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------------------------ кадр
def _inside(u, v, z, plate, pad):
    return ((z > 0) & (u >= pad * plate.Wp) & (u < (1 - pad) * plate.Wp)
            & (v >= pad * plate.Hp) & (v < (1 - pad) * plate.Hp))


def _project(plate, points, label):
    """(u, v, z) вершин; точки за камерой (z ≤ 0) — вне кадра, без ошибки; нечисловое — ValueError."""
    points = np.asarray(points, float)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise ValueError(f"{label}: нужен непустой массив Nx3")
    if not np.isfinite(points).all():
        raise ValueError(f"{label}: нечисловая вершина")
    u, v, z = (np.asarray(a, float) for a in plate.project(points))
    front = z > 0
    if not np.isfinite(z).all() or not (np.isfinite(u[front]).all() and np.isfinite(v[front]).all()):
        raise ValueError(f"{label}: нечисловая проекция")
    return u, v, z


def frame_metrics(plate, X, head_idx, cfg):
    """{in_frame_frac, head_in_frame} по вершинам X; ошибка — ValueError."""
    pad = float(cfg["frame_pad"])
    if not math.isfinite(pad) or not 0 <= pad < 0.5:
        raise ValueError("frame_pad должен быть в [0, 0.5)")
    u, v, z = _project(plate, X, "тело")
    inside = _inside(u, v, z, plate, pad)
    head = None
    if head_idx is not None and len(head_idx):
        head = bool(inside[np.asarray(head_idx, int)].all())
    return float(inside.mean()), head


def location_basis(plate):
    """Оси «вправо / вглубь / вверх» в горизонтали камеры (как manual_pose.location_basis; смещения body_dx/dy/dz)."""
    x = np.array([plate.R[0, 0], plate.R[1, 0], 0.0], float)
    x /= np.linalg.norm(x)
    y = np.cross([0.0, 0.0, 1.0], x)
    if y @ plate.R[:, 2] < 0:
        y = -y
    return np.column_stack([x, y, [0.0, 0.0, 1.0]])


def frame_fix(plate, X, head_idx, cfg):
    """Минимальный сдвиг тела (body_dx, body_dy: вправо / вглубь, м), после которого кадр проходит; None — не нужен
    или не найден (масштаб фигуры не меняем). Ищем по сетке «дальше от камеры × вбок», меньший сдвиг — раньше."""
    fix = cfg["frame_fix"]
    sample = np.asarray(X, float)
    stride = max(1, len(sample) // int(fix["sample_vertices"]))
    sample = sample[::stride]
    head = None if head_idx is None or not len(head_idx) else np.asarray(X, float)[np.asarray(head_idx, int)]
    pad, need = float(cfg["frame_pad"]), float(cfg["min_in_frame_frac"])
    basis = location_basis(plate)

    def passes(shift):
        u, v, z = _project(plate, sample + shift, "сдвиг")
        if float(_inside(u, v, z, plate, pad).mean()) < need:
            return False
        if cfg.get("head_required", True) and head is not None:
            u, v, z = _project(plate, head + shift, "голова")
            return bool(_inside(u, v, z, plate, pad).all())
        return True
    if passes(np.zeros(3)):
        return None
    options = []
    steps = int(round(float(fix["max_away_m"]) / float(fix["step_m"])))
    for away in range(0, steps + 1):
        for lateral in fix["lateral_m"]:
            options.append((away * float(fix["step_m"]) + abs(float(lateral)), float(lateral),
                            away * float(fix["step_m"])))
    for _, lateral, away in sorted(options):
        if (lateral or away) and passes(basis @ np.array([lateral, away, 0.0])):
            return [round(lateral, 3), round(away, 3)]
    return None


# ------------------------------------------------------------------------------------------------ контакт
def contact_indices(X, reg, parts, lowest_only, cfg):
    """Часть → индексы точек контакта: вершины региона не выше самой низкой вершины части + contact_band_m
    (не меньше contact_min_points самых низких). lowest_only — берётся только часть с самой низкой вершиной."""
    X, reg = np.asarray(X, float), np.asarray(reg)
    found = {}
    for part in parts:
        idx = np.flatnonzero(reg == part)
        if len(idx):
            found[part] = idx
    if lowest_only and found:
        best = min(found, key=lambda p: float(X[found[p], 2].min()))
        found = {best: found[best]}
    out = {}
    for part, idx in found.items():
        z = X[idx, 2]
        low = idx[z <= z.min() + float(cfg["contact_band_m"])]
        if len(low) < int(cfg["contact_min_points"]):
            low = idx[np.argsort(z)[:int(cfg["contact_min_points"])]]
        out[part] = low
    return out


def _unit_rays(plate, u, v):
    """Направления лучей камеры в мире при z_камеры = 1 (нужна только вертикальная компонента)."""
    d = np.column_stack(((u - plate.cx) / plate.fx, (v - plate.cy) / plate.fx, np.ones(len(u))))
    return d @ np.asarray(plate.R, float).T


def part_gaps(plate, points, scale, cfg):
    """Зазоры одной части по её точкам. -> dict(screen_gap_m, vertical_gap_m, points, valid, measure_error)."""
    rec = {"screen_gap_m": None, "vertical_gap_m": None, "points": int(len(points)), "valid": 0,
           "measure_error": None}
    try:
        u, v, z = _project(plate, points, "контакт")
        inside = _inside(u, v, z, plate, 0.0)
        ix = np.clip(np.rint(u).astype(np.int64), 0, plate.Wp - 1)
        iy = np.clip(np.rint(v).astype(np.int64), 0, plate.Hp - 1)
        scene = np.asarray(plate.Z, float)[iy, ix]
        ok = inside & np.isfinite(scene) & (scene > 0)
        rec["valid"] = int(ok.sum())
        if ok.mean() < float(cfg["contact_min_valid_frac"]):
            raise ValueError(f"глубина карты есть только у {ok.sum()} из {len(ok)} точек контакта")
        depth_gap = (scene[ok] - z[ok]) * scale
        vertical = -(scene[ok] - z[ok]) * _unit_rays(plate, u[ok], v[ok])[:, 2] * scale
        if not (np.isfinite(depth_gap).all() and np.isfinite(vertical).all()):
            raise ValueError("нечисловой зазор")
        rec["screen_gap_m"] = round(float(np.median(depth_gap)), 4)
        rec["vertical_gap_m"] = round(float(np.median(vertical)), 4)
    except ValueError as exc:
        rec["measure_error"] = str(exc)
    return rec


def gap_of(rec, cfg):
    """Зазор части, по которому принимается решение: vertical_gap_m (по умолчанию) или screen_gap_m (gap_basis)."""
    return rec["screen_gap_m"] if cfg.get("gap_basis") == "depth" else rec["vertical_gap_m"]


def contact_status(parts, cfg):
    """(статус, screen_gap_m, vertical_gap_m): floating | penetrating | contact | unknown (нет ни одной части).
    Решает худшая часть по gap_of; зазоры в ответе — у решающей части (при контакте — медианы частей)."""
    done = [r for r in parts.values() if r["measure_error"] is None]
    if not done:
        return "unknown", None, None
    high, low = max(gap_of(r, cfg) for r in done), min(gap_of(r, cfg) for r in done)
    if high > float(cfg["floating_gap_m"]):
        status, pick = "floating", max(done, key=lambda r: gap_of(r, cfg))
    elif low < -float(cfg["penetrating_gap_m"]):
        status, pick = "penetrating", min(done, key=lambda r: gap_of(r, cfg))
    else:
        status, pick = "contact", None
    if pick is None:
        return (status, round(float(np.median([r["screen_gap_m"] for r in done])), 4),
                round(float(np.median([r["vertical_gap_m"] for r in done])), 4))
    return status, pick["screen_gap_m"], pick["vertical_gap_m"]


def required_parts(pose_class, support_parts, cfg):
    """(части, lowest_only) для класса позы; lie/recline — support_parts запроса, если они заданы."""
    parts = list((cfg["contact_parts"] or {}).get(pose_class) or [])
    if pose_class in ("lie", "recline") and support_parts:
        asked = [p for p in support_parts if p in ("pelvis", "torso") or "." in p]
        parts = asked or parts
    return parts, pose_class in (cfg.get("lowest_only") or [])


def measure(plate, X, reg, pose_class, support_parts, cfg, room_scale=None):
    """Полная проверка кандидата; не бросает: сбои — в measure_error."""
    scale_source = "room_scale" if room_scale else "plate_units"
    scale = float(room_scale) if room_scale else float(cfg["room_scale_default"])
    out = {"in_frame_frac": None, "head_in_frame": None, "min_in_frame_frac": float(cfg["min_in_frame_frac"]),
           "frame_ok": None, "contact_status": "n/a", "screen_gap_m": None, "vertical_gap_m": None, "parts": {},
           "room_scale": scale, "scale_source": scale_source, "measure_error": [], "frame_fix_dxdy_m": None}
    errors = out["measure_error"]
    reg = np.asarray(reg)
    head_idx = np.flatnonzero(reg == "head")
    try:
        frac, head = frame_metrics(plate, X, head_idx, cfg)
        out["in_frame_frac"], out["head_in_frame"] = round(frac, 4), head
        head_bad = bool(cfg.get("head_required", True)) and head is not True
        out["frame_ok"] = frac >= float(cfg["min_in_frame_frac"]) and not head_bad
        if not out["frame_ok"]:
            out["frame_fix_dxdy_m"] = frame_fix(plate, X, head_idx, cfg)
    except ValueError as exc:
        errors.append(f"кадр: {exc}")
    parts, lowest = required_parts(pose_class, support_parts, cfg)
    if not parts:
        out["contact_status"] = "n/a"
    elif not (math.isfinite(scale) and scale > 0):
        errors.append("контакт: room_scale должен быть конечным и положительным")
        out["contact_status"] = "unknown"
    else:
        try:
            idx = contact_indices(X, reg, parts, lowest, cfg)
            missing = [p for p in parts if p not in idx and not lowest]
            if missing:
                raise ValueError("нет вершин части " + ",".join(missing))
            if not idx:
                raise ValueError("нет вершин обязательных частей")
            X = np.asarray(X, float)
            out["parts"] = {p: part_gaps(plate, X[i], scale, cfg) for p, i in idx.items()}
            for p, rec in out["parts"].items():
                if rec["measure_error"]:
                    errors.append(f"контакт {p}: {rec['measure_error']}")
            out["contact_status"], out["screen_gap_m"], out["vertical_gap_m"] = contact_status(out["parts"], cfg)
        except ValueError as exc:
            errors.append(f"контакт: {exc}")
            out["contact_status"] = "unknown"
    out["auto_flag"] = auto_flag(out)
    return out


def auto_flag(screen):
    """Автофлаг по измерению: cropped / floating / penetrating (через «+») / measure_error / ok."""
    if not isinstance(screen, dict):
        return "measure_error"
    flags = []
    if screen.get("frame_ok") is False:
        flags.append("cropped")
    if screen.get("contact_status") in ("floating", "penetrating"):
        flags.append(screen["contact_status"])
    if flags:
        return "+".join(flags)
    return "measure_error" if screen.get("measure_error") else "ok"


# ------------------------------------------------------------------------------------------------ оценка и settle
def score_factor(screen, cfg):
    """(фактор 0..1, {имя: число}) для итога кандидата. Нет измерения вообще — ошибка измерения, не идеал."""
    if not isinstance(screen, dict):
        factor = float(cfg["measure_error_factor"])
        return factor, {"screen_factor": round(factor, 3), "screen_measured": 0.0}
    factor = 1.0
    if screen.get("measure_error"):
        factor *= float(cfg["measure_error_factor"])
    if screen.get("frame_ok") is False:
        factor *= float(cfg["frame_fail_factor"])
    if screen.get("contact_status") == "floating":
        factor *= float(cfg["floating_factor"])
    elif screen.get("contact_status") == "penetrating":
        factor *= float(cfg["penetrating_factor"])
    return factor, {"screen_factor": round(factor, 3), "screen_measured": 1.0}


def settle_dz(screen, cfg):
    """Вертикальный сдвиг тела (body_dz, м), чтобы контакт лёг на поверхность карты глубины; None — не нужен.
    Только когда все измеренные части висят (тело целиком над поверхностью) — вниз на наименьший вертикальный зазор
    (оставляя target_m), или все утоплены — вверх. Если одна часть лежит, а другая висит (наклон корпуса не тот),
    вертикальный сдвиг утопил бы лежащую — None, это дело наклона, а не высоты."""
    if not isinstance(screen, dict) or screen.get("measure_error"):
        return None
    done = [r for r in (screen.get("parts") or {}).values()
            if isinstance(r, dict) and r.get("measure_error") is None and isinstance(r.get("vertical_gap_m"), (int, float))]
    if not done:
        return None
    s = cfg["settle"]
    if all(gap_of(r, cfg) > float(cfg["floating_gap_m"]) for r in done):
        dz = -min(max(min(r["vertical_gap_m"] for r in done) - float(s["target_m"]), 0.0), float(s["max_down_m"]))
    elif all(gap_of(r, cfg) < -float(cfg["penetrating_gap_m"]) for r in done):
        dz = min(max(-max(r["vertical_gap_m"] for r in done) + float(s["target_m"]), 0.0), float(s["max_up_m"]))
    else:
        return None
    return round(dz, 4) if abs(dz) >= float(s["min_m"]) else None


def floating_parts(screen, cfg):
    """Имена измеренных частей, которые висят по карте глубины (screen_gap_m > floating_gap_m)."""
    parts = (screen or {}).get("parts") if isinstance(screen, dict) else None
    return [name for name, r in (parts or {}).items()
            if isinstance(r, dict) and r.get("measure_error") is None
            and isinstance(r.get("screen_gap_m"), (int, float)) and gap_of(r, cfg) > float(cfg["floating_gap_m"])]


def torso_floats_while_others_rest(screen, cfg):
    """Торс висит по карте глубины, а другая измеренная часть (таз) лежит: поправляет наклон, а не высота."""
    floating = floating_parts(screen, cfg)
    measured = [n for n, r in (screen.get("parts") or {}).items() if r.get("measure_error") is None]
    return "torso" in floating and len(floating) < len(measured)


# ------------------------------------------------------------------------------------------------ задание кандидата
def read_obj_vertices(path):
    rows = []
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("v "):
                rows.append(line.split()[1:4])
    if not rows:
        raise ValueError("в body.obj нет вершин")
    return np.asarray(rows, float)


def measure_job(job, plate, pose_class, support_parts, cfg, room_scale=None):
    """Измерение по каталогу результата постановки: fit/body.obj (вершины тела в системе плиты) и
    fit/label_joints.npz (kp — для регионов тела). Не бросает."""
    try:
        import body
        X = read_obj_vertices(Path(job) / "fit" / "body.obj")
        with np.load(Path(job) / "fit" / "label_joints.npz", allow_pickle=False) as data:
            kp = np.asarray(data["kp"], float)
        reg = body.regions(X, kp)
    except Exception as exc:  # noqa: BLE001 — нет тела/регионов = ошибка измерения, а не идеал
        out = {"in_frame_frac": None, "head_in_frame": None, "frame_ok": None, "contact_status": "unknown",
               "screen_gap_m": None, "vertical_gap_m": None, "parts": {}, "frame_fix_dxdy_m": None,
               "measure_error": [f"тело: {type(exc).__name__}: {exc}"]}
        out["auto_flag"] = auto_flag(out)
        return out
    parts = [p.strip() for p in str(support_parts or "").split(",") if p.strip()]
    return measure(plate, X, reg, pose_class, parts, cfg, room_scale)
