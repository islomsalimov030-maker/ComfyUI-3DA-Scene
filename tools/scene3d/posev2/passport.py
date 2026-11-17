"""E2 (0.3.36): паспорт локации — разовая разметка комнаты, которую берут все строки с этой локацией.

Только чистая логика (numpy, PIL; без GPU и CLI). Узел 3DALocationPassport считает паспорт один раз по выходам
3DALocation (objects + профиль + plate.json) и кладёт его в кэш <user>/3da/passports/<id>.json; человек смотрит
preview, при необходимости правит JSON и ставит "confirmed": true. 3DAPlaceCandidates с входом passport берёт
опоры, плоскости и room_scale из паспорта, без пересчёта (classify/enrich/room_scale из candidates.py).

Схема 1 (координаты — мир плиты 3DA: z вверх, пол z=0, камера в (0, 0, camera_height_m); метры):
  id, image_sha1, version, room_scale, room_scale_flag, camera_height_m, floor_visible, floor_level_m,
  objects: [{id, type, bbox_m [[x,y,z],[x,y,z]], front_dir [x,y] | null, front_from, backrest, backrest_ids,
             size_m, support_planes: [{kind: seat|top|bed|floor, height_m (над полом), polygon [[x,y],...],
                                       area_m2, surface_ids [id поверхности профиля]}]}],
  confirmed, notes.
Тип мебели — геометрическая метка E1 (candidates.classify), не распознавание: поэтому нужна проверка человеком.
Все числа — config/placement.yaml, разделы candidates и passport.
"""
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import candidates as cm  # noqa: E402

SCHEMA = 1
KINDS = ("seat", "top", "bed", "floor")


def load_config(path=None):
    """Раздел candidates (как у 3DAPlaceCandidates) + раздел passport."""
    import yaml
    cfg = cm.load_config(path)
    data = yaml.safe_load(Path(path or cm.CONFIG_PATH).read_text(encoding="utf-8"))
    cfg["passport"] = dict(data["passport"])
    return cfg


def image_sha1(pixels):
    """sha1 картинки по пикселям uint8 (H×W×3) и размеру — не зависит от кодировщика PNG."""
    import numpy as np
    a = np.ascontiguousarray(pixels, dtype=np.uint8)
    h = hashlib.sha1()
    h.update(("%dx%dx%d:" % a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def safe_id(value):
    v = str(value or "").strip()
    if not v or len(v) > 96 or not all(c.isalnum() or c in "_-." for c in v) or v.startswith("."):
        raise ValueError("passport_id: только латиница, цифры, _ - . (до 96 символов)")
    return v


# ------------------------------------------------------------------------------------------------ геометрия
def hull(points):
    """Выпуклая оболочка xy (монотонная цепь), против часовой; меньше 3 точек — как есть."""
    pts = sorted({(round(float(p[0]), 4), round(float(p[1]), 4)) for p in points})
    if len(pts) < 3:
        return [list(p) for p in pts]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return [list(p) for p in lower[:-1] + upper[:-1]]


def thin(poly, max_points):
    """Не больше max_points вершин (равномерная выборка по контуру)."""
    if len(poly) <= max_points:
        return poly
    step = len(poly) / max_points
    return [poly[int(k * step)] for k in range(max_points)]


def cell_corners(cells, cell_m):
    h = cell_m / 2
    return [(x + dx, y + dy) for x, y in ((c[0], c[1]) for c in cells) for dx in (-h, h) for dy in (-h, h)]


def polygon_of(obj, cfg):
    cells = obj.get("cells_xy")
    if cells:
        poly = hull(cell_corners(cells, cfg["cell_m"]))
    else:
        xy = obj.get("center_m") or obj.get("seed_world")
        poly = [[float(xy[0]), float(xy[1])]] if xy else []
    return [[round(p[0], 3), round(p[1], 3)] for p in thin(poly, int(cfg["passport"]["hull_max_points"]))]


def _center(cells, obj):
    if cells:
        n = len(cells)
        return [sum(c[0] for c in cells) / n, sum(c[1] for c in cells) / n]
    xy = obj.get("center_m") or obj.get("seed_world")
    return [float(xy[0]), float(xy[1])] if xy else None


def _unit(v):
    n = math.hypot(v[0], v[1])
    return [round(v[0] / n, 3), round(v[1] / n, 3)] if n > 1e-9 else None


def front_of(cells, center, back_center, cam_xy):
    """Лицевая сторона (единичный xy): от спинки к сиденью; без спинки — короткая ось следа, в сторону камеры."""
    if center is None:
        return None, None
    if back_center is not None:
        d = _unit([center[0] - back_center[0], center[1] - back_center[1]])
        if d:
            return d, "backrest"
    to_cam = [cam_xy[0] - center[0], cam_xy[1] - center[1]]
    if cells and len(cells) >= 3:
        import numpy as np
        P = np.asarray(cells, float)[:, :2]
        P = P - P.mean(0)
        val, vec = np.linalg.eigh(P.T @ P)
        if val[1] > 1.5 * max(val[0], 1e-12):          # вытянутый след: лицом — длинная сторона
            minor = vec[:, 0]
            s = 1.0 if minor[0] * to_cam[0] + minor[1] * to_cam[1] >= 0 else -1.0
            d = _unit([s * minor[0], s * minor[1]])
            if d:
                return d, "camera_long_side"
    return _unit(to_cam), "camera"


# ------------------------------------------------------------------------------------------------ сборка
def plane_kind(label, h_rel, cfg):
    kind = cfg["passport"]["kinds"].get(label, "top")
    if kind == "seat" and not (cfg["geometry"]["seat_h"][0] <= h_rel <= cfg["geometry"]["seat_h"][1]):
        kind = "top"
    return kind


def build(objects, profile=None, profile_dir=None, plate_info=None, cfg=None, pid="", sha1=""):
    """Паспорт (dict схемы 1) по objects 3DALocation и профилю. Логика меток и масштаба — E1 (candidates.py)."""
    t0 = time.monotonic()
    cfg = cfg or load_config()
    objs = cm.enrich(cm.parse_objects(objects), profile, profile_dir, cfg)
    labels = cm.classify(objs, cfg)
    scale, scale_flags = cm.room_scale(objs, labels, cfg)
    fl = cm.floor_level(objs, cfg)
    by_id = {o["id"]: o for o in objs}
    info = plate_info or {}
    cam_h = info.get("cam_height_m")
    notes = ["типы мебели — догадка по геометрии (высота, площадь, спинка); проверьте по preview"]
    # спинки диванов — не опора: их куски не входят в объекты, только backrest_ids дивана
    back_of = {}
    for sid, v in labels.items():
        b = (v.get("features") or {}).get("backrest_id")
        if v["label"] == "sofa" and b:
            back_of.setdefault(v["group"], b)
    back_ids = set(back_of.values())
    groups = {}
    skipped = []
    for o in objs:
        lab = labels.get(o["id"])
        if lab is None or o["id"] in back_ids:
            continue
        if lab["label"] == "below_floor":
            skipped.append(o["id"])
            continue
        groups.setdefault(lab["group"], []).append(o)
    if skipped:
        notes.append("ниже пола (не опоры, пропущены): " + ", ".join(skipped))
    out = []
    for gid, members in groups.items():
        label = labels[members[0]["id"]]["label"]
        planes, cells_all = [], []
        for o in sorted(members, key=lambda m: (-(float(m.get("area_m2") or 0.0)), m["id"])):
            h_rel = (cm._h(o) or 0.0) - fl
            cells_all += list(o.get("cells_xy") or [])
            planes.append({"kind": "floor" if label == "floor" else plane_kind(label, h_rel, cfg),
                           "height_m": round(h_rel, 3), "polygon": polygon_of(o, cfg),
                           "area_m2": None if o.get("area_m2") is None else round(float(o["area_m2"]), 3),
                           "surface_ids": [o["id"]]})
        center = _center(cells_all, members[0])
        back = back_of.get(gid)
        back_center = None
        if back and back in by_id:
            back_center = _center(list(by_id[back].get("cells_xy") or []), by_id[back])
        front, front_from = (None, None) if label == "floor" else front_of(cells_all, center, back_center, [0.0, 0.0])
        top = max(p["height_m"] for p in planes)
        xs = [p[0] for pl in planes for p in pl["polygon"]]
        ys = [p[1] for pl in planes for p in pl["polygon"]]
        bbox = [[round(min(xs), 3), round(min(ys), 3), 0.0 if label != "floor" else round(top, 3)],
                [round(max(xs), 3), round(max(ys), 3), round(top, 3)]] if xs else None
        ext = cm._extent(cells_all, cfg) if len(cells_all) >= 3 else None
        out.append({"id": "", "type": label, "label_source": "geometry", "bbox_m": bbox,
                    "size_m": [round(ext[0], 2), round(ext[1], 2)] if ext else None,
                    "front_dir": front, "front_from": front_from, "backrest": back is not None,
                    "backrest_ids": [back] if back else [], "support_planes": planes,
                    "_area": sum(float(p["area_m2"] or 0.0) for p in planes)})
    out.sort(key=lambda o: (o["type"] != "floor", -o["_area"]))
    for k, o in enumerate(out, 1):
        o["id"] = f"o{k}"
        o.pop("_area")
    floor_planes = [p for o in out if o["type"] == "floor" for p in o["support_planes"]]
    floor_visible = bool((info.get("floor") or {}).get("visible", True)) and bool(floor_planes)
    status = ((profile or {}).get("auto_surfaces") or {}).get("floor_status")
    if status and status != "ok":
        notes.append(f"пол у профиля не однозначный ({status}): пол — куски на уровне пола")
    if not floor_visible:
        notes.append("пол в кадре не найден")
    if scale is None:
        notes.append("масштаб комнаты не проверен (room_scale_unknown): нет объектов со смысловыми именами")
    return {"version": SCHEMA, "id": pid, "image_sha1": sha1,
            "room_scale": scale, "room_scale_flag": scale_flags[0] if scale_flags else None,
            "camera_height_m": None if cam_h is None else round(float(cam_h) - fl, 3),
            "floor_visible": floor_visible, "floor_level_m": round(fl, 3),
            "objects": out, "confirmed": False, "notes": notes,
            "computed_at": time.strftime("%Y-%m-%d %H:%M:%S"), "compute_s": round(time.monotonic() - t0, 3)}


# ------------------------------------------------------------------------------------------------ проверка схемы
def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def validate(p):
    """Список ошибок схемы (пусто — паспорт годен)."""
    if not isinstance(p, dict):
        return ["паспорт — не объект JSON"]
    err = []
    if p.get("version") != SCHEMA:
        err.append(f"version должна быть {SCHEMA}")
    for key in ("id", "image_sha1"):
        if not isinstance(p.get(key), str):
            err.append(f"{key}: нужна строка")
    if not isinstance(p.get("confirmed"), bool):
        err.append("confirmed: нужно true или false")
    rs = p.get("room_scale")
    if rs is not None and not (_num(rs) and rs > 0):
        err.append("room_scale: положительное число или null")
    if not isinstance(p.get("objects"), list):
        return err + ["objects: нужен список"]
    seen = set()
    for k, o in enumerate(p["objects"]):
        where = f"objects[{k}]"
        if not isinstance(o, dict):
            err.append(where + ": нужен объект")
            continue
        if not isinstance(o.get("id"), str) or not o["id"] or o["id"] in seen:
            err.append(where + ".id: непустая уникальная строка")
        seen.add(o.get("id"))
        if not isinstance(o.get("type"), str) or not o["type"].strip():
            err.append(where + ".type: непустая строка")
        fd = o.get("front_dir")
        if fd is not None and not (isinstance(fd, list) and len(fd) == 2 and all(_num(x) for x in fd)):
            err.append(where + ".front_dir: [x, y] или null")
        if not isinstance(o.get("backrest", False), bool):
            err.append(where + ".backrest: true/false")
        planes = o.get("support_planes")
        if not isinstance(planes, list):
            err.append(where + ".support_planes: нужен список")
            continue
        for j, pl in enumerate(planes):
            w = f"{where}.support_planes[{j}]"
            if not isinstance(pl, dict):
                err.append(w + ": нужен объект")
                continue
            if pl.get("kind") not in KINDS:
                err.append(w + ".kind: одно из " + "|".join(KINDS))
            if not _num(pl.get("height_m")):
                err.append(w + ".height_m: число")
            poly = pl.get("polygon")
            if not isinstance(poly, list) or not all(isinstance(q, list) and len(q) == 2 and all(_num(x) for x in q)
                                                     for q in poly):
                err.append(w + ".polygon: список [x, y]")
            ids = pl.get("surface_ids", [])
            if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
                err.append(w + ".surface_ids: список строк")
    return err


def loads(text):
    """(паспорт, ошибки) из строки JSON; пусто — (None, [])."""
    if isinstance(text, dict):
        p = text
    else:
        if not isinstance(text, str) or not text.strip():
            return None, []
        try:
            p = json.loads(text)
        except ValueError as exc:
            return None, [f"не JSON: {exc}"]
    if isinstance(p, dict) and "3da_error" in p:
        return None, [str(p["3da_error"])[:200]]
    err = validate(p)
    return (p if not err else None), err


# ------------------------------------------------------------------------------------------------ кэш
def passport_path(directory, pid):
    return Path(directory) / f"{safe_id(pid)}.json"


def read_cached(path, sha1):
    """(паспорт, причина): паспорт из файла, если он годен и снят с той же картинки; иначе (None, причина)."""
    path = Path(path)
    if not path.is_file():
        return None, "нет файла"
    try:
        p, err = loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        return None, f"не читается: {exc}"
    if p is None:
        return None, "файл не годен: " + "; ".join(err[:3])
    if sha1 and p.get("image_sha1") != sha1:
        return None, "image_sha1 другой картинки"
    return p, "кэш"


def write(path, p):
    """Атомарная запись (временный файл + replace)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(p, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)
    return path


def get_or_build(directory, pid, sha1, build_fn):
    """Паспорт из кэша (не пересчитывая) или новый через build_fn() с записью.
    Подтверждённый человеком файл другой картинки не перезаписывается (новый паспорт только в выход).
    -> (паспорт, источник: cache | built | built_not_saved, причина)."""
    path = passport_path(directory, pid)
    p, why = read_cached(path, sha1)
    if p is not None:
        return p, "cache", why
    fresh = build_fn()
    if path.is_file():
        try:
            old = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            old = None
        if isinstance(old, dict) and old.get("confirmed") is True:
            fresh["notes"] = list(fresh.get("notes") or []) + [f"файл {path.name} подтверждён для другой картинки — "
                                                               "не перезаписан; задайте свой passport_id"]
            return fresh, "built_not_saved", why
    write(path, fresh)
    return fresh, "built", why


# ------------------------------------------------------------------------------------------------ для кандидатов
def to_candidates(p, location_objects, cfg):
    """Паспорт -> (objs, labels, room_scale, flags, notes) в форме candidates.py, без classify/enrich.
    Поверхности, которых нет в objects локации, пропускаются; нет ни одной — ValueError (паспорт другой комнаты)."""
    loc = {o["id"]: o for o in cm.parse_objects(location_objects)}
    fl = float(p.get("floor_level_m") or 0.0)
    objs, labels, notes, missing = [], {}, [], []
    for o in p["objects"]:
        label = o["type"].strip().lower()
        group = "floor" if label == "floor" else "p_" + o["id"]
        sids = []
        for pl in o["support_planes"]:
            ids = [i for i in pl.get("surface_ids") or []]
            missing += [i for i in ids if i not in loc]
            ids = [i for i in ids if i in loc and i not in labels and i not in sids]   # кусок — в одном объекте
            if not ids:
                continue
            area = pl.get("area_m2")
            for sid in ids:
                h = fl + float(pl["height_m"])
                objs.append({"id": sid, "type": loc[sid].get("type", "other"), "h0": h, "height_m": h,
                             "area_m2": None if area is None else float(area) / len(ids),
                             "passport_object": o["id"], "plane_kind": pl["kind"]})
                sids.append(sid)
        for sid in sids:
            labels[sid] = {"label": label, "group": group,
                           "features": {"group_ids": list(sids), "passport_object": o["id"],
                                        "h_rel_m": round(next(x["h0"] for x in objs if x["id"] == sid) - fl, 3)}}
    if not objs:
        raise ValueError("в паспорте нет поверхностей этой локации (паспорт другой комнаты?)")
    if missing:
        notes.append("паспорт: нет в локации " + ", ".join(sorted(set(missing))[:6]))
    rs = p.get("room_scale")
    if _num(rs) and rs > 0:
        flags = ["room_scale_suspect"] if abs(rs - 1) > cfg["room_scale_tol"] else []
    else:
        rs, flags = None, ["room_scale_unknown"]
    return objs, labels, rs, flags, notes


def split_baseline(cands, cfg):
    """С паспортом запасной «пол» (variant baseline) откладывается, если есть мебельные кандидаты:
    он нужен, только когда мебель не дала тела (иначе лучше wrong_surface, чем fallback)."""
    if not cfg["passport"]["defer_baseline"] or not any(c["variant"] == "support" for c in cands):
        return list(cands), []
    return [c for c in cands if c["variant"] != "baseline"], [c for c in cands if c["variant"] == "baseline"]


# ------------------------------------------------------------------------------------------------ preview
COLORS = [(255, 196, 0), (0, 200, 255), (255, 90, 90), (120, 230, 90), (210, 120, 255), (255, 150, 40),
          (90, 140, 255), (255, 110, 200)]


def projector(plate_info, width, height):
    """Мир плиты -> пиксели картинки (W×H), как pose_fit.Plate.project; без plate.json — None."""
    try:
        import numpy as np
        R = np.asarray(plate_info["R_world_from_cv"], float)
        C = np.array([0.0, 0.0, float(plate_info["cam_height_m"])])
        f = (width / 2) / math.tan(math.radians(float(plate_info["fov_x_deg"])) / 2)
    except (KeyError, TypeError, ValueError):
        return None

    def project(points):
        c = (np.asarray(points, float) - C) @ R
        z = c[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.column_stack([f * c[:, 0] / z + width / 2, f * c[:, 1] / z + height / 2]), z
    return project


def _font(size):
    from PIL import ImageFont
    try:
        return ImageFont.load_default(size=size)
    except TypeError:                                   # Pillow < 10.1
        return ImageFont.load_default()


def render_preview(image, p, plate_info=None):
    """Локация с подписанными объектами: номер, тип, высота опоры (см), стрелка лицевой стороны. PIL RGB."""
    from PIL import ImageDraw
    im = image.convert("RGB").copy()
    W, H = im.size
    draw = ImageDraw.Draw(im)
    font, small = _font(max(14, H // 40)), _font(max(12, H // 52))
    proj = projector(plate_info or {}, W, H)
    fl = float(p.get("floor_level_m") or 0.0)
    occupied = []
    legend = []
    for k, o in enumerate(p.get("objects") or [], 1):
        color = COLORS[(k - 1) % len(COLORS)]
        planes = o.get("support_planes") or []
        main = max(planes, key=lambda pl: float(pl.get("area_m2") or 0.0), default=None)
        h_cm = None if main is None else round(float(main["height_m"]) * 100)
        text = f"{k} {o.get('type')}" + ("" if o.get("type") == "floor" or h_cm is None else f" {h_cm}cm")
        legend.append(text + (" +back" if o.get("backrest") else ""))
        anchor = None
        if proj is not None:
            for pl in planes:
                poly = pl.get("polygon") or []
                if len(poly) < 2:
                    continue
                z = fl + float(pl["height_m"])
                uv, depth = proj([[q[0], q[1], z] for q in poly])
                if (depth <= 0).any():
                    continue
                pts = [tuple(map(float, q)) for q in uv] + [tuple(map(float, uv[0]))]
                draw.line(pts, fill=(0, 0, 0), width=5)
                draw.line(pts, fill=color, width=3 if pl is main else 2)
            if main is not None and main.get("polygon"):
                poly = main["polygon"]
                cx = sum(q[0] for q in poly) / len(poly)
                cy = sum(q[1] for q in poly) / len(poly)
                z = fl + float(main["height_m"])
                fd = o.get("front_dir")
                pts = [[cx, cy, z]] + ([[cx + fd[0] * 0.4, cy + fd[1] * 0.4, z]] if fd else [])
                uv, depth = proj(pts)
                if (depth > 0).all() and all(map(math.isfinite, uv.ravel().tolist())):
                    anchor = (float(uv[0][0]), float(uv[0][1]))
                    if fd:
                        a, b = anchor, (float(uv[1][0]), float(uv[1][1]))
                        draw.line([a, b], fill=(0, 0, 0), width=7)
                        draw.line([a, b], fill=color, width=4)
                        ang = math.atan2(b[1] - a[1], b[0] - a[0])
                        L = 14
                        head = [b, (b[0] - L * math.cos(ang - .5), b[1] - L * math.sin(ang - .5)),
                                (b[0] - L * math.cos(ang + .5), b[1] - L * math.sin(ang + .5))]
                        draw.polygon(head, fill=color, outline=(0, 0, 0))
        if anchor is None:
            anchor = (8, 40 + 26 * k)
        box = draw.textbbox((0, 0), text, font=font)
        w, h = box[2] - box[0] + 10, box[3] - box[1] + 10
        x = int(min(max(anchor[0] - w / 2, 0), max(0, W - w)))
        y = int(min(max(anchor[1] - h - 6, 0), max(0, H - h)))
        for _ in range(40):
            if not any(x < b[2] and x + w > b[0] and y < b[3] and y + h > b[1] for b in occupied):
                break
            y = y + h + 2 if y + 2 * h + 2 <= H else max(0, y - 3 * h)
        occupied.append((x, y, x + w, y + h))
        draw.rectangle(occupied[-1], fill=(0, 0, 0), outline=color, width=2)
        draw.text((x + 5, y + 5 - box[1]), text, fill=color, font=font)
    if legend:                                          # список внизу слева: подписи на картинке могут перекрываться
        lb = [draw.textbbox((0, 0), t, font=small) for t in legend]
        lh = max(b[3] - b[1] for b in lb) + 6
        lw = max(b[2] - b[0] for b in lb) + 12
        y0 = max(0, H - lh * len(legend) - 8)
        draw.rectangle((0, y0, lw, H), fill=(0, 0, 0))
        for k, (t, b) in enumerate(zip(legend, lb)):
            draw.text((6, y0 + 4 + k * lh - b[1]), t, fill=COLORS[k % len(COLORS)], font=small)
    rs = p.get("room_scale")
    head = (f"passport {p.get('id', '')[:16]} | confirmed: {'YES' if p.get('confirmed') else 'NO - check'} | "
            f"room_scale: {rs if rs is not None else 'unknown'} | cam {p.get('camera_height_m')} m | "
            f"floor: {'yes' if p.get('floor_visible') else 'no'}")
    hb = draw.textbbox((0, 0), head, font=small)
    draw.rectangle((0, 0, W, hb[3] - hb[1] + 10), fill=(0, 0, 0))
    draw.text((6, 5 - hb[1]), head, fill=(255, 255, 255) if p.get("confirmed") else (255, 210, 0), font=small)
    return im
