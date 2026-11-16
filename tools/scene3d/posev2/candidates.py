"""E1 (0.3.31): кандидаты постановки — опора × место × разворот × рост — и их оценка.

Только чистая логика (без CLI, без GPU): узел 3DAPlaceCandidates строит план, по очереди зовёт штатную
постановку 3DAPlacePose (invoke('place', ...)) и оценивает её отчёт этим модулем. place.py не меняется.

Опоры авто-локации (3DALocation, auto_surfaces.py) — наблюдаемые горизонтальные куски без семантики
(id surf_N, тип other). Поэтому тип мебели здесь — геометрическая метка по высоте и площади группы соседних
кусков (classify): floor | bed | sofa | bench | stool | table | nightstand | shelf. Метка — догадка по геометрии,
не распознавание; она пишется в candidates_json вместе с признаками, чтобы её можно было проверить.
Все числа — config/placement.yaml, раздел candidates.
"""
import json
import math
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "placement.yaml"
STATUS_OK = ("ok", "partial")


def load_config(path=None):
    """Раздел candidates + размер ячейки сетки top_region (для масок поверхностей)."""
    data = yaml.safe_load(Path(path or CONFIG_PATH).read_text(encoding="utf-8"))
    cfg = dict(data["candidates"])
    cfg["cell_m"] = float(data["top_region"]["cell"])
    return cfg


# ------------------------------------------------------------------------------------------------ входы
def parse_objects(objects):
    """objects 3DALocation (JSON-строка или список) -> список dict; не JSON — пустой список."""
    if isinstance(objects, str):
        try:
            objects = json.loads(objects)
        except ValueError:
            return []
    return [dict(o) for o in objects if isinstance(o, dict) and o.get("id")] if isinstance(objects, list) else []


def parse_hint(hint_json):
    """{"support":"sofa","facing":"camera","near_wall":true,"pose_class":"sit_edge"}; пусто/ошибка -> {}."""
    if isinstance(hint_json, dict):
        return dict(hint_json)
    if not isinstance(hint_json, str) or not hint_json.strip():
        return {}
    try:
        value = json.loads(hint_json)
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


def enrich(objects, profile_data=None, profile_dir=None, cfg=None):
    """Дополнить objects полями профиля (h0, area_m2, center_m, seed_world) и xy ячеек маски (если читается)."""
    by_id = {o.get("id"): o for o in (profile_data or {}).get("objects", []) if isinstance(o, dict)}
    out = []
    for obj in objects:
        o = dict(obj)
        src = by_id.get(o["id"], {})
        for key in ("h0", "area_m2", "center_m", "seed_world", "name", "top_grid_npz", "flags"):
            if key in src and key not in o:
                o[key] = src[key]
        if o.get("h0") is None and o.get("height_m") is not None:
            o["h0"] = o["height_m"]
        if profile_dir is not None and o.get("top_grid_npz") and cfg is not None and "cells_xy" not in o:
            o["cells_xy"] = mask_cells(Path(profile_dir) / o["top_grid_npz"], cfg)
        out.append(o)
    return out


def mask_cells(path, cfg):
    """xy центров ячеек маски top_grid_npz (подвыборка до cfg.max_cells); не читается — None."""
    try:
        import numpy as np
        with np.load(path, allow_pickle=False) as d:
            origin, mask = np.asarray(d["origin"], float), np.asarray(d["mask"], bool)
        ij = np.argwhere(mask)
        if not len(ij):
            return None
        step = max(1, len(ij) // int(cfg["max_cells"]))
        return ((ij[::step] + .5) * cfg["cell_m"] + origin).tolist()
    except Exception:
        return None


# ------------------------------------------------------------------------------------------------ геометрия опор
def _h(o):
    v = o.get("h0", o.get("height_m"))
    return float(v) if isinstance(v, (int, float)) and math.isfinite(v) else None


def floor_level(objects, cfg):
    """Высота пола: floor_1 (тип floor), если он сам на уровне пола (|h| ≤ floor_band_m), иначе 0."""
    for o in objects:
        h = _h(o)
        if o.get("type") == "floor" and h is not None and abs(h) <= cfg["floor_band_m"]:
            return h
    return 0.0


def _cells(o):
    c = o.get("cells_xy")
    if c:
        return c
    xy = o.get("center_m") or o.get("seed_world")
    return [xy[:2]] if xy else None


def _min_dist(a, b):
    if not a or not b:
        return None
    try:
        import numpy as np
        A, B = np.asarray(a, float)[:, :2], np.asarray(b, float)[:, :2]
        best = math.inf
        for k in range(0, len(A), 512):             # блоки, чтобы не строить огромную матрицу
            d = np.sqrt(((A[k:k + 512, None, :] - B[None, :, :]) ** 2).sum(-1)).min()
            best = min(best, float(d))
        return best
    except ImportError:
        return min(math.dist(p[:2], q[:2]) for p in a for q in b)


def _extent(cells, cfg):
    """(длина, ширина) следа группы по главным осям, перцентили cfg.extent_percentiles."""
    if not cells or len(cells) < 3:
        return None
    import numpy as np
    P = np.asarray(cells, float)[:, :2]
    P = P - P.mean(0)
    _, vec = np.linalg.eigh(P.T @ P)
    lo, hi = cfg["extent_percentiles"]
    ext = sorted((float(np.percentile(P @ v, hi) - np.percentile(P @ v, lo)) + cfg["cell_m"] for v in vec.T), reverse=True)
    return ext[0], ext[1]


def classify(objects, cfg):
    """id -> {label, group, features}. Метки — по геометрии (см. docstring модуля), правила — cfg.geometry."""
    g = cfg["geometry"]
    fl = floor_level(objects, cfg)
    surfaces = [o for o in objects if o.get("type") != "wall" and _h(o) is not None]
    out = {}
    raised = []
    for o in surfaces:
        rel = _h(o) - fl
        if abs(rel) <= cfg["floor_band_m"]:
            out[o["id"]] = {"label": "floor", "group": "floor", "features": {"h_rel_m": round(rel, 3),
                                                                            "area_m2": o.get("area_m2")}}
        elif rel < 0:
            out[o["id"]] = {"label": "below_floor", "group": None, "features": {"h_rel_m": round(rel, 3)}}
        else:
            raised.append((o, rel))
    # группы: куски с близкими по xy ячейками и разницей высот ≤ group_dh_m (одна мебель, разные корзины высоты)
    n = len(raised)
    parent = list(range(n))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(n):
        for j in range(i + 1, n):
            if abs(raised[i][1] - raised[j][1]) > g["group_dh_m"]:
                continue
            d = _min_dist(_cells(raised[i][0]), _cells(raised[j][0]))
            if d is not None and d <= g["group_link_m"]:
                parent[root(i)] = root(j)
    groups, backs = {}, set()
    for i in range(n):
        groups.setdefault(root(i), []).append(i)
    for gid, members in groups.items():
        objs = [raised[i] for i in members]
        cells = [c for o, _ in objs for c in (_cells(o) or [])]
        area = sum(float(o.get("area_m2") or 0.0) for o, _ in objs)
        ext = _extent(cells, cfg) if len(cells) >= 3 else None
        top = max(r for _, r in objs)
        seat = [(o, r) for o, r in objs if g["seat_h"][0] <= r <= g["seat_h"][1]]
        seat_h = max(seat, key=lambda t: float(t[0].get("area_m2") or 0.0))[1] if seat else None
        # спинка: отдельный более высокий кусок рядом с сиденьем (диван/кресло)
        back = None
        if seat_h is not None:
            seat_cells = [c for o, r in seat for c in (_cells(o) or [])]
            for o, r in surfaces_rel(surfaces, fl):
                if g["back_dh_m"][0] <= r - seat_h <= g["back_dh_m"][1]:
                    d = _min_dist(seat_cells, _cells(o))
                    if d is not None and d <= g["back_link_m"]:
                        back = o["id"]
                        break
        L, W = ext if ext else (None, None)
        # маленький верх на высоте сиденья или стола — тумба (табурет по геометрии от неё не отличить)
        if area < g["small_top_max_area_m2"] and top <= g["table_h"][1]:
            label = "nightstand"
        elif seat_h is not None and L is not None and L >= g["bed_min_l_m"] and W >= g["bed_min_w_m"]:
            label = "bed"
        elif seat_h is not None and back is not None:
            label = "sofa"
        elif seat_h is not None:
            label = "bench"
        elif top <= g["table_h"][1]:
            label = "table"
        else:
            label = "shelf"
        if label == "sofa":
            backs.add(back)
        feats = {"group_ids": [o["id"] for o, _ in objs], "area_m2": round(area, 3),
                 "extent_m": [round(L, 2), round(W, 2)] if L is not None else None,
                 "seat_h_m": None if seat_h is None else round(seat_h, 3), "top_h_m": round(top, 3),
                 "backrest_id": back}
        for o, r in objs:
            out[o["id"]] = {"label": label, "group": f"g{gid}", "features": dict(feats, h_rel_m=round(r, 3))}
    for oid in backs:                               # верх спинки дивана — не опора (если он не сам сиденье)
        if oid in out and out[oid]["label"] not in ("sofa", "bed", "bench"):
            out[oid]["label"] = "backrest"
    return out


def surfaces_rel(surfaces, fl):
    return [(o, _h(o) - fl) for o in surfaces]


def semantic_label(obj, labels):
    """Имя объекта, если оно смысловое (старые профили: sofa_1/type sofa), иначе геометрическая метка."""
    for key in ("type", "name"):
        v = obj.get(key)
        if isinstance(v, str) and v not in ("other", "floor", "wall", "") and not v.startswith("surf"):
            return v
    return labels.get(obj["id"], {}).get("label", "other")


def words(label, cfg):
    return [label] + list(cfg["synonyms"].get(label, []))


def support_rank(pose_class, label, cfg):
    """(ярус, номер) опоры для класса позы; None — не годится. Ярус 0 — имя в списке класса,
    ярус 1 — подходит только синоним (например table → massage_table), имя в списке «не годятся» — None."""
    rule = cfg["support_by_class"].get(pose_class)
    if rule is None:
        return None
    allowed, deny = rule.get("allowed", []), rule.get("deny", [])
    if label in deny:
        return None
    if label in allowed:
        return 0, allowed.index(label)
    idx = [allowed.index(w) for w in cfg["synonyms"].get(label, []) if w in allowed]
    return (1, min(idx)) if idx else None


def matches_word(label, word, cfg):
    w = str(word).strip().lower()
    return bool(w) and (w == label or w in cfg["synonyms"].get(label, []))


def supports_for(pose_class, objects, labels, hint, cfg):
    """Подходящие опоры по порядку: подсказка → ярус/номер класса → площадь (больше — раньше) → id."""
    found = []
    for o in objects:
        if o.get("type") == "wall" or _h(o) is None:
            continue
        label = semantic_label(o, labels)
        rank = support_rank(pose_class, label, cfg)
        if rank is None:
            continue
        hinted = bool(hint.get("support")) and (matches_word(label, hint["support"], cfg) or
                                                 str(hint["support"]).lower() == o["id"])
        found.append({"id": o["id"], "type": o.get("type", "other"), "label": label, "rank": list(rank),
                      "hinted": hinted, "area_m2": o.get("area_m2")})
    found.sort(key=lambda s: (not s["hinted"], s["rank"], -(s["area_m2"] or 0.0), s["id"]))
    # одна опора на группу мебели (куски одной кровати — один вариант места)
    seen, out = set(), []
    for s in found:
        grp = labels.get(s["id"], {}).get("group")
        key = grp if grp and grp != "floor" else s["id"]
        if s["label"] == "floor":
            key = "floor"
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


# ------------------------------------------------------------------------------------------------ масштаб комнаты
def room_scale(objects, labels, cfg):
    """room_scale = медиана(типовой/измеренный) по объектам со смысловыми именами и типовыми размерами.
    Геометрические метки не используются (метка сама выведена из высоты — круговое рассуждение)."""
    sizes = cfg["typical_sizes"]
    fl = floor_level(objects, cfg)
    ratios = []
    for o in objects:
        name = semantic_label(o, {}) if not cfg["scale_from_geometry_labels"] else semantic_label(o, labels)
        if name not in sizes:
            continue
        h = _h(o)
        if o.get("type") == "wall" or h is None or h - fl <= 0:
            continue
        lo, hi = sizes[name]
        ratios.append((lo + hi) / 2 / (h - fl))
    if not ratios:
        return None, ["room_scale_unknown"]
    ratios.sort()
    k = len(ratios)
    med = ratios[k // 2] if k % 2 else (ratios[k // 2 - 1] + ratios[k // 2]) / 2
    flags = ["room_scale_suspect"] if abs(med - 1) > cfg["room_scale_tol"] else []
    return round(med, 3), flags


# ------------------------------------------------------------------------------------------------ план
def plan(pose_class, objects, labels, hint, n, cfg, height_m, secondary_contacts=""):
    """Начальный список кандидатов (до n). Производные варианты (разворот +180, рост/масштаб) добавляет
    derived() по отчётам прогонов."""
    def cand(variant, support, **kw):
        c = {"variant": variant, "target_object": support["type"] if support else "floor",
             "instance_hint": support["id"] if support else "", "support_id": support["id"] if support else None,
             "label": support["label"] if support else "floor", "secondary_contacts": secondary_contacts,
             "body_dx": 0.0, "body_dy": 0.0, "body_dyaw": 0.0, "height_m": float(height_m)}
        c.update(kw)
        return c
    out = []
    if pose_class == "auto":
        # 3DA не определяет класс сам (reference_place: pose_class_inferred=False): один штатный прогон.
        return [cand("auto", None, target_object="floor", instance_hint="", label="auto")]
    sups = supports_for(pose_class, objects, labels, hint, cfg)
    if pose_class == "stand":
        floor = next((s for s in sups if s["type"] == "floor"), None)
        base = floor or (sups[0] if sups else None)
        if base is not None and base["type"] == "floor":
            out.append(cand("floor", base, target_object="floor", instance_hint=""))
        elif base is not None:
            out.append(cand("floor", base))
        else:
            out.append(cand("floor", None))
        if hint.get("near_wall") and out:
            wall = ",".join(p + ":wall" for p in cfg["wall_parts"])
            sec = ",".join(x for x in (secondary_contacts, wall) if x)
            out.append(dict(out[0], variant="wall", secondary_contacts=sec))
        for yaw in cfg["stand_yaws_deg"]:
            if len(out) >= n:
                break
            out.append(dict(out[0], variant=f"yaw{yaw:+g}", body_dyaw=float(yaw)))
        return out[:n]
    has_floor = any(s["label"] == "floor" for s in sups)
    room = n - (0 if has_floor or n < 2 else 1)          # последнее место — «как сейчас» (пол), если пола нет в списке
    for s in sups:
        if len(out) >= room:
            break
        if s["type"] == "floor" and s["id"] == "floor_1":
            out.append(cand("support", s, target_object="floor", instance_hint=""))
        else:
            out.append(cand("support", s))
    if not any(c["label"] == "floor" for c in out) and len(out) < n:
        # запасной вариант — штатная постановка 3DA на пол (как базовый замер): лучше wrong_surface, чем fallback
        out.append(cand("baseline", None, target_object="floor", instance_hint="", label="floor"))
    return out


def derived(cands, results, cfg, n_max, scale_flags, scale, best_cand=None):
    """Варианты по отчётам: разворот +180 у мебели, если отчёт видит сторону/спинку не так (в пределах n_max);
    при room_scale_suspect — сверх n_max ещё лучший кандидат с height_m / room_scale."""
    extra = []
    for c, r in zip(cands, results):
        rep = (r or {}).get("report") or {}
        if c["variant"] != "support" or c["label"] == "floor":
            continue
        flags = set(rep.get("flags") or [])
        back = rep.get("back_gap_cm")
        if flags & set(cfg["flip_flags"]) or (isinstance(back, (int, float)) and back > cfg["back_gap_flip_cm"]):
            extra.append(dict(c, variant="yaw+180", body_dyaw=180.0, origin="derived"))
    out = extra[:max(0, n_max - len(cands))]
    if "room_scale_suspect" in scale_flags and scale and cands:
        best = best_cand or cands[0]
        out.append(dict(best, variant="room_scale", height_m=round(best["height_m"] / scale, 3), origin="derived"))
    return out


def retries(cand, place_status, flags, cfg):
    """no_target из-за zone_too_small (маска опоры меньше следа тела) — тот же вариант с мягкой зоной
    (candidates_cli: e1_zone_min_frac = cfg.zone_relax_frac); один раз, только для мебели."""
    frac = cfg.get("zone_relax_frac")
    if (not frac or place_status != "no_target" or "zone_too_small" not in (flags or []) or cand.get("zone_min_frac")
            or cand["label"] == "floor"):
        return []
    return [dict(cand, variant=cand["variant"] + "+zone", zone_min_frac=float(frac), origin="retry")]


# ------------------------------------------------------------------------------------------------ оценка
def report_of(status_text):
    """Вторая строка статуса 3DAPlacePose -> (status, report); fallback/ошибка -> ('fallback', {})."""
    if not isinstance(status_text, str):
        return "fallback", {}
    first, _, rest = status_text.partition("\n")
    try:
        data = json.loads(rest)
        rep = data.get("report", {}) if isinstance(data, dict) else {}
    except ValueError:
        rep = {}
    st = rep.get("status") if isinstance(rep.get("status"), str) else None
    if st is None:
        st = "fallback" if "fallback" in first else first.split(" ", 1)[0].strip() or "fallback"
    return st, rep


def _wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def score(cand, status, rep, pose_class, cfg):
    """Итог 0..1 = Σ вес·составляющая (веса cfg.weights, сумма весов нормируется)."""
    w, s = cfg["weights"], cfg["score"]
    comp = {"status": float(s["status"].get(status, 0.0))}
    gap = rep.get("support_gap_cm")
    comp["gap"] = math.exp(-abs(gap) / s["gap_scale_cm"]) if isinstance(gap, (int, float)) else 0.0
    pen = rep.get("max_penetration_cm")
    comp["penetration"] = math.exp(-max(pen, 0.0) / s["pen_scale_cm"]) if isinstance(pen, (int, float)) else 0.0
    # совпадение с референсом — то, что 3DA уже считает: отклонение наклона/крена и ракурса к камере
    dev = None
    total = (rep.get("orient") or {}).get("total_change_deg")
    ang = rep.get("angle_to_camera_deg") or {}
    if isinstance(total, (int, float)):
        dev = float(total)
        if isinstance(ang.get("ref"), (int, float)) and isinstance(ang.get("plate"), (int, float)):
            dev += abs(_wrap(ang["plate"] - ang["ref"]))
    elif isinstance(rep.get("reference_pose_error_m"), (int, float)):
        dev = rep["reference_pose_error_m"] * s["ref_error_deg_per_m"]
    comp["pose_match"] = math.exp(-dev / s["pose_scale_deg"]) if dev is not None else s["pose_unknown"]
    rank = support_rank(pose_class, cand.get("label", "floor"), cfg) if pose_class != "auto" else None
    if pose_class == "auto":
        comp["support_class"] = s["support_unknown"]
    elif rank is None:
        comp["support_class"] = 0.0
    else:
        comp["support_class"] = max(0.0, (1.0 if rank[0] == 0 else s["support_tier1"]) - rank[1] * s["support_step"])
    frame = rep.get("frame") or {}
    comp["visibility"] = 1.0 if frame.get("ok") else (s["partly_visible"] if frame.get("visible") else 0.0)
    total_w = sum(w.values())
    # без постановки (fallback, no_target) итог 0: верная метка опоры не делает пустой вариант лучше реального
    value = sum(w[k] * comp[k] for k in w) / total_w if total_w and comp["status"] > 0 else 0.0
    return round(value, 4), {k: round(v, 3) for k, v in comp.items()}


def actual_support(rep, cand, labels, objects):
    """id:метка фактической опоры по отчёту (support_object); не на известной поверхности — 'other'."""
    sid = rep.get("support_object")
    if not isinstance(sid, str) or sid in ("other", "none", "unresolved", ""):
        return "other"
    obj = next((o for o in objects if o.get("id") == sid), None)
    if obj is None:
        return sid
    return f"{sid}:{semantic_label(obj, labels)}"


def choose(entries, pick):
    """pick ≥ 0 и есть такой номер — он; иначе лучший по оценке (при равенстве — меньший номер)."""
    done = [e for e in entries if e.get("score") is not None]
    if not done:
        return None
    if isinstance(pick, int) and pick >= 0:
        hit = next((e for e in done if e["index"] == pick), None)
        if hit is not None:
            return hit
    return max(done, key=lambda e: (e["score"], -e["index"]))
