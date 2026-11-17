"""E1 (0.3.31): CLI одного кандидата 3DAPlaceCandidates — штатная постановка pack_cli.do_place без изменений,
плюс две вещи только в этом процессе:

1. отчёт place() возвращается и тогда, когда тела нет (no_target: zone_too_small и т. п.) — pack_cli в этом
   случае падает «placement has no body geometry» без отчёта, и узел кандидатов не видит причину;
2. вариант «зона мягче» (request e1_zone_min_frac): no_target по zone_too_small только если маска опоры меньше
   e1_zone_min_frac × след тела (штатно 1.0);
3. пол для авто-профиля без однозначного пола (request e1_floor_id: кусок surf_N, выбранный кандидатами) —
   штатно pack_cli отказывает «auto profile floor is ambiguous»;
4. куски одной мебели — одна опора (request e1_merge_ids: «surf_4,surf_3,surf_2», первый — цель);
5. меньше ячеек зоны на одну опору (request e1_attempts) — кандидаты сами перебирают опоры, лимит времени общий.
Кэш постановок с опциями 2–5 — с отдельным ключом, чтобы 3DAPlacePose с теми же входами не получил постановку,
которую сам бы отверг.

python candidates_cli.py place --request R --output O   (узел; дальше сам запускает geometry-воркер)
"""
import argparse
import hashlib
import json
import sys
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCENE = HERE.parent
sys.path.insert(0, str(SCENE))
import pack_cli  # noqa: E402


def patch(request, config, captured):
    pack_cli.modules(config)
    import manual_pose
    import place
    import reference_place
    frac = request.get("e1_zone_min_frac")
    for mod in (place, reference_place):
        original = mod.place

        def wrapped(*args, _original=original, **kwargs):
            result = _original(*args, **kwargs)
            captured["status"] = result.get("status")
            captured["report"] = result.get("rep")
            return result
        mod.place = wrapped
    salt = []
    if frac is not None:
        frac = float(frac)
        if not 0 < frac <= 1:
            raise ValueError("e1_zone_min_frac must be in (0, 1]")

        def zone_too_small(diagnostics):
            area, body = diagnostics["zone_mask_area_m2"], diagnostics["body_footprint_area_m2"]
            return area is not None and (area == 0 or (body is not None and area < frac * body))
        place.zone_too_small = zone_too_small
        salt.append(f"e1_zone_min_frac={frac}")
    floor_id = request.get("e1_floor_id")
    if floor_id:
        # пол авто-профиля не определён однозначно (floor_status != ok) — кандидаты выбрали кусок на уровне пола;
        # в этом процессе он становится единственным полом профиля
        import profiles
        init = profiles.LocationProfile.__init__

        def __init__(self, path, _init=init):
            _init(self, path)
            obj = next((o for o in self.data["objects"] if o.get("id") == floor_id and "top_grid_npz" in o), None)
            if obj is None:
                raise ValueError(f"e1_floor_id {floor_id!r}: нет такой поверхности в профиле")
            for o in self.data["objects"]:
                if o.get("type") == "floor" and o is not obj:
                    o["type"] = "other"
            obj["type"] = obj["name"] = "floor"
            self.data.setdefault("auto_surfaces", {})["floor_status"] = "ok"
        profiles.LocationProfile.__init__ = __init__
        salt.append(f"e1_floor_id={floor_id}")
    attempts = request.get("e1_attempts")
    if attempts:
        place.ATTEMPTS = int(attempts)              # ячеек зоны на опору (штатно place.attempts из YAML)
        salt.append(f"e1_attempts={int(attempts)}")
    merge = [x for x in str(request.get("e1_merge_ids") or "").split(",") if x]
    if len(merge) > 1:
        merge_support(merge, place)
        salt.append("e1_merge_ids=" + ",".join(merge))
    if salt:
        key_fn = manual_pose.placement_key

        def placement_key(*args, **kwargs):
            return hashlib.sha256((key_fn(*args, **kwargs) + ":" + ",".join(salt)).encode()).hexdigest()
        manual_pose.placement_key = placement_key


def merge_support(ids, place):
    """Куски одной мебели (кровать/диван, разрезанные корзинами высоты 2 см) — одна опора ids[0]: маски на общей
    сетке объединяются, остальные куски убираются из профиля, и match_auto_support не перевыбирает один кусок."""
    import numpy as np
    from dataclasses import replace
    from scipy import ndimage
    import profiles
    import top_region as tr
    target, rest = ids[0], set(ids[1:])
    init, region, match = profiles.LocationProfile.__init__, profiles.LocationProfile.region, place.match_auto_support

    def __init__(self, path, _init=init):
        _init(self, path)
        found = {o["id"]: o for o in self.data["objects"] if o.get("id") in ids and "top_grid_npz" in o}
        if set(found) != set(ids):
            raise ValueError("e1_merge_ids: нет поверхностей " + ",".join(sorted(set(ids) - set(found))))
        self._e1_merge = [found[i] for i in ids]
        self.data["objects"] = [o for o in self.data["objects"] if o.get("id") not in rest]

    def merged_region(self, obj, _region=region):
        if obj.get("id") != target or not getattr(self, "_e1_merge", None):
            return _region(self, obj)
        regs = [_region(self, o) for o in self._e1_merge]
        if any(r.mask.shape != regs[0].mask.shape or not np.allclose(r.origin, regs[0].origin) for r in regs):
            raise ValueError("e1_merge_ids: куски на разных сетках")
        mask = np.logical_or.reduce([r.mask for r in regs])
        height = np.full(mask.shape, np.nan)
        for r in regs:
            height = np.where(r.mask & np.isnan(height), r.height, height)
        dt = ndimage.distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1] * tr.CELL
        out = replace(regs[0], mask=mask, height=height, dt=dt, h0=float(np.nanmedian(height[mask])),
                      object_mask=mask.copy(), top_diagnostics=dict(regs[0].top_diagnostics, e1_merged=list(ids)))
        out.object_id = target
        return out

    def match_auto_support(plate, legacy, H, _match=match):
        if getattr(legacy, "object_id", None) == target:
            return None, {"surface_match": "e1_merged", "merged_ids": list(ids)}
        return _match(plate, legacy, H)
    profiles.LocationProfile.__init__ = __init__
    profiles.LocationProfile.region = merged_region
    place.match_auto_support = match_auto_support


def summary(report):
    """Короткий отчёт для узла (no_target: причина и площади зоны/следа)."""
    if not isinstance(report, dict):
        return None
    keep = ("status", "flags", "zone_diagnostics", "zone", "support_info", "surface_match", "shot_spec")
    out = {k: report[k] for k in keep if k in report}
    zd = out.get("zone_diagnostics")
    if isinstance(zd, dict):
        out["zone_diagnostics"] = {k: zd.get(k) for k in ("zone_mask_area_m2", "body_footprint_area_m2",
                                                           "support_vertex_count", "zero_support_cause")}
    return json.loads(json.dumps(out, default=str))


def geometry_worker(request, config):
    captured = {}
    patch(request, config, captured)
    try:
        return pack_cli.do_place(request, config)
    except Exception as exc:
        result = pack_cli.error_result(exc)
        if captured.get("report") is not None:
            result["report"] = summary(captured["report"])
            result["place_status"] = captured.get("status")
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("place",))
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--geometry-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        request = pack_cli.read_json(args.request)
        config = pack_cli.config_for(request)
        if args.geometry_worker:
            result = geometry_worker(request, config)
        else:
            # Как pack_cli: геометрия только в geometry_python, не в окружении ComfyUI.
            out = Path(args.output).resolve().with_name(".cand-" + uuid.uuid4().hex + ".json")
            try:
                pack_cli.run_stage([*pack_cli.worker_python(config, config["geometry_python"]), Path(__file__).resolve(),
                                    args.command, "--request", Path(args.request).resolve(), "--output", out,
                                    "--geometry-worker"], "candidate geometry worker")
                result = pack_cli.read_json(pack_cli.required_file(out, "worker result"))
            finally:
                out.unlink(missing_ok=True)
    except (Exception, SystemExit) as exc:
        result = pack_cli.error_result(exc)
    pack_cli.save_json(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
