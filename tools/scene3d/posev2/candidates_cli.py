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


def add_support_depth(request, config, result):
    """Добавить глубину центра выбранной опоры в системе камеры (только отчёт нового узла)."""
    report = result.get("report") if isinstance(result, dict) else None
    if not isinstance(report, dict):
        return result
    try:
        import numpy as np
        profile_path = pack_cli.required_file(request.get("location"), "location profile.json")
        profile = pack_cli.read_json(profile_path)
        sid = report.get("target_instance") or report.get("support_object")
        obj = next((o for o in profile.get("objects", []) if o.get("id") == sid), None)
        point = (obj or {}).get("center_m") or (obj or {}).get("seed_world")
        if not (isinstance(point, list) and len(point) >= 3):
            return result
        plate_dir = pack_cli.assets_for(config, pack_cli.location_id(profile.get("loc_id")), profile_path.parent)
        pf, _ = pack_cli.modules(config)
        plate = pf.Plate(plate_dir)
        import manual_pose
        basis = manual_pose.location_basis(plate)[:2, :2]
        if np.isfinite(basis).all():
            report["candidate_offset_basis_xy"] = basis.tolist()
        _, _, depths = plate.project(np.asarray([point[:3]], dtype=float))
        depth = float(depths[0])
        if np.isfinite(depth) and depth > 0:
            report["support_depth_m"] = depth
    except Exception:  # depth is advisory; placement output must remain usable
        return result
    return result


def add_visual_metrics(config, result):
    """Measure the actual saved render; failures stay explicit and do not invent proxy values."""
    report = result.get('report') if isinstance(result, dict) else None
    if not isinstance(report, dict):
        return result
    try:
        import numpy as np
        import manual_pose
        directory = Path(result['silhouette']).parent
        job = directory.parents[1]
        joints = np.load(pack_cli.required_file(job / 'fit/label_joints.npz', 'fitted body joints'))['kp']
        pelvis = (np.asarray(joints[9], dtype=float) + np.asarray(joints[10], dtype=float)) / 2
        pf, _ = pack_cli.modules(config)
        plate = pf.Plate(result['scene_setup']['plate_dir'])
        basis = manual_pose.location_basis(plate)[:2, :2]
        report['candidate_pelvis_xy'] = pelvis[:2].tolist()
        report['candidate_offset_basis_xy'] = basis.tolist()
    except Exception as exc:
        report['placement_metrics_error'] = f'{type(exc).__name__}: {exc}'
    if not result.get('silhouette'):
        report.setdefault('visual_metrics_error', 'silhouette output unavailable')
        return result
    try:
        import candidates
        import threeda_blender
        cfg = candidates.load_config()['visibility']
        directory = Path(result['silhouette']).parent
        scene = pack_cli.required_file(directory / 'scene.blend', 'candidate render scene')
        import numpy as np
        from PIL import Image
        from candidate_visual import measure
        output = directory / 'candidate_visual_metrics.json'
        full = directory / 'candidate_visual_body.png'
        visible = directory / 'candidate_visual_visible.png'
        for path in (output, full, visible):
            path.unlink(missing_ok=True)
        pack_cli.run_stage(threeda_blender.command(config, HERE / 'candidate_visual_blender.py', [
            '--scene', scene, '--out', output, '--expand', cfg['canvas_expansion']]),
            'candidate silhouette passes')
        with Image.open(pack_cli.required_file(full, 'body silhouette')) as image:
            body = np.asarray(image.convert('L'))
        with Image.open(pack_cli.required_file(visible, 'visible silhouette')) as image:
            vis = np.asarray(image.convert('L'))
        n = int(cfg['canvas_expansion'])
        h, w = body.shape
        if n < 3 or n % 2 != 1 or h % n or w % n:
            raise ValueError('invalid expanded silhouette size')
        height, width = h // n, w // n
        x, y = (n - 1) * width // 2, (n - 1) * height // 2
        report['visual_metrics'] = measure(body, vis, (x, y, x + width, y + height), cfg['mask_threshold'])
    except Exception as exc:
        report['visual_metrics_error'] = f'{type(exc).__name__}: {exc}'
    return result


def add_screen_metrics(request, config, result):
    """E1c: измерение «в кадре» (кадр, висит/утоплен по карте глубины) в отчёт кандидата: report['screen'].
    Сбой измерения пишется в screen.measure_error и штрафуется оценкой, а не пропускается."""
    report = result.get('report') if isinstance(result, dict) else None
    if not isinstance(report, dict):
        return result
    try:
        import candidate_screen
        cfg = candidate_screen.load_config()
        silhouette = result.get('silhouette')
        if not silhouette:
            raise ValueError('нет рендера кандидата: каталог результата неизвестен')
        job = Path(silhouette).parent.parents[1]
        pf, _ = pack_cli.modules(config)
        plate = pf.Plate(result['scene_setup']['plate_dir'])
        report['screen'] = candidate_screen.measure_job(job, plate, request.get('pose_class'),
                                                        request.get('support_parts'), cfg)
        report['screen']['job'] = job.name                  # каталог результата в <farm>/pack/jobs — для разбора
    except Exception as exc:  # noqa: BLE001
        report['screen'] = {'measure_error': [f'{type(exc).__name__}: {exc}'], 'auto_flag': 'measure_error',
                            'contact_status': 'unknown', 'frame_ok': None}
    flags = set(report.get('flags') or [])
    for name in str(report['screen'].get('auto_flag') or '').split('+'):
        if name not in ('', 'ok'):
            flags.add('screen_' + name)
    report['flags'] = sorted(flags)
    return result


def geometry_worker(request, config):
    captured = {}
    patch(request, config, captured)
    try:
        result = add_support_depth(request, config, pack_cli.do_place(request, config))
        result = add_visual_metrics(config, result)
        return add_screen_metrics(request, config, result)
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
