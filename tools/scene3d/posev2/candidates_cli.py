"""E1 (0.3.31): CLI одного кандидата 3DAPlaceCandidates — штатная постановка pack_cli.do_place без изменений,
плюс две вещи только в этом процессе:

1. отчёт place() возвращается и тогда, когда тела нет (no_target: zone_too_small и т. п.) — pack_cli в этом
   случае падает «placement has no body geometry» без отчёта, и узел кандидатов не видит причину;
2. вариант «зона мягче» (request e1_zone_min_frac): no_target по zone_too_small только если маска опоры меньше
   e1_zone_min_frac × след тела (штатно 1.0). Кэш таких постановок — с отдельным ключом, чтобы 3DAPlacePose
   с теми же входами не получил постановку, которую сам бы отверг.

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
    if frac is not None:
        frac = float(frac)
        if not 0 < frac <= 1:
            raise ValueError("e1_zone_min_frac must be in (0, 1]")

        def zone_too_small(diagnostics):
            area, body = diagnostics["zone_mask_area_m2"], diagnostics["body_footprint_area_m2"]
            return area is not None and (area == 0 or (body is not None and area < frac * body))
        place.zone_too_small = zone_too_small
        key_fn = manual_pose.placement_key

        def placement_key(*args, **kwargs):
            return hashlib.sha256((key_fn(*args, **kwargs) + f":e1_zone_min_frac={frac}").encode()).hexdigest()
        manual_pose.placement_key = placement_key


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
