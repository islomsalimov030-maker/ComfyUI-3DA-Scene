"""E16: персонаж «3D-контроля» — тело (MHR) и манекен MPFB из данных персонажа.

Стейси — прежний путь: `config.mannequin` и `body_json` узла. Другой персонаж (`character` в запросе, из ручки
«персонаж» 3D-контроля) подменяет и то и другое файлами из корня данных:
  characters/<id>/body.json        телосложение (MHR shape/scale, рост, body_dims, цвета прокси)
  characters/<id>/mannequin.blend  манекен MPFB этого персонажа (рост = height_m, без волос и одежды)
Рост постановки: body.json `placement_height_m` (рост, который отдаётся постановке, чтобы меш MHR получился ростом
height_m: масштаб постановки считается по длинам сегментов, а не по росту меша), иначе `height_m`.
"""
import json
import math
from pathlib import Path
import re

DEFAULT = "stacy"
ID = re.compile(r"[a-z][a-z0-9_]{0,31}")


def paths(config, character):
    """Файлы персонажа в корне данных (без проверки существования)."""
    if not isinstance(character, str) or not ID.fullmatch(character):
        raise ValueError("character: нужен короткий латинский идентификатор")
    root = Path(config["farm_root"]) / "characters" / character
    return root / "body.json", root / "mannequin.blend"


def resolve(request, config):
    """None — прежний путь (Стейси). Иначе dict(character, body_json, mannequin, height_m, true_height_m, body)."""
    character = request.get("character") or DEFAULT
    if character == DEFAULT:
        return None
    body_path, mannequin = paths(config, character)
    for path, label in ((body_path, "профиль тела body.json"), (mannequin, "манекен mannequin.blend")):
        if not path.is_file():
            raise ValueError(f"персонаж {character}: нет файла «{label}» ({path}); загрузите данные персонажа "
                             f"characters/{character}/ или выберите другого персонажа")
    body = json.loads(body_path.read_text(encoding="utf-8"))
    dims = body.get("body_dims")
    if not isinstance(dims, dict):
        raise ValueError(f"персонаж {character}: в body.json нет body_dims")
    true_height = body.get("height_m")
    height = body.get("placement_height_m", true_height)
    for label, value in (("height_m", true_height), ("placement_height_m", height)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not .5 < value < 3.5:
            raise ValueError(f"персонаж {character}: body.json {label} должен быть числом 0,5–3,5 м")
    return dict(character=character, body_json=str(body_path), mannequin=str(mannequin), height_m=float(height),
                true_height_m=float(true_height), body=body)
