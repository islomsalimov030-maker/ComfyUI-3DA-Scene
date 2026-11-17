"""E5: разбор ответа LLM о позе в подсказку постановки hint_json для 3DAPlaceCandidates.

Подсказка: {"pose_class", "support", "facing", "near_wall", "prop"}. JSON берётся из ответа модели
(в т.ч. из ```-блока), поля проверяются по допустимым значениям. Поле, которого нет или оно неверное, —
из запасного разбора по ключевым словам исходного промпта; тогда ok=False и причина в note.
Все слова и таблицы — config/placement.yaml, раздел parse_hint. Только чистая логика, без GPU.
"""
import json
import re
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parents[1] / 'config' / 'placement.yaml'
FIELDS = ('pose_class', 'support', 'facing', 'near_wall', 'prop')
REQUIRED = ('pose_class', 'support', 'facing', 'near_wall')


def load_config(path=None):
    return yaml.safe_load(Path(path or CONFIG_PATH).read_text(encoding='utf-8'))['parse_hint']


# ------------------------------------------------------------------------------------------------ JSON из ответа
def _objects(text):
    """Все JSON-объекты верхнего уровня в тексте (по парным скобкам вне строк), по порядку."""
    out, depth, start, in_str, esc = [], 0, None, False, False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == '\\':
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"' and depth:
            in_str = True
        elif ch == '{':
            if not depth:
                start = i
            depth += 1
        elif ch == '}' and depth:
            depth -= 1
            if not depth:
                out.append(text[start:i + 1])
    return out


def extract_json(text):
    """Первый JSON-объект (dict) из ответа модели: ```json-блок или голый объект; нет — (None, причина)."""
    text = str(text or '').strip()
    if not text:
        return None, 'пустой ответ модели'
    blocks = re.findall(r'```(?:json|JSON)?\s*(.*?)```', text, re.S)
    for chunk in blocks + [text]:
        for candidate in _objects(chunk):
            for variant in (candidate, re.sub(r',\s*([}\]])', r'\1', candidate)):   # хвостовая запятая
                try:
                    value = json.loads(variant)
                except ValueError:
                    continue
                if isinstance(value, dict):
                    return value, ''
    return None, 'в ответе нет JSON-объекта' if '{' not in text else 'JSON в ответе не читается'


# ------------------------------------------------------------------------------------------------ проверка полей
def _key(value):
    return re.sub(r'[\s\-]+', '_', str(value).strip().lower())


def _words(value, limit):
    words = re.findall(r'[a-z0-9_]+', str(value).lower())
    return ' '.join(words[:limit])


def norm_support(value, cfg):
    """Опора → слово из support_words (длинные фразы раньше; «_» = пробел); нет в словаре — первые слова."""
    text = ' ' + re.sub(r'[_\-]+', ' ', str(value).lower()) + ' '
    for phrase in sorted(cfg['support_words'], key=len, reverse=True):
        if re.search(r'(?<![a-z])' + re.escape(phrase.replace('_', ' ')) + r'(?![a-z])', text):
            return cfg['support_words'][phrase]
    return _words(value, cfg['support_max_words'])


def check_field(name, value, cfg):
    """(значение, '') или (None, причина)."""
    if name == 'pose_class':
        if not isinstance(value, str) or not value.strip():
            return None, 'pose_class пуст'
        key = _key(value)
        key = cfg['aliases']['pose_class'].get(key, key)
        return (key, '') if key in cfg['pose_classes'] else (None, f'pose_class «{value}» не из списка 3DA')
    if name == 'facing':
        if not isinstance(value, str) or not value.strip():
            return None, 'facing пуст'
        key = _key(value)
        key = cfg['aliases']['facing'].get(key, key)
        return (key, '') if key in cfg['facing_values'] else (None, f'facing «{value}» не из {"/".join(cfg["facing_values"])}')
    if name == 'near_wall':
        if isinstance(value, bool):
            return value, ''
        if isinstance(value, str) and value.strip().lower() in ('true', 'yes', '1', 'false', 'no', '0'):
            return value.strip().lower() in ('true', 'yes', '1'), ''
        if isinstance(value, (int, float)) and value in (0, 1):
            return bool(value), ''
        return None, f'near_wall «{value}» не true/false'
    if name == 'support':
        if not isinstance(value, str) or not norm_support(value, cfg):
            return None, 'support пуст'
        return norm_support(value, cfg), ''
    if name == 'prop':
        if value is None or (isinstance(value, str) and value.strip().lower() in ('', 'none', 'null', 'nothing', 'no')):
            return '', ''
        if not isinstance(value, str):
            return None, 'prop не строка'
        return _words(value, cfg['prop_max_words']), ''
    raise KeyError(name)


# ------------------------------------------------------------------------------------------------ запасной разбор
def _rule(rules, text):
    for rule in rules:
        if any(re.search(p, text) for p in rule['any']):
            return rule['value']
    return None


def keyword_hint(prompt, cfg):
    """Подсказка по ключевым словам исходного промпта (таблица parse_hint.keywords); пусто — значения default."""
    text = ' '.join(str(prompt or '').lower().split())
    kw, d = cfg['keywords'], cfg['default']
    pose = _rule(kw['pose_class'], text) or d['pose_class']
    support = ''
    positions = []
    for phrase, word in cfg['support_words'].items():
        m = re.search(r'(?<![a-z])' + re.escape(phrase.replace('_', ' ')) + r'(?![a-z])', text)
        if m:
            positions.append((m.start(), -len(phrase), word))
    if positions:
        support = min(positions)[2]                       # первая упомянутая опора (длинная фраза — при равенстве)
    if not support or pose in cfg['floor_classes'] and support not in cfg['floor_words']:
        support = d['support']                            # стоя/на коленях/на корточках — пол, если не коврик
    facing = _rule(kw['facing'], text) or d['facing']
    near_wall = _rule(kw['near_wall'], text)
    prop = ''
    for pattern in kw['prop']:
        m = re.search(pattern, text)
        if m:
            prop = _words(m.group(1), cfg['prop_max_words'])
            break
    return {'pose_class': pose, 'support': support, 'facing': facing,
            'near_wall': bool(near_wall) if near_wall is not None else bool(d['near_wall']), 'prop': prop}


# ------------------------------------------------------------------------------------------------ узел
def parse_hint(text, prompt='', cfg=None):
    """(hint dict, ok, note). ok — ответ модели дал все обязательные поля верно; иначе недостающее — по словам промпта."""
    cfg = cfg or load_config()
    value, why = extract_json(text)
    problems, hint, fallback = [], {}, None
    if value is None:
        problems.append(why)
        value = {}
    lowered = {str(k).strip().lower(): v for k, v in value.items()}
    for name in FIELDS:
        if name not in lowered:
            if name in REQUIRED:
                problems.append(f'нет поля {name}')
            got = None
        else:
            got, why = check_field(name, lowered[name], cfg)
            if why:
                problems.append(why)
        if got is None:
            if fallback is None:
                fallback = keyword_hint(prompt, cfg)
            got = fallback[name]
        hint[name] = got
    ok = not problems
    if ok:
        note = 'ok: подсказка из ответа модели'
    else:
        src = 'по словам промпта' if str(prompt or '').strip() else 'значения по умолчанию (промпта нет)'
        filled = [n for n in REQUIRED if n not in lowered or check_field(n, lowered[n], cfg)[0] is None]
        note = f'запасной разбор ({src}) для: {", ".join(filled)} — ' + '; '.join(problems)
    return hint, ok, note


def safe_parse(text, prompt=''):
    """Как parse_hint, но никогда не бросает: при любой ошибке — запасной разбор или значения по умолчанию."""
    try:
        hint, ok, note = parse_hint(text, prompt)
    except Exception as exc:  # noqa: BLE001 — узел не роняет граф
        try:
            cfg = load_config()
            hint, note = keyword_hint(prompt, cfg), f'запасной разбор по словам промпта: ошибка разбора ({exc})'
        except Exception as exc2:  # noqa: BLE001 — нет даже конфига
            hint = {'pose_class': 'stand', 'support': 'floor', 'facing': 'camera', 'near_wall': False, 'prop': ''}
            note = f'значения по умолчанию: {exc}; {exc2}'
        ok = False
    return json.dumps({k: hint[k] for k in FIELDS}, ensure_ascii=False), bool(ok), note
