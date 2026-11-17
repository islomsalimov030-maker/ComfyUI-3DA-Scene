"""E6: роль «зрение» — разбор ответа модели о кандидатах постановки и шаг итерации.

Модель (VLM) смотрит contact_sheet 3DAPlaceCandidates и картинку локации и отвечает JSON
{"best": <номер или -1>, "reason": "...", "problems": {"<номер>": "floating|sinking|...|ok"}}.
Из ответа берутся только номер, причина и метки проблем; координаты, сдвиги и любые другие поля не читаются.
Числа (лимит прогонов, порядок наборов) — config/placement.yaml, candidates.vision.
"""
import json
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parents[1] / 'config' / 'placement.yaml'
PROBLEMS = ('floating', 'sinking', 'wrong_support', 'wrong_facing', 'bad_scale', 'ok')
PROBLEMS_RU = {'floating': 'висит', 'sinking': 'проваливается', 'wrong_support': 'не та опора',
               'wrong_facing': 'не тот разворот', 'bad_scale': 'не тот масштаб', 'ok': 'ок'}


def load_config(path=None):
    data = yaml.safe_load(Path(path or CONFIG_PATH).read_text(encoding='utf-8'))
    return dict(data['candidates']['vision'])


def parse_answer(text):
    """Первый JSON-объект в ответе (```json-обёртка, текст вокруг — допустимы); нет — None."""
    if not isinstance(text, str):
        return None
    dec = json.JSONDecoder()
    pos = text.find('{')
    while pos >= 0:
        try:
            value, _ = dec.raw_decode(text, pos)
        except ValueError:
            pos = text.find('{', pos + 1)
            continue
        if isinstance(value, dict):
            return value
        pos = text.find('{', pos + 1)
    return None


def as_index(value):
    """Номер кандидата: int или строка из цифр (со знаком); bool, дробь, прочее — None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().lstrip('#').lstrip('-').isdigit():
        return int(value.strip().lstrip('#'))
    return None


def valid_indices(candidates_json):
    """Номера выполненных кандидатов (есть оценка) из candidates_json 3DAPlaceCandidates."""
    try:
        data = json.loads(candidates_json) if isinstance(candidates_json, str) else candidates_json
    except ValueError:
        return []
    cands = data.get('candidates') if isinstance(data, dict) else None
    out = []
    for c in cands or []:
        if isinstance(c, dict) and c.get('score') is not None and as_index(c.get('index')) is not None:
            out.append(as_index(c['index']))
    return out


def clean_problems(problems, valid):
    """{номер: метка} только для известных номеров и меток из PROBLEMS."""
    out = {}
    if not isinstance(problems, dict):
        return out
    for k, v in problems.items():
        i = as_index(k)
        tag = str(v).strip().lower() if isinstance(v, str) else None
        if i in valid and tag in PROBLEMS:
            out[i] = tag
    return out


def artist_reason(problems):
    """Причина для художника по меткам модели: «висит ×2, не та опора ×1»; меток нет — общая фраза."""
    counts = {}
    for tag in problems.values():
        if tag != 'ok':
            counts[tag] = counts.get(tag, 0) + 1
    if not counts:
        return 'модель забраковала все кандидаты без пояснения'
    return ', '.join(f'{PROBLEMS_RU[t]} ×{n}' for t, n in sorted(counts.items(), key=lambda x: (-x[1], x[0])))


def next_variant_set(round_index, all_bad, cfg):
    """Следующий variant_set или None: модель довольна, или прогонов уже max_rounds (жёсткий предел)."""
    if not all_bad:
        return None
    nxt = int(round_index) + 1
    order = list(cfg['variant_order'])
    if nxt >= int(cfg['max_rounds']) or nxt >= len(order):
        return None
    return int(order[nxt])


def decide(text, candidates_json, round_index=0, cfg=None):
    """Ответ модели → решение. Никогда не падает.
    pick: номер кандидата; −1 — «лучший по оценке» (ответ битый, номер вне списка или модель сказала −1).
    all_bad: модель сказала −1 (все плохие). artist: all_bad на последнем прогоне — строку смотрит художник."""
    cfg = cfg or load_config()
    round_index = max(0, int(round_index or 0))
    valid = valid_indices(candidates_json)
    info = {'round': round_index, 'pick': -1, 'all_bad': False, 'artist': False, 'next_variant_set': None,
            'answer_ok': False, 'best_raw': None, 'reason': '', 'problems': {}, 'valid': valid}
    ans = parse_answer(text)
    if ans is None:
        info['note'] = 'ответ модели не разобран — лучший по оценке'
        return info
    best = as_index(ans.get('best'))
    info['best_raw'] = ans.get('best') if isinstance(ans.get('best'), (int, float, str)) else None
    info['reason'] = str(ans.get('reason') or '')[:300]
    info['problems'] = clean_problems(ans.get('problems'), valid)
    if best == -1:
        info.update(answer_ok=True, all_bad=True)
        info['next_variant_set'] = next_variant_set(round_index, True, cfg)
        if info['next_variant_set'] is None:
            info['artist'] = True
            info['note'] = 'художнику: ' + artist_reason(info['problems'])
        else:
            info['note'] = f"модель: все кандидаты плохи — повтор с variant_set {info['next_variant_set']}"
        return info
    if best is None or best not in valid:
        info['note'] = f"номер {ans.get('best')!r} вне списка {valid} — лучший по оценке"
        return info
    info.update(answer_ok=True, pick=best, note=f'модель выбрала кандидата {best}')
    return info


class VisionPick:
    """3DA · выбор кандидата (зрение): ответ модели + candidates_json → pick для 3DAPlaceCandidates."""
    CATEGORY = '3DA/auto'
    FUNCTION = 'execute'
    RETURN_TYPES = ('INT', 'BOOLEAN', 'STRING')
    RETURN_NAMES = ('pick', 'all_bad', 'note')

    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'text': ('STRING', {'forceInput': True, 'tooltip': 'Ответ модели (JSON best/reason/problems)'}),
                             'candidates_json': ('STRING', {'forceInput': True,
                                                            'tooltip': 'Выход candidates_json узла «постановка: кандидаты»'})},
                'optional': {'round': ('INT', {'default': 0, 'min': 0, 'max': 2,
                                               'tooltip': 'Номер прогона 0..2; на последнем «все плохие» → художнику'})}}

    def execute(self, text, candidates_json, round=0):  # noqa: A002 — имя входа в графе
        try:
            info = decide(text, candidates_json, round)
            note = info['note'] + '\n' + json.dumps({k: v for k, v in info.items() if k != 'note'},
                                                    ensure_ascii=False, default=str)
            return {'ui': {'text': [note]}, 'result': (int(info['pick']), bool(info['all_bad']), note)}
        except Exception as exc:  # noqa: BLE001 — узел никогда не падает
            note = f'зрение: ошибка разбора ({str(exc)[:200]}) — лучший по оценке'
            return {'ui': {'text': [note]}, 'result': (-1, False, note)}
