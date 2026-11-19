"""Безопасная интерпретация ответа модели проверки итогового кадра."""
import json
import re
from pathlib import Path

import yaml


def parse_answer(text):
    """Один JSON без повторных ключей; допускается только Markdown-обёртка."""
    if not isinstance(text, str):
        return None
    text = text.strip()
    match = re.fullmatch(r'```(?:json)?\s*\n([\s\S]*?)\n```', text, re.IGNORECASE)
    if match:
        text = match.group(1).strip()

    def unique_object(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError('duplicate JSON key')
            obj[key] = value
        return obj

    try:
        result = json.loads(text, object_pairs_hook=unique_object,
                            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        return result if isinstance(result, dict) else None
    except (ValueError, RecursionError):
        return None

CONFIG_PATH = Path(__file__).resolve().parents[1] / 'config' / 'auto_judge.yaml'
FIELDS = {
    'one_person': 'в кадре не один человек',
    'pose_matches': 'поза не соответствует заданию',
    'on_expected_support': 'неверная опора',
    'anatomy_ok': 'ошибка анатомии',
    'prop_ok': 'ошибка реквизита',
}

# Only these exact explanations can be associated safely with a single false
# field. Arbitrary natural language may hide a second issue or a contradiction.
PROBLEM_NOTES = {
    'one_person': 'More or fewer than one person is visible.',
    'pose_matches': 'The pose does not match the request.',
    'on_expected_support': 'The expected support is incorrect.',
    'anatomy_ok': 'The anatomy is incorrect.',
    'prop_ok': 'The required prop is missing or incorrect.',
}


def decide(text, attempt=0, cfg=None):
    """Неполный или неоднозначный ответ всегда удерживается для художника."""
    if cfg is None:
        cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding='utf-8'))
    answer = parse_answer(text)
    if not isinstance(answer, dict):
        return 'artist', 'Ответ модели не содержит корректный JSON', False
    if set(answer) != set(FIELDS) | {'notes'}:
        return 'artist', 'Ответ модели не соответствует схеме полей', False
    for key in FIELDS:
        value = answer.get(key)
        if type(value) is not bool and not (key == 'prop_ok' and key in answer and value is None):
            return 'artist', f'Неоднозначное или отсутствующее поле: {key}', False
    if not isinstance(answer.get('notes'), str):
        return 'artist', 'Неоднозначное или отсутствующее поле: notes', False
    problem_keys = [key for key in FIELDS if answer[key] is False]
    notes = answer['notes'].strip()
    # Only a single, exact explanation of the one failed field is trusted.
    # Other notes (including extra or conflicting claims) require a human.
    if notes and (len(problem_keys) != 1 or notes != PROBLEM_NOTES[problem_keys[0]]):
        return 'artist', 'Примечание модели требует проверки художником', False
    if not problem_keys:
        return 'ok', 'Проверка пройдена', True
    reason = '; '.join(FIELDS[key] for key in problem_keys)
    if len(problem_keys) == cfg['retry_problem_count'] and attempt == cfg['first_attempt']:
        return 'retry', reason, False
    return 'artist', reason, False


class AutoJudge:
    CATEGORY = '3DA/auto'
    FUNCTION = 'execute'
    RETURN_TYPES = ('STRING', 'STRING', 'BOOLEAN')
    RETURN_NAMES = ('verdict', 'reasons', 'ok')

    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'text': ('STRING', {'forceInput': True})},
                'optional': {'attempt': ('INT', {'default': 0, 'min': 0, 'max': 1})}}

    def execute(self, text, attempt=0):
        try:
            return decide(text, attempt)
        except Exception:  # noqa: BLE001 — ошибка проверки не должна выпускать кадр
            return 'artist', 'Ошибка проверки результата', False
