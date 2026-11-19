"""Безопасная интерпретация ответа модели проверки итогового кадра."""
from pathlib import Path

import yaml

from .vision import parse_answer

CONFIG_PATH = Path(__file__).resolve().parents[1] / 'config' / 'auto_judge.yaml'
FIELDS = {
    'one_person': 'в кадре не один человек',
    'pose_matches': 'поза не соответствует заданию',
    'on_expected_support': 'неверная опора',
    'anatomy_ok': 'ошибка анатомии',
    'prop_ok': 'ошибка реквизита',
}


def decide(text, attempt=0, cfg=None):
    """Неполный или неоднозначный ответ всегда удерживается для художника."""
    if cfg is None:
        cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding='utf-8'))
    answer = parse_answer(text)
    if not isinstance(answer, dict):
        return 'artist', 'Ответ модели не содержит корректный JSON', False
    for key in FIELDS:
        value = answer.get(key)
        if type(value) is not bool and not (key == 'prop_ok' and key in answer and value is None):
            return 'artist', f'Неоднозначное или отсутствующее поле: {key}', False
    if not isinstance(answer.get('notes'), str):
        return 'artist', 'Неоднозначное или отсутствующее поле: notes', False
    problems = [description for key, description in FIELDS.items() if answer[key] is False]
    if not problems:
        return 'ok', 'Проверка пройдена', True
    reason = '; '.join(problems)
    if len(problems) == cfg['retry_problem_count'] and attempt == cfg['first_attempt']:
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
