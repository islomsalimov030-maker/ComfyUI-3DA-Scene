"""E5: разбор ответа LLM в hint_json (3DAParseHint), запасной разбор, hint_json из состояния пакета (3DAAutoRow).
Без сервера и GPU. Запуск: python -m pytest tests/test_parse_hint.py -q"""
import ast
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from auto import parse as P, store  # noqa: E402

FIXTURE = ROOT / 'tests/fixtures/batch_table'
CFG = P.load_config()
GOOD = '{"pose_class": "sit_edge", "support": "sofa", "facing": "camera", "near_wall": false, "prop": "book"}'
PROMPT = 'The woman sits on the bench, holding a towel, facing the camera.'


def hint_of(text, prompt=''):
    hint_json, ok, note = P.safe_parse(text, prompt)
    return json.loads(hint_json), ok, note


# ------------------------------------------------------------------------------------------------ извлечение JSON
@pytest.mark.parametrize('text', [
    GOOD,
    '```json\n' + GOOD + '\n```',
    '```\n' + GOOD + '\n```',
    'Here is the result:\n' + GOOD + '\nDone.',
    GOOD.replace('"book"}', '"book",}'),                      # хвостовая запятая
])
def test_extract_json_variants(text):
    value, why = P.extract_json(text)
    assert why == '' and value['pose_class'] == 'sit_edge' and value['prop'] == 'book'


def test_extract_json_nested_and_braces_in_strings():
    value, _ = P.extract_json('note {not json} then {"pose_class": "lie", "support": "bed {big}", "x": {"y": 1}}')
    assert value['support'] == 'bed {big}' and value['x'] == {'y': 1}


@pytest.mark.parametrize('text,why', [('', 'пустой'), ('   ', 'пустой'), ('the woman sits', 'нет JSON'),
                                      ('{"pose_class": "lie", ', 'не читается'), (None, 'пустой')])
def test_extract_json_missing(text, why):
    value, reason = P.extract_json(text)
    assert value is None and why in reason


# ------------------------------------------------------------------------------------------------ проверка значений
def test_good_answer_ok():
    hint, ok, note = hint_of(GOOD, PROMPT)
    assert ok and note.startswith('ok')
    assert hint == {'pose_class': 'sit_edge', 'support': 'sofa', 'facing': 'camera', 'near_wall': False, 'prop': 'book'}
    assert list(hint) == ['pose_class', 'support', 'facing', 'near_wall', 'prop']


def test_normalization_aliases_and_support_words():
    text = '{"Pose_Class": "Sit-Edge", "support": "Massage Table", "facing": "profile", "near_wall": "yes", "prop": null}'
    hint, ok, _ = hint_of(text)
    assert ok and hint == {'pose_class': 'sit_edge', 'support': 'massage_table', 'facing': 'side',
                           'near_wall': True, 'prop': ''}
    for raw, word in (('weight bench', 'bench'), ('exercise mat', 'mat'), ('gym machine frame', 'frame'),
                      ('the couch', 'couch'), ('window ledge seat thing', 'window ledge seat')):
        assert P.norm_support(raw, CFG) == word
    assert hint_of(GOOD.replace('"sit_edge"', '"lying"'))[0]['pose_class'] == 'lie'


def test_classes_match_nodes_py():
    """Список классов в YAML = CLASSES в nodes.py без auto (nodes.py не импортируем: torch)."""
    tree = ast.parse((ROOT / 'nodes.py').read_text(encoding='utf-8'))
    classes = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                   and getattr(n.targets[0], 'id', '') == 'CLASSES')
    assert CFG['pose_classes'] == [c for c in classes if c != 'auto']


@pytest.mark.parametrize('field,bad', [('pose_class', 'levitate'), ('facing', 'up'), ('near_wall', 'maybe'),
                                       ('support', ''), ('pose_class', 5)])
def test_bad_value_field_fallback(field, bad):
    value = json.loads(GOOD)
    value[field] = bad
    hint, ok, note = hint_of(json.dumps(value), PROMPT)
    assert not ok and field in note and 'запасной разбор' in note
    fb = P.keyword_hint(PROMPT, CFG)
    assert hint[field] == fb[field]
    others = {k: v for k, v in json.loads(GOOD).items() if k != field}
    assert all(hint[k] == v for k, v in others.items())     # верные поля модели не трогаются


def test_empty_object_uses_fallback():
    prompt = 'A person sitting on a chair facing the camera'
    hint, ok, note = hint_of('{}', prompt)
    assert not ok and all(f'нет поля {field}' in note for field in P.REQUIRED)
    assert hint == P.keyword_hint(prompt, CFG)
    assert hint['pose_class'] == 'sit_edge' and hint['support'] == 'chair'


def test_missing_field():
    value = json.loads(GOOD)
    del value['near_wall'], value['prop']
    hint, ok, note = hint_of(json.dumps(value), 'The woman stands against the wall.')
    assert not ok and 'нет поля near_wall' in note and hint['near_wall'] is True and hint['pose_class'] == 'sit_edge'
    assert 'prop' not in note                                 # prop необязателен


# ------------------------------------------------------------------------------------------------ запасной разбор
@pytest.mark.parametrize('text', ['', 'Sorry, I cannot help.', '{"pose_class": "flying", "support": "", "facing": "?", "near_wall": 3}',
                                  '{broken', '[1, 2]', None])
def test_broken_answers_never_fail(text):
    hint, ok, note = hint_of(text, PROMPT)
    assert not ok and note
    assert hint['pose_class'] == 'sit_edge' and hint['support'] == 'bench' and hint['facing'] == 'camera'
    assert hint['near_wall'] is False and hint['prop'] == 'towel'


@pytest.mark.parametrize('prompt,expect', [
    ('She lies on her back on the exercise mat doing a sit-up.', {'pose_class': 'recline', 'support': 'mat'}),
    ('She lies on her side on the sofa, head propped on her hand.', {'pose_class': 'lie', 'support': 'sofa', 'facing': 'camera'}),
    ('She lies on the bed propped on a raised pillow, reading a book.', {'pose_class': 'recline', 'prop': 'book'}),
    ('She sits on the sofa, one knee pulled up to her chest.', {'pose_class': 'sit_tucked', 'support': 'sofa'}),
    ('She sits cross-legged on the floor.', {'pose_class': 'sit_floor', 'support': 'floor'}),
    ('She squats on the floor holding the bars.', {'pose_class': 'crouch', 'support': 'floor', 'prop': 'bars'}),
    ('She kneels on the yoga mat.', {'pose_class': 'kneel', 'support': 'mat'}),
    ('She is on all fours on the rug.', {'pose_class': 'all_fours', 'support': 'rug'}),
    ('She stands at the stove with her back to the camera.', {'pose_class': 'stand', 'support': 'floor',
                                                              'facing': 'away', 'near_wall': True}),
    ('She leans her back against the frame of the machine.', {'pose_class': 'stand', 'near_wall': True}),
    ('She sits on the massage table, seen from the side.', {'pose_class': 'sit_edge', 'support': 'massage_table',
                                                            'facing': 'side'}),
])
def test_keyword_hint(prompt, expect):
    hint = P.keyword_hint(prompt, CFG)
    assert {k: hint[k] for k in expect} == expect


def test_no_prompt_defaults():
    hint, ok, note = hint_of('nonsense', '')
    assert not ok and 'по умолчанию' in note
    assert hint == {'pose_class': 'stand', 'support': 'floor', 'facing': 'camera', 'near_wall': False, 'prop': ''}


def test_safe_parse_survives_config_error(monkeypatch):
    monkeypatch.setattr(P, 'CONFIG_PATH', Path('/nonexistent/placement.yaml'))
    hint_json, ok, note = P.safe_parse(GOOD, PROMPT)
    assert not ok and json.loads(hint_json)['pose_class'] == 'stand' and 'по умолчанию' in note


# ------------------------------------------------------------------------------------------------ узлы и состояние
@pytest.fixture
def roots(tmp_path):
    store.set_roots(tmp_path / 'input' / 'auto', tmp_path / 'output' / 'auto')
    shutil.copytree(FIXTURE, tmp_path / 'input' / 'auto' / 'pack')
    yield tmp_path
    store._roots.clear()


def test_node_parse_hint():
    pytest.importorskip('torch')
    from auto import nodes as Nd
    assert Nd.NODE_CLASS_MAPPINGS['3DAParseHint'] is Nd.ParseHint
    assert Nd.NODE_DISPLAY_NAME_MAPPINGS['3DAParseHint'] == '3DA · разбор промпта' and Nd.ParseHint.CATEGORY == '3DA/auto'
    assert Nd.ParseHint.RETURN_NAMES == ('hint_json', 'ok', 'note')
    hint_json, ok, note = Nd.ParseHint().execute('```json\n' + GOOD + '\n```', PROMPT)['result']
    assert ok and json.loads(hint_json)['support'] == 'sofa'
    hint_json, ok, note = Nd.ParseHint().execute('', PROMPT)['result']
    assert not ok and json.loads(hint_json)['support'] == 'bench'


def test_row_hint_json_from_phase1(roots):
    """Фаза 1 сохраняет hint_json (AutoSave, поле text) → фаза 2 читает его выходом hint_json узла «строка»."""
    pytest.importorskip('torch')
    from auto import nodes as Nd
    old = ('n_images', 'prompt', 'mod_character', 'mod_face', 'mod_prop', 'row_id', 'skip')
    assert Nd.AutoRow.RETURN_NAMES[Nd.N:Nd.N + 7] == old and Nd.AutoRow.RETURN_NAMES[-1] == 'hint_json'
    assert Nd.AutoRow.RETURN_TYPES[-1] == 'STRING' and len(Nd.AutoRow.RETURN_TYPES) == Nd.N + 8
    batch_json = Nd.AutoBatch().execute('pack', '1,2')['result'][0]
    row = Nd.AutoRow().execute(batch_json, 0, 1)['result']
    assert row[-1] == '' and row[-2] is False                 # до фазы 1 подсказки нет
    hint_json, ok, _ = Nd.ParseHint().execute(GOOD, row[Nd.N + 1])['result']
    assert 'сохранено файлов 1' in Nd.AutoSave().execute(batch_json, 0, 1, text=hint_json, status='parsed')['result'][0]
    row2 = Nd.AutoRow().execute(batch_json, 0, 2)['result']
    assert row2[-1] == hint_json and row2[-2] is False and json.loads(row2[-1])['pose_class'] == 'sit_edge'
    assert Nd.AutoRow().execute(batch_json, 1, 2)['result'][-1] == ''     # другая строка — своё (пусто)
    # повтор фазы 1 с другим текстом — отдаётся последний сохранённый
    data = json.loads(batch_json)
    from auto import state as S
    st = S.load(data['out_dir'])
    st['rows']['001']['phase_done'].pop('1')
    S.save(data['out_dir'], st)
    Nd.AutoSave().execute(batch_json, 0, 1, text='{"pose_class": "lie"}')
    assert Nd.AutoRow().execute(batch_json, 0, 2)['result'][-1] == '{"pose_class": "lie"}'
    # битая ссылка на файл / сломанный пакет — пусто, без исключения
    assert Nd.saved_text(data['out_dir'], {'files': {'1': ['rows/001/nope/p1_text.txt']}}, 1) == ''
    assert Nd.AutoRow().execute('{"error": "x"}', 0, 2)['result'][-1] == ''
