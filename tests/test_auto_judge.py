import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = 'auto_judge_test_package'
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT / 'auto')]
sys.modules[PACKAGE] = package
spec = importlib.util.spec_from_file_location(f'{PACKAGE}.judge', ROOT / 'auto' / 'judge.py')
judge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(judge)


def answer(**updates):
    data = dict(one_person=True, pose_matches=True, on_expected_support=True,
                anatomy_ok=True, prop_ok=None, notes='')
    data.update(updates)
    return json.dumps(data)


def test_success_and_wrapped_json():
    assert judge.decide('```json\n' + answer() + '\n```') == ('ok', 'Проверка пройдена', True)


@pytest.mark.parametrize('field', list(judge.FIELDS))
def test_one_problem_retries_only_once(field):
    text = answer(**{field: False})
    assert judge.decide(text, 0)[0] == 'retry'
    assert judge.decide(text, 1)[0] == 'artist'


@pytest.mark.parametrize('field', list(judge.FIELDS))
def test_one_problem_with_explanation_retries_only_once(field):
    text = answer(**{field: False, 'notes': judge.PROBLEM_NOTES[field]})
    verdict, reason, ok = judge.decide(text, 0)
    assert (verdict, reason, ok) == ('retry', judge.FIELDS[field], False)
    assert judge.decide(text, 1)[0] == 'artist'


@pytest.mark.parametrize('text', [
    answer(notes=judge.PROBLEM_NOTES['one_person']),
    answer(one_person=False, notes=judge.PROBLEM_NOTES['anatomy_ok']),
    answer(one_person=False, notes=judge.PROBLEM_NOTES['one_person'] + ' Anatomy is incorrect.'),
    answer(one_person=False, notes=judge.PROBLEM_NOTES['one_person'] + ' Actually one person.'),
    answer(one_person=False, notes='Perhaps there are two people.'),
    answer(one_person=False, anatomy_ok=False, notes=judge.PROBLEM_NOTES['one_person']),
    answer(pose_matches=None, notes='The pose is uncertain.'),
])
def test_contradictory_or_uncertain_notes_are_held(text):
    assert judge.decide(text)[0] == 'artist'


def test_multiple_problems_go_to_artist():
    assert judge.decide(answer(one_person=False, anatomy_ok=False))[0] == 'artist'


@pytest.mark.parametrize('text', ['oops', '{}', answer(one_person=1), answer(prop_ok='null'),
                                  answer(notes=None), answer(pose_matches=None)])
def test_invalid_response_is_never_approved(text):
    verdict, _, ok = judge.decide(text)
    assert verdict == 'artist'
    assert not ok


@pytest.mark.parametrize('text', [
    answer().replace('"one_person": true', '"one_person": false, "one_person": true'),
    answer() + '\n' + answer(one_person=False),
    'The image has two people.\n' + answer(),
    answer() + '\nThe image has two people.',
    answer(notes='Two people are visible.'),
    answer(notes='Анатомия повреждена'),
    answer(notes='Everything looks fine.'),
    answer(additional_observation='Two people visible'),
])
def test_ambiguous_answer_is_held_for_artist(text):
    verdict, _, ok = judge.decide(text)
    assert verdict == 'artist'
    assert not ok


def test_node_catches_config_failure(monkeypatch):
    def fail(*args, **kwargs):
        raise OSError('missing config')
    monkeypatch.setattr(judge, 'decide', fail)
    assert judge.AutoJudge().execute('{}')[0] == 'artist'
