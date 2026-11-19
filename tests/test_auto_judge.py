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


def test_multiple_problems_go_to_artist():
    assert judge.decide(answer(one_person=False, anatomy_ok=False))[0] == 'artist'


@pytest.mark.parametrize('text', ['oops', '{}', answer(one_person=1), answer(prop_ok='null'),
                                  answer(notes=None), answer(pose_matches=None)])
def test_invalid_response_is_never_approved(text):
    verdict, _, ok = judge.decide(text)
    assert verdict == 'artist'
    assert not ok


def test_node_catches_config_failure(monkeypatch):
    def fail(*args, **kwargs):
        raise OSError('missing config')
    monkeypatch.setattr(judge, 'decide', fail)
    assert judge.AutoJudge().execute('{}')[0] == 'artist'
