"""E16: персонаж «3D-контроля» (Стейси по умолчанию, Саманта) — выбор, настройки, подмена тела и манекена.
Без GPU, Blender и сервера. Запуск: python -m pytest tests/test_character_profile.py -q
"""
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
NODES = ROOT / 'nodes.py' if (ROOT / 'nodes.py').is_file() else ROOT / 'comfyui/ComfyUI-3DA/nodes.py'
sys.path[:0] = [str(ROOT), str(ROOT / 'tools/scene3d')]
spec = importlib.util.spec_from_file_location('threeda_nodes_character_test', NODES)
N = importlib.util.module_from_spec(spec)
spec.loader.exec_module(N)
EMPTY_FARM = tempfile.mkdtemp(prefix='e16_farm_')
N.config = lambda: {'repo': str(ROOT), 'farm_root': EMPTY_FARM}
import character_profile as cp  # noqa: E402


def make_character(root, name='samantha', height=2.30, placement=2.4151, dims=True):
    folder = root / 'characters' / name
    folder.mkdir(parents=True)
    body = {'char_id': name, 'version': 2, 'height_m': height, 'placement_height_m': placement}
    if dims:
        body['body_dims'] = {'H': placement, 'L_thigh': .55, 'L_torso': .76}
    (folder / 'body.json').write_text(json.dumps(body), encoding='utf-8')
    (folder / 'mannequin.blend').write_bytes(b'BLENDER')
    return folder


# ------------------------------------------------------------------------------------------ выбор в 3D-контроле
def test_control_choices_stacy_default_samantha_added():
    required = N.Control.INPUT_TYPES()['required']['character']
    assert list(required[0]) == ['Стейси', 'Саманта'] and required[1]['default'] == 'Стейси'
    assert N.CHARACTERS == {'Стейси': 'stacy', 'Саманта': 'samantha'}


def test_control_outputs_and_settings():
    control = N.Control()
    mode, prop, character, settings = control.execute()                  # умолчания узла
    assert character == 'stacy' and settings['character'] == 'stacy' and mode == 2
    _, _, character, settings = control.execute(character='Саманта')
    assert character == 'samantha' and settings['character'] == 'samantha'
    _, _, character, settings = control.execute(character='Стейси')
    assert character == 'stacy' and settings['character'] == 'stacy'
    _, _, character, settings = control.execute(character='кто-то ещё')   # неизвестное значение — Стейси
    assert character == 'stacy' and settings['character'] == 'stacy'


def test_scene_settings_character():
    assert N.scene_settings(None)['character'] == 'stacy'
    assert N.scene_settings({})['character'] == 'stacy'
    assert N.scene_settings({'character': 'samantha'})['character'] == 'samantha'
    assert N.scene_settings({'character': '../etc'})['character'] == 'stacy'
    assert N.scene_settings({'character': None})['character'] == 'stacy'
    # остальные настройки не затронуты
    plain = N.scene_settings({'strength_margin': 3.})
    assert plain['strength_margin'] == 3. and plain['body_model'] == 'mpfb'


# ------------------------------------------------------------------------------------------ запрос постановки
class Capture(N.PlacePose):
    def __init__(self):
        self.fields = None

    def runner(self, command, fields, images):
        self.fields = dict(fields)
        raise RuntimeError('stop after capture')


def place_fields(settings):
    node = Capture()
    node.execute('{}', torch.zeros(1, 8, 8, 3), 'auto', 'auto', '', '', '', 1.62, 'characters/stacy/body.json', 'color', 1., 101,
                 settings=settings)
    return node.fields


def test_place_request_stacy_unchanged():
    no_settings = place_fields(None)
    stacy = place_fields(N.scene_settings({'character': 'stacy'}))
    assert 'character' not in no_settings and 'character' not in stacy
    assert no_settings == stacy
    assert no_settings['body_json'] == 'characters/stacy/body.json' and no_settings['height_m'] == 1.62


def test_place_request_samantha_passes_character():
    fields = place_fields(N.scene_settings({'character': 'samantha'}))
    assert fields['character'] == 'samantha'
    stacy = place_fields(None)
    assert {k: v for k, v in fields.items() if k != 'character'} == stacy


# ------------------------------------------------------------------------------------------ данные персонажа
def test_resolve_stacy_is_old_path(tmp_path):
    assert cp.resolve({}, {'farm_root': str(tmp_path)}) is None
    assert cp.resolve({'character': 'stacy'}, {'farm_root': str(tmp_path)}) is None
    assert cp.resolve({'character': None}, {'farm_root': str(tmp_path)}) is None


def test_resolve_samantha(tmp_path):
    make_character(tmp_path)
    got = cp.resolve({'character': 'samantha'}, {'farm_root': str(tmp_path)})
    assert got['character'] == 'samantha'
    assert Path(got['body_json']) == tmp_path / 'characters/samantha/body.json'
    assert Path(got['mannequin']) == tmp_path / 'characters/samantha/mannequin.blend'
    assert got['height_m'] == 2.4151 and got['true_height_m'] == 2.30


def test_resolve_height_falls_back_to_height_m(tmp_path):
    folder = make_character(tmp_path)
    body = json.loads((folder / 'body.json').read_text(encoding='utf-8'))
    del body['placement_height_m']
    (folder / 'body.json').write_text(json.dumps(body), encoding='utf-8')
    got = cp.resolve({'character': 'samantha'}, {'farm_root': str(tmp_path)})
    assert got['height_m'] == 2.30


@pytest.mark.parametrize('missing', ['body.json', 'mannequin.blend'])
def test_resolve_missing_files_clear_error(tmp_path, missing):
    folder = make_character(tmp_path)
    (folder / missing).unlink()
    with pytest.raises(ValueError, match='персонаж samantha: нет файла'):
        cp.resolve({'character': 'samantha'}, {'farm_root': str(tmp_path)})


def test_resolve_bad_inputs(tmp_path):
    make_character(tmp_path)
    for bad in ('../x', 'Samantha', 'a b', '', 'x' * 40, 7):
        if bad == '':
            assert cp.resolve({'character': bad}, {'farm_root': str(tmp_path)}) is None   # пусто = умолчание
            continue
        with pytest.raises(ValueError):
            cp.resolve({'character': bad}, {'farm_root': str(tmp_path)})
    make_character(tmp_path, 'nodims', dims=False)
    with pytest.raises(ValueError, match='body_dims'):
        cp.resolve({'character': 'nodims'}, {'farm_root': str(tmp_path)})
    make_character(tmp_path, 'tiny', height=0.1, placement=0.1)
    with pytest.raises(ValueError, match='0,5–3,5'):
        cp.resolve({'character': 'tiny'}, {'farm_root': str(tmp_path)})


def test_fingerprint_stacy_key_unchanged_by_character_code(tmp_path):
    """Ключ кэша Стейси не зависит от данных других персонажей; у выбранной Саманты — зависит от её файлов."""
    cfg = {'repo': str(ROOT), 'farm_root': str(tmp_path)}
    N.config = lambda: cfg
    try:
        folder = make_character(tmp_path)
        stacy = N.PlacePose.fingerprint('x', '[]')
        assert stacy == N.PlacePose.fingerprint('x', '[]', settings=N.scene_settings({}))
        samantha = N.PlacePose.fingerprint('x', '[]', settings=N.scene_settings({'character': 'samantha'}))
        assert samantha != stacy
        (folder / 'mannequin.blend').write_bytes(b'BLENDER2')
        assert N.PlacePose.fingerprint('x', '[]', settings=N.scene_settings({'character': 'samantha'})) != samantha
        assert N.PlacePose.fingerprint('x', '[]') == stacy
    finally:
        N.config = lambda: {'repo': str(ROOT), 'farm_root': EMPTY_FARM}
