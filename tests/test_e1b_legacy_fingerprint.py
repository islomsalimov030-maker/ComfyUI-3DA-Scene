"""Frozen-node cache keys against the released baseline, at the same absolute paths.

BASE is the 3DA release that runs on the studio server (0.3.47, origin/main e64de38). The historical c17932b baseline
no longer matches since 0.3.44: released changes to nodes.py and posev2 (E12, E14) moved every key, independently of
E1c. Frozen nodes (AIO_manual_V3.0 and everything released before 16.11) must keep their keys: their hashed inputs
stay byte-identical to BASE, except the posev2 modules used only by the AUTO node 3DAPlaceCandidates (AUTO_ONLY).
Requires the BASE git object locally. No server or GPU is used.
"""
import importlib.util
import io
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
BASE = 'e64de38'
# posev2 modules read only by 3DAPlaceCandidates (candidates_helpers / candidates_cli); 3DAPlacePose does not import them,
# but its IS_CHANGED hashes every posev2/*.py. Changes to them are allowed; changes to any other hashed input are not.
AUTO_ONLY = {'candidates.py', 'candidates_cli.py', 'candidate_screen.py', 'candidate_visual.py',
             'candidate_visual_blender.py'}


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    # The historical code hashes absolute paths. Both revisions must therefore
    # run in the very same directory, not in two separate checkouts.
    tree = tmp_path / 'same_path'
    tree.mkdir()
    archive = subprocess.run(
        ['git', '-C', str(ROOT), 'archive', '--format=tar', BASE],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        # This archive is generated from the local project repository.
        tar.extractall(tree)
    monkeypatch.setenv('THREEDA_CONFIG', str(tree / 'config.yaml'))
    farm = tree / '_test_farm'
    farm.mkdir()
    (farm / 'props' / 'fixture').mkdir(parents=True)
    (farm / 'props' / 'fixture' / 'prop.json').write_text('{}', encoding='utf-8')
    (farm / 'locations' / 'sample').mkdir(parents=True)
    (farm / 'locations' / 'sample' / 'asset.txt').write_text('asset', encoding='utf-8')
    (farm / 'body.json').write_text('{}', encoding='utf-8')
    return tree, farm


def load_nodes(tree, farm):
    path = tree / 'nodes.py'
    spec = importlib.util.spec_from_file_location('_legacy_fingerprint_nodes', path)
    module = importlib.util.module_from_spec(spec)
    # Compile the file itself rather than relying on timestamp-based pyc caching.
    exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
    module.config = lambda: {'repo': str(tree), 'farm_root': str(farm)}
    return module


def install_current_fingerprint_inputs(tree):
    # Overlay every static path read by the historical IS_CHANGED methods,
    # including every posev2 Python module matched by their glob.
    paths = [
        'nodes.py', 'config.yaml', 'config/placement.yaml',
        'tools/scene3d/pack_cli.py', 'tools/scene3d/prop_composite.py',
        'tools/scene3d/prop_validation.py', 'tools/scene3d/plate/prop_render.py',
        'tools/scene3d/scene_edit.py', 'tools/scene3d/scene_export_blender.py',
        'tools/scene3d/prop_gizmo.py', 'tools/scene3d/prompt_h3.py',
    ]
    for relative in paths:
        source, target = ROOT / relative, tree / relative
        if source.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        elif target.exists():
            target.unlink()
    current = ROOT / 'tools/scene3d/posev2'
    target = tree / 'tools/scene3d/posev2'
    for path in target.glob('*.py'):
        if path.name not in AUTO_ONLY:
            path.unlink()
    for path in current.glob('*.py'):
        if path.name not in AUTO_ONLY:
            shutil.copyfile(path, target / path.name)


def keys(module):
    return {
        'Location': module.Location.IS_CHANGED('sample'),
        'Prop.empty': module.Prop.IS_CHANGED(module.NO_PROP, props_in='[]'),
        'Prop': module.Prop.IS_CHANGED('fixture'),
        'PlacePose': module.PlacePose.IS_CHANGED(body_json='body.json'),
        'NativeGuides': module.NativeGuides.IS_CHANGED(),
        'NativeComposite': module.NativeComposite.IS_CHANGED(),
        'SceneEdits': module.SceneEdits.IS_CHANGED(),
        'SceneExport': module.SceneExport.IS_CHANGED(),
    }


def assert_real_keys(values):
    for name, value in values.items():
        assert isinstance(value, str), name
        assert not value.startswith(('unfingerprinted:', 'config-error:')), (name, value)


def test_old_node_fingerprints_equal_released_baseline(checkout):
    tree, farm = checkout
    baseline = keys(load_nodes(tree, farm))
    assert_real_keys(baseline)
    install_current_fingerprint_inputs(tree)
    current = keys(load_nodes(tree, farm))
    assert_real_keys(current)
    assert current == baseline


@pytest.mark.parametrize('revision', ['baseline', 'current'])
def test_old_code_change_invalidates_legacy_keys(checkout, revision):
    tree, farm = checkout
    if revision == 'current':
        install_current_fingerprint_inputs(tree)
    module = load_nodes(tree, farm)
    before = keys(module)
    assert_real_keys(before)
    old_file = tree / 'nodes.py'
    with old_file.open('ab') as stream:
        stream.write(b'\n# test change to old node code\n')
    after = keys(module)
    # Location does not hash nodes.py, unlike the following nodes.
    assert after['Location'] == before['Location']
    for name in ('Prop.empty', 'Prop', 'PlacePose', 'NativeGuides',
                 'NativeComposite', 'SceneEdits', 'SceneExport'):
        assert after[name] != before[name], name


@pytest.mark.parametrize('revision', ['baseline', 'current'])
def test_old_pose_module_change_invalidates_place_pose(checkout, revision):
    tree, farm = checkout
    if revision == 'current':
        install_current_fingerprint_inputs(tree)
    module = load_nodes(tree, farm)
    before = keys(module)
    assert_real_keys(before)
    with (tree / 'tools/scene3d/posev2/place.py').open('ab') as stream:
        stream.write(b'\n# test change to old pose code\n')
    after = keys(module)
    assert after['PlacePose'] != before['PlacePose']


def test_frozen_inputs_are_byte_identical_to_baseline(checkout):
    """Every hashed input except AUTO_ONLY modules equals BASE byte for byte (nodes.py, config/placement.yaml, ...)."""
    tree, _ = checkout
    names = ['nodes.py', 'config.yaml', 'config/placement.yaml', 'tools/scene3d/pack_cli.py',
             'tools/scene3d/prop_composite.py', 'tools/scene3d/prop_validation.py', 'tools/scene3d/plate/prop_render.py',
             'tools/scene3d/scene_edit.py', 'tools/scene3d/scene_export_blender.py', 'tools/scene3d/prop_gizmo.py',
             'tools/scene3d/prompt_h3.py']
    names += sorted({'tools/scene3d/posev2/' + p.name for p in (tree / 'tools/scene3d/posev2').glob('*.py')} |
                    {'tools/scene3d/posev2/' + p.name for p in (ROOT / 'tools/scene3d/posev2').glob('*.py')})
    changed = []
    for relative in names:
        old, new = tree / relative, ROOT / relative
        if relative.startswith('tools/scene3d/posev2/') and Path(relative).name in AUTO_ONLY:
            continue
        if old.is_file() != new.is_file() or (old.is_file() and old.read_bytes() != new.read_bytes()):
            changed.append(relative)
    assert not changed, changed
