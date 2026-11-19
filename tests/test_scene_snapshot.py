import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('scene_snapshot', Path(__file__).parents[1] / 'auto' / 'scene_snapshot.py')
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)
ROW = 'a' * 64


def test_roundtrip_is_canonical_and_independent():
    assets = {'mesh': b'geometry', 'image': b'pixels', 'guides': b'guide data'}
    selection = {'candidate': 2, 'outputs': ['image', 'guides']}
    payload, digest = S.create(ROW, selection, assets)
    assert (payload, digest) == S.create(ROW, {'outputs': ['image', 'guides'], 'candidate': 2},
                                         dict(reversed(list(assets.items()))))
    assets['mesh'] = b'changed'
    selection['candidate'] = 1
    selected, restored = S.restore(payload, digest, ROW)
    assert selected['candidate'] == 2
    assert restored['mesh'] == b'geometry'
    with pytest.raises(TypeError):
        restored['mesh'] = b'changed'
    selected['candidate'] = 3
    assert S.restore(payload, digest, ROW)[0]['candidate'] == 2


def test_reject_tamper_wrong_identity_and_swapped_snapshot():
    one, digest = S.create(ROW, {'candidate': 0}, {'mesh': b'a'})
    other, _ = S.create(ROW, {'candidate': 1}, {'mesh': b'a'})
    with pytest.raises(ValueError):
        S.restore(one + b' ', digest, ROW)
    with pytest.raises(ValueError):
        S.restore(one, digest, 'b' * 64)
    with pytest.raises(ValueError):
        S.restore(other, digest, ROW)


@pytest.mark.parametrize('name', ['../outside', '/absolute', 'a/b', 'a\\b', '..', ''])
def test_no_paths(name):
    with pytest.raises(ValueError):
        S.create(ROW, {}, {name: b'data'})


def test_reject_malformed_data_even_with_matching_outer_digest():
    payload, _ = S.create(ROW, {}, {'mesh': b'data'})
    obj = json.loads(payload)
    obj['blobs']['mesh']['data'] = 'eHh4eA=='
    corrupt = json.dumps(obj, sort_keys=True, separators=(',', ':')).encode('ascii')
    with pytest.raises(ValueError):
        S.restore(corrupt, hashlib.sha256(corrupt).hexdigest(), ROW)


def test_duplicate_keys_are_rejected():
    payload = b'{"version":1,"version":1}'
    with pytest.raises(ValueError):
        S.restore(payload, hashlib.sha256(payload).hexdigest(), ROW)


def test_limits_and_non_json_rejected(monkeypatch):
    monkeypatch.setattr(S, '_MAX_BLOB', 2)
    with pytest.raises(ValueError):
        S.create(ROW, {}, {'mesh': b'abc'})
    with pytest.raises(ValueError):
        S.create(ROW, {'bad': float('nan')}, {'mesh': b'a'})
    with pytest.raises(ValueError):
        S.create(ROW, {}, {'mesh': bytearray(b'a')})
