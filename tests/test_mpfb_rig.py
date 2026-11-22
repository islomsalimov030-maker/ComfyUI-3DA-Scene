"""E10c: риг MPFB в 3D-окне — формат файла web/mpfbRig.dat, порт retarget_world/LBS (web/mpfbRig.js), привязка в poseHandles.
Без GPU, Blender и сервера. Запуск: python -m pytest tests/test_mpfb_rig.py -q
"""
import gzip
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools/scene3d'), str(ROOT / 'tests')]
import mpfb_rig_dump  # noqa: E402
import test_pose_ik as base  # noqa: E402

NODE = shutil.which('node')
ASSET = ROOT / 'web/mpfbRig.dat'
ASSET_SAMANTHA = ROOT / 'web/mpfbRig_samantha.dat'      # E16: риг манекена Саманты (та же топология и веса, свои кости/меш)


def read_asset(path):
    raw = gzip.decompress(Path(path).read_bytes())
    (hl,) = struct.unpack_from('<I', raw)
    header = json.loads(raw[4:4 + hl])
    data = {}
    for name, (offset, dtype, count) in header['sections'].items():
        start = 4 + hl + offset
        data[name] = np.frombuffer(raw, np.dtype(dtype), count, start)
    return header, data


def test_asset_layout_and_weights():
    header, d = read_asset(ASSET)
    nV, nT, nB, top = header['n_vertices'], header['n_triangles'], len(header['bones']), header['top']
    assert (nV, nT, nB, top) == (54078, 108144, 163, 8)
    assert header['format'] == '3da-mpfb-rig-1' and 'CC0' in header['license'] and 'GPL' in header['license']
    assert d['rest'].size == nB * 16 and d['v_arm'].size == nV * 3 and d['idx'].size == nV * top and d['tris'].size == nT * 3
    assert d['parent'][0] == -1 and (d['parent'][1:] < np.arange(1, nB)).all()
    w = d['wgt'].reshape(nV, top) / 65535.0
    assert np.abs(w.sum(1) - 1).max() < 2e-4
    assert d['idx'].max() < nB and d['tris'].max() == nV - 1
    assert (np.diff(w, axis=1) <= 1e-4).all(), 'веса по убыванию'
    rest = d['rest'].reshape(nB, 4, 4)
    assert np.allclose(rest[:, 3], [0, 0, 0, 1])
    # кости rest: ортонормальный базис
    R = rest[:, :3, :3]
    assert np.allclose(np.einsum('bij,bkj->bik', R, R), np.eye(3), atol=1e-5)
    # геометрия рига — человек ростом ~1.5 м
    v = d['v_arm'].reshape(-1, 3)
    assert 1.4 < np.ptp(v[:, 2]) < 1.7


def test_asset_samantha_layout():
    header, d = read_asset(ASSET_SAMANTHA)
    st, ds = read_asset(ASSET)
    assert (header['n_vertices'], header['n_triangles'], header['bones'], header['top']) == (
        st['n_vertices'], st['n_triangles'], st['bones'], st['top'])
    # топология MPFB общая (окно узнаёт тело по числу треугольников); у нескольких квадов другая диагональ — триангуляция зависит от формы
    diff = (d['tris'].reshape(-1, 3) != ds['tris'].reshape(-1, 3)).any(1)
    assert diff.sum() < 200, diff.sum()
    assert set(np.unique(d['tris'])) == set(np.unique(ds['tris']))
    assert np.abs(d['wgt'].astype(float).reshape(-1, 8).sum(1) / 65535 - 1).max() < 2e-4
    v = d['v_arm'].reshape(-1, 3)
    height = np.ptp(v[:, 2]) * header['rig_world'][2][2]      # v_arm в осях рига, масштаб рига в rig_world
    assert 2.27 < height < 2.33, f'манекен Саманты ростом 2,30 м (по мешу рига {height:.3f})'
    assert not np.allclose(v, ds['v_arm'].reshape(-1, 3), atol=1e-3), 'у Саманты своя фигура'


def test_writer_roundtrip(tmp_path):
    nV, nB, top = 5, 3, 2
    state = dict(names=['root', 'a', 'b'], parent=np.array([-1, 0, 1]), rest=np.tile(np.eye(4), (nB, 1, 1)),
                 length=np.array([.1, .2, .3]), v_arm=np.arange(nV * 3, dtype=float).reshape(nV, 3) / 10,
                 idx=np.array([[0, 1], [1, 2], [2, 0], [0, 2], [1, 0]], np.uint8),
                 wgt=np.array([[.75, .25], [.5, .5], [1, 0], [.9, .1], [.6, .4]]), tris=np.array([[0, 1, 2], [2, 3, 4]], np.uint32),
                 dropped=np.zeros(nV), rig_world=np.eye(4), body_world=np.eye(4))
    out = tmp_path / 'rig.dat'
    mpfb_rig_dump.write_asset(state, out, {'mannequin': 'x.blend'}, top=top)
    header, d = read_asset(out)
    assert header['bones'] == ['root', 'a', 'b'] and header['n_vertices'] == nV and header['n_triangles'] == 2
    assert np.allclose(d['v_arm'].reshape(nV, 3), state['v_arm'], atol=1e-6)
    assert np.array_equal(d['idx'].reshape(nV, top), state['idx'])
    assert np.allclose(d['wgt'].reshape(nV, top) / 65535, state['wgt'], atol=1e-5)
    assert out.read_bytes() == (tmp_path / 'rig.dat').read_bytes()
    # детерминизм: тот же вход -> те же байты (mtime в gzip нулевой)
    mpfb_rig_dump.write_asset(state, tmp_path / 'rig2.dat', {'mannequin': 'x.blend'}, top=top)
    assert (tmp_path / 'rig2.dat').read_bytes() == out.read_bytes()


@pytest.mark.skipif(NODE is None, reason='node не установлен')
def test_rig_binding_and_pose_handles_cpu(tmp_path):
    """Настоящие mpfbRig.js/poseHandles.js на настоящем файле рига: привязка, правки, грудь, отказы, запасной путь."""
    run_cpu_test(tmp_path)


@pytest.mark.skipif(NODE is None, reason='node не установлен')
def test_rig_binding_and_pose_handles_cpu_samantha(tmp_path):
    """E16: тот же CPU-тест ручек E10/E10b на ригу и меше Саманты + проба «сначала Стейси, затем её файл»."""
    run_cpu_test(tmp_path, 'mpfbRig_samantha.dat')


def run_cpu_test(tmp_path, rig_file=None):
    K = base.skeleton()
    data = tmp_path / 'data.json'
    data.write_text(json.dumps(dict(K=K.tolist(), conf=base.CONF)), encoding='utf-8')
    res = subprocess.run([NODE, str(ROOT / 'tests/mpfb_rig_cpu.mjs'), str(data)] + ([rig_file] if rig_file else []),
                         capture_output=True, text=True, timeout=600)
    assert res.returncode == 0, res.stdout + res.stderr
    assert 'mpfb_rig_cpu: ok' in res.stdout
