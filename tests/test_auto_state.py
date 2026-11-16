"""Состояние пакета: сведение, фазы, атомарная запись и возобновление после обрыва; приём файлов загрузки.
Запуск: python -m pytest tests/test_auto_state.py -q"""
import asyncio
import json
import os
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from auto import batch as B, state as S, store  # noqa: E402

FIXTURE = ROOT / 'tests/fixtures/batch_table'


@pytest.fixture
def roots(tmp_path):
    store.set_roots(tmp_path / 'input' / 'auto', tmp_path / 'output' / 'auto')
    shutil.copytree(FIXTURE, tmp_path / 'input' / 'auto' / 'pack')
    yield tmp_path
    store._roots.clear()


def test_sync_statuses(roots):
    name, parsed, state, out_dir = store.open_batch('pack')
    assert out_dir == roots / 'output' / 'auto' / 'pack'
    assert json.loads((out_dir / 'state.json').read_text(encoding='utf-8'))['order'] == ['001', '002', '003', '004', '005']
    st = state['rows']
    assert st['001']['status'] == 'parsed' and st['001']['attempts'] == 0
    assert st['003']['status'] == 'error' and st['003']['reason'] == 'нет фото 2 (поза)'
    assert S.pending(state, 1) == ['001', '002']
    assert S.counts(state)['error'] == 3
    # повторное сведение без изменений не переписывает файл
    mtime = (out_dir / 'state.json').stat().st_mtime_ns
    store.open_batch('pack')
    assert (out_dir / 'state.json').stat().st_mtime_ns == mtime


def test_mark_and_resume_after_crash(roots, monkeypatch):
    name, parsed, state, out_dir = store.open_batch('pack')
    S.mark(out_dir, '001', 1, status='placed', files=['rows/001/p1_1.png'])
    # «обрыв» посреди записи: tmp написан, replace не случился → на диске прежний целый state.json
    real_replace = os.replace

    def crash(src, dst):
        raise KeyboardInterrupt('обрыв')
    monkeypatch.setattr(os, 'replace', crash)
    with pytest.raises(KeyboardInterrupt):
        S.mark(out_dir, '002', 1, status='placed')
    monkeypatch.setattr(os, 'replace', real_replace)
    assert list(out_dir.glob('state.json.*.tmp'))
    on_disk = S.load(out_dir)
    assert '1' in on_disk['rows']['001']['phase_done'] and on_disk['rows']['002']['phase_done'] == {}
    # повторный запуск: хвост убран, прогресс сохранён, продолжаем с места
    name, parsed, state, out_dir = store.open_batch('pack')
    assert not list(out_dir.glob('state.json.*.tmp'))
    assert state['rows']['001']['status'] == 'placed' and state['rows']['001']['attempts'] == 1
    assert state['rows']['001']['files'] == {'1': ['rows/001/p1_1.png']}
    assert S.pending(state, 1) == ['002'] and S.pending(state, 2) == ['001', '002']
    assert S.skip(state['rows']['001'], 1) and not S.skip(state['rows']['001'], 2)
    assert S.skip(state['rows']['003'], 2)  # ошибка разбора — всегда пропуск


def test_fixed_row_returns_to_parsed(roots):
    name, parsed, state, out_dir = store.open_batch('pack')
    assert state['rows']['004']['status'] == 'error'
    from PIL import Image
    Image.new('RGB', (8, 8)).save(roots / 'input' / 'auto' / 'pack' / 'missing.png')
    state = store.open_batch('pack')[2]
    assert state['rows']['004']['status'] == 'parsed' and state['rows']['004']['reason'] == ''
    # ошибка, поставленная фазой (не разбором), при повторном сведении не сбрасывается
    S.mark(out_dir, '004', 2, status='error', reason='кадр не прошёл проверку')
    state = store.open_batch('pack')[2]
    assert state['rows']['004']['status'] == 'error' and state['rows']['004']['reason'] == 'кадр не прошёл проверку'


def test_mark_rejects_unknown(roots):
    out_dir = store.open_batch('pack')[3]
    with pytest.raises(ValueError):
        S.mark(out_dir, '001', 1, status='finished')
    with pytest.raises(KeyError):
        S.mark(out_dir, '999', 1)


def test_resolve_variants(roots):
    assert store.resolve('pack')[1] == roots / 'input' / 'auto' / 'pack'
    data = json.dumps({'name': 'pack', 'dir': str(roots / 'input/auto/pack'), 'out_dir': str(roots / 'output/auto/pack')})
    assert store.resolve(data) == ('pack', roots / 'input/auto/pack', roots / 'output/auto/pack')
    with pytest.raises(ValueError):
        store.resolve(json.dumps({'name': 'pack', 'dir': 'X', 'out_dir': 'Y'}))
    assert store.resolve(str(FIXTURE))[1] == FIXTURE
    for bad in ('', '../pack', 'a/b'):
        with pytest.raises(ValueError):
            store.resolve(bad)


def test_upload_targets_and_limit(roots, monkeypatch):
    assert store.target_path('new_20261117_1030', 'poses/p.png') == (roots / 'input/auto/new_20261117_1030/poses/p.png').resolve()
    for bad in ('../x.png', '/abs/x.png', 'C:\\x.png', 'a/../../x.png'):
        with pytest.raises(ValueError):
            store.target_path('new', bad)
    with pytest.raises(ValueError):
        store.target_path('../new', 'x.png')
    with pytest.raises(ValueError):
        store.target_path('new', 'run.exe')
    monkeypatch.setattr(store, 'MAX_FILE', 10)
    target = store.target_path('new', 'big.png')
    with pytest.raises(OverflowError):
        store.write_stream(target, [b'12345', b'67890', b'x'])
    assert not target.exists() and not list(target.parent.glob('*.part'))
    assert store.write_stream(target, [b'12345']) == 5 and target.read_bytes() == b'12345'


def test_upload_route(roots, monkeypatch):
    aiohttp = pytest.importorskip('aiohttp')
    from aiohttp import web, FormData
    from aiohttp.test_utils import TestClient, TestServer
    from auto import routes

    async def run():
        app = web.Application()
        table = web.RouteTableDef()
        routes.register(table)
        app.add_routes(table)
        async with TestClient(TestServer(app)) as client:
            form = FormData()
            form.add_field('batch', 'up_20261117_1030')
            for rel in ('пакет.csv', '_пакет.txt', 'room.png', 'pose.png', 'extra.png', 'poses/pose2.png'):
                form.add_field('path', rel)
                form.add_field('file', (FIXTURE / rel).read_bytes(), filename=Path(rel).name)
            r = await client.post('/3da/auto/upload', data=form)
            ok = await r.json()
            assert r.status == 200 and ok['ok'] and len(ok['saved']) == 6
            bad = FormData()
            bad.add_field('batch', 'up_20261117_1030')
            bad.add_field('path', '../evil.png')
            bad.add_field('file', b'x', filename='evil.png')
            r = await client.post('/3da/auto/upload', data=bad)
            assert r.status == 400 and 'вне папки' in (await r.json())['errors'][0]
            monkeypatch.setattr(store, 'MAX_FILE', 4)
            big = FormData()
            big.add_field('batch', 'up_20261117_1030')
            big.add_field('file', b'0123456789', filename='big.png')
            r = await client.post('/3da/auto/upload', data=big)
            assert r.status == 413
            monkeypatch.setattr(store, 'MAX_FILE', 50 * 1024 * 1024)
            r = await client.get('/3da/auto/state', params={'batch': 'up_20261117_1030'})
            st = await r.json()
            assert st['rows_count'] == 5 and [x['status'] for x in st['rows']][:2] == ['parsed', 'parsed']
            r = await client.get('/3da/auto/batches')
            assert (await r.json())['batches'][0]['batch'] == 'up_20261117_1030'
            r = await client.get('/3da/auto/zip', params={'batch': 'up_20261117_1030'})
            assert r.status == 200 and (await r.read())[:2] == b'PK'
            r = await client.get('/3da/auto/state', params={'batch': '../x'})
            assert r.status == 400
            # чужие папки: JSON с dir/out_dir вне input/output и абсолютный путь — только 400
            outside = roots / 'outside'
            (outside / 'pk').mkdir(parents=True)
            (outside / 'pk' / 'secret.txt').write_text('x', encoding='utf-8')
            evil = json.dumps({'name': 'pk', 'dir': str(outside / 'pk'), 'out_dir': str(outside)})
            evil_in = json.dumps({'name': 'up_20261117_1030', 'dir': str(roots / 'input/auto/up_20261117_1030'),
                                  'out_dir': str(outside)})
            for value in (evil, evil_in, str(outside / 'pk'), str(roots / 'input/auto/up_20261117_1030')):
                for route in ('/3da/auto/state', '/3da/auto/zip'):
                    r = await client.get(route, params={'batch': value})
                    assert r.status == 400, (route, value)
    asyncio.run(run())
    assert not (roots / 'input' / 'evil.png').exists() and not (roots / 'input' / 'auto' / 'evil.png').exists()
    assert (roots / 'input/auto/up_20261117_1030/poses/pose2.png').is_file()
    assert sorted(p.relative_to(roots / 'outside').as_posix() for p in (roots / 'outside').rglob('*')) == ['pk', 'pk/secret.txt']


def test_resolve_confines_dirs(roots, tmp_path):
    """JSON из узла: выход только output/auto/<имя>, вход внутри input/auto (или явная папка — только из узла)."""
    other = tmp_path / 'other'
    other.mkdir()
    good = store.resolve(json.dumps({'name': 'pk', 'dir': str(roots / 'input/auto/pk'), 'out_dir': str(roots / 'output/auto/pk')}))
    assert good[2] == roots / 'output/auto/pk'
    for out in (other, roots / 'output/auto', roots / 'output/auto/pk/..'):
        with pytest.raises(ValueError):
            store.resolve(json.dumps({'name': 'pk', 'dir': str(roots / 'input/auto/pk'), 'out_dir': str(out)}))
    with pytest.raises(ValueError):
        store.resolve(json.dumps({'name': '../pk', 'dir': str(roots / 'input/auto/pk'), 'out_dir': str(roots / 'output/auto/pk')}))
    with pytest.raises(ValueError):
        store.resolve(json.dumps({'name': 'pk', 'dir': str(other), 'out_dir': str(roots / 'output/auto/pk')}), allow_path=False)
    with pytest.raises(ValueError):
        store.resolve(str(other), allow_path=False)
    for bad in ('', '../x', 'a/b', str(other), '{"name": "pk"}'):
        with pytest.raises(ValueError):
            store.resolve_name(bad)
    assert not list(other.iterdir())
