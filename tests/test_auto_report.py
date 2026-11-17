"""Отчёт, раскладка готово/художнику, ZIP; сквозной прогон узлов 3DA/auto по фазам (без сервера и GPU).
Запуск: python -m pytest tests/test_auto_report.py -q"""
import json
import shutil
import sys
import zipfile
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from auto import report as R, state as S, store  # noqa: E402

FIXTURE = ROOT / 'tests/fixtures/batch_table'


@pytest.fixture
def roots(tmp_path):
    store.set_roots(tmp_path / 'input' / 'auto', tmp_path / 'output' / 'auto')
    shutil.copytree(FIXTURE, tmp_path / 'input' / 'auto' / 'pack')
    yield tmp_path
    store._roots.clear()


def fake_result(out_dir, rid, phase, n=1):
    files = []
    for k in range(1, n + 1):
        p = out_dir / 'rows' / rid / f'p{phase}_{k}.png'
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.new('RGB', (8, 8), (10 * k, 20, 30)).save(p)
        files.append(p.relative_to(out_dir).as_posix())
    return files


def read_report(path):
    if path.suffix == '.xlsx':
        import openpyxl
        wb = openpyxl.load_workbook(path)
        return [[c if c is not None else '' for c in row] for row in wb.active.iter_rows(values_only=True)]
    import csv
    with open(path, encoding='utf-8-sig') as f:
        return list(csv.reader(f, delimiter=';'))


def run_build(roots):
    name, parsed, state, out_dir = store.open_batch('pack')
    S.mark(out_dir, '001', 1, status='placed', files=fake_result(out_dir, '001', 1))
    S.mark(out_dir, '001', 2, status='done', files=fake_result(out_dir, '001', 2))
    S.mark(out_dir, '002', 1, status='artist', reason='опора не найдена', files=fake_result(out_dir, '002', 1, n=2))
    return store.build_report('pack'), out_dir


def test_report_layout_zip(roots):
    pytest.importorskip('openpyxl')
    result, out_dir = run_build(roots)
    assert result['summary'] == 'готово 1 / художнику 1 / ошибки 3'
    assert (out_dir / 'готово' / '001.png').is_file()
    assert sorted(p.name for p in (out_dir / 'художнику' / '002').iterdir()) == ['p1_1.png', 'p1_2.png', 'причина.txt']
    assert 'опора не найдена' in (out_dir / 'художнику' / '002' / 'причина.txt').read_text(encoding='utf-8')
    table = read_report(Path(result['report']))
    assert Path(result['report']).name == 'отчёт.xlsx'
    assert table[0][-3:] == ['статус', 'причина', 'файл'] and table[0][0] == '№'
    assert table[1][-3:] == ['готово', '', 'готово/001.png']
    assert table[2][-3:] == ['художнику', 'опора не найдена', 'художнику/002/']
    assert table[3][-3:-1] == ['ошибка', 'нет фото 2 (поза)']
    names = zipfile.ZipFile(result['zip']).namelist()
    assert 'pack/готово/001.png' in names and 'pack/отчёт.xlsx' in names and 'pack/художнику/002/причина.txt' in names
    assert not any('state.json' in n or n.endswith(('.tmp', '.zip')) for n in names)
    # повторная сборка: раскладка пересобирается, ZIP не вкладывается сам в себя
    S.mark(out_dir, '002', 2, status='done', files=fake_result(out_dir, '002', 2))
    result = store.build_report('pack')
    assert result['summary'] == 'готово 2 / художнику 0 / ошибки 3'
    assert not (out_dir / 'художнику' / '002').exists() and (out_dir / 'готово' / '002.png').is_file()


def test_report_csv_without_openpyxl(roots, monkeypatch):
    monkeypatch.setitem(sys.modules, 'openpyxl', None)
    result, out_dir = run_build(roots)
    assert Path(result['report']).name == 'отчёт.csv'
    table = read_report(Path(result['report']))
    assert table[1][-3:] == ['готово', '', 'готово/001.png']
    assert 'pack/отчёт.csv' in zipfile.ZipFile(result['zip']).namelist()


def test_zip_rebuilt_when_stale(roots):
    path = store.zip_path('pack')
    assert path.is_file() and path.name == 'pack.zip'
    first = path.stat().st_mtime_ns
    assert store.zip_path('pack').stat().st_mtime_ns == first   # состояние не менялось — тот же архив


def test_nodes_phase_driver_order(roots):
    """Как драйвер: все строки фазы 1, потом все строки фазы 2; повторный запуск пропускает пройденное."""
    torch = pytest.importorskip('torch')
    from auto import nodes as Nd
    batch_out = Nd.AutoBatch().execute('pack', '1,2')
    batch_json, rows_count, summary = batch_out['result']
    assert rows_count == 5 and 'Строк: 5, с ошибками: 3' in summary and 'Мод лицо: face_b (1)' in summary
    data = json.loads(batch_json)
    assert data['phases'] == [1, 2] and data['name'] == 'pack'
    order = []
    for phase in (1, 2):
        for i in range(rows_count):
            row = Nd.AutoRow().execute(batch_json, i, phase)['result']
            images, n_images, prompt, mc, mf, mp, rid, skip = row[:5], *row[5:12]
            assert all(tuple(im.shape[1:]) in ((12, 16, 3), (64, 64, 3)) for im in images)
            if skip:
                continue
            order.append((phase, rid))
            assert n_images >= 2 and prompt and mc == 'character_a'
            status = 'done' if phase == 2 else 'placed'
            out = torch.cat([images[0], images[1]]) if rid == '001' else images[0]
            info = Nd.AutoSave().execute(batch_json, i, phase, images=out, text=prompt, status=status)['result'][0]
            assert 'сохранено' in info
    assert order == [(1, '001'), (1, '002'), (2, '001'), (2, '002')]
    # строка с ошибкой: чёрные 64×64 вместо отсутствующего фото, skip
    row = Nd.AutoRow().execute(batch_json, 2, 1)['result']
    assert row[11] is True and tuple(row[1].shape) == (1, 64, 64, 3) and row[5] == 1
    # повтор той же задачи — пропуск, файлы не трогаются
    assert 'пропуск' in Nd.AutoSave().execute(batch_json, 0, 1, text='x')['result'][0]
    rep = Nd.AutoReport().execute(batch_json)
    text, zip_path = rep['result']
    assert text.startswith('готово 2 / художнику 0 / ошибки 3') and Path(zip_path).is_file()
    out_dir = Path(data['out_dir'])
    assert sorted(p.name for p in (out_dir / 'готово').iterdir()) == ['001_1.png', '001_2.png', '002.png']
    assert sorted(S.load(out_dir)['rows']['001']['phase_done']) == ['1', '2']
    # пакет без строк/без имени — текст на узле, без исключения
    assert Nd.AutoBatch().execute('', '1')['result'][1] == 0
    assert Nd.AutoRow().execute('{"error": "x"}', 0, 1)['result'][11] is True
