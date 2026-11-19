"""CPU-тесты безопасной выдачи на синтетическом пакете."""
import csv
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import openpyxl
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from auto import secure_delivery as delivery, secure_report, store  # noqa: E402


@pytest.fixture
def pack(tmp_path):
    store.set_roots(tmp_path / 'input' / 'auto', tmp_path / 'output' / 'auto')
    folder = tmp_path / 'input' / 'auto' / 'pack'
    folder.mkdir(parents=True)
    Image.new('RGB', (2, 2), 'red').save(folder / 'a.png')
    Image.new('RGB', (2, 2), 'blue').save(folder / 'b.png')
    def write_rows(first='=1+1'):
        with (folder / 'batch.csv').open('w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['№', 'photo1', 'photo2', 'prompt', '=header'])
            writer.writerows([
                ['1', 'a.png', 'b.png', first, '=2+2'],
                ['2', 'a.png', 'b.png', 'safe', '+external'],
                ['3', 'missing.png', 'b.png', 'safe', '@external'],
            ])
    write_rows()
    source, _ = delivery._batch_data('pack')
    root = delivery._root('pack')
    payload = io.BytesIO()
    Image.new('RGB', (3, 3), 'green').save(payload, format='PNG')
    png = payload.getvalue()
    digest = hashlib.sha256(png).hexdigest()
    path = root / 'quarantine' / '001'
    path.mkdir(parents=True)
    (path / f'0-{digest}.png').write_bytes(png)
    state = {'rows': {
        '001': {'identity': delivery._identity(source['rows'][0]), 'status': 'done',
                'reason': '=reason', 'attempt': 0, 'approved': digest, 'frames': {'0': digest}},
        '002': {'identity': delivery._identity(source['rows'][1]), 'status': 'artist',
                'reason': '=artist', 'attempt': 0, 'frames': {}},
    }}
    delivery._write(root, state)
    yield folder, root, write_rows, png
    store._roots.clear()


def test_mixed_statuses_xlsx_and_whitelist(pack):
    _, root, _, png = pack
    (root / 'untrusted.png').write_bytes(b'secret')
    (root / 'delivery.zip').write_bytes(b'old archive')
    result = secure_report.build('pack')
    assert result['rows'] == 3 and result['approved'] == 1
    with zipfile.ZipFile(io.BytesIO(result['zip'])) as archive:
        assert set(archive.namelist()) == {'отчёт.xlsx', 'manifest.json', 'готово/001.png'}
        assert archive.read('готово/001.png') == png
        assert archive.read('отчёт.xlsx') == result['xlsx']
        manifest = json.loads(archive.read('manifest.json'))
        assert [x['status'] for x in manifest] == ['done', 'artist', 'error']
        assert 'не найден' in manifest[2]['reason']
    wb = openpyxl.load_workbook(io.BytesIO(result['xlsx']))
    ws = wb.active
    assert ws.max_row == 4
    assert [ws.cell(row, 6).value for row in (2, 3, 4)] == ['готово', 'художнику', 'ошибка']
    assert ws['D2'].value == '=1+1' and ws['D2'].data_type == 's'
    assert ws['E1'].value == '=header' and ws['E1'].data_type == 's'
    assert ws['G2'].value == '=reason' and ws['G2'].data_type == 's'
    assert wb.sheetnames == ['отчёт', 'ошибки пакета']
    assert all(cell.font.name == 'Arial' and cell.data_type != 'f'
               for sheet in wb for row in sheet for cell in row)
    assert not (root / 'отчёт.xlsx').exists()


def test_changed_source_rejects_instead_of_stale_output(pack):
    folder, root, write_rows, _ = pack
    secure_report.build('pack')
    write_rows('changed')
    with pytest.raises(ValueError, match='изменилось'):
        secure_report.build('pack')
    assert not (root / 'отчёт.xlsx').exists()
    write_rows()
    (folder / 'a.png').write_bytes(b'modified photo')
    with pytest.raises(ValueError, match='изменилось'):
        secure_report.build('pack')


def test_source_change_during_build_rejected(pack, monkeypatch):
    folder, _, write_rows, _ = pack
    original_xlsx = secure_report._xlsx

    def change_source(*args, **kwargs):
        result = original_xlsx(*args, **kwargs)
        write_rows('changed during xlsx generation')
        return result

    monkeypatch.setattr(secure_report, '_xlsx', change_source)
    with pytest.raises(ValueError, match='изменился во время сборки'):
        secure_report.build('pack')
    assert (folder / 'batch.csv').is_file()


def test_tampered_approved_png_rejected(pack):
    _, root, _, _ = pack
    next((root / 'quarantine' / '001').iterdir()).write_bytes(b'tampered')
    with pytest.raises(ValueError, match='изменён'):
        secure_report.build('pack')


def test_artist_frame_is_held_with_or_without_retry(pack):
    _, root, _, png = pack
    state = delivery._read(root)
    row = state['rows']['002']
    digest = hashlib.sha256(png).hexdigest()
    folder = root / 'quarantine' / '002'
    folder.mkdir(parents=True)
    (folder / f'0-{digest}.png').write_bytes(png)
    row['frames'] = {'0': digest}
    delivery._write(root, state)
    result = secure_report.build('pack')
    with zipfile.ZipFile(io.BytesIO(result['zip'])) as archive:
        assert 'художнику/002.png' not in archive.namelist()
        artist = json.loads(archive.read('manifest.json'))[1]
        assert artist['status'] == 'artist' and 'file' not in artist
    wb = openpyxl.load_workbook(io.BytesIO(result['xlsx']))
    assert wb.active['H3'].value in ('', None)
    wb.close()
    # Если повторная генерация не дала кадра, старый retry не выдаётся.
    row['attempt'] = 1
    delivery._write(root, state)
    with zipfile.ZipFile(io.BytesIO(secure_report.build('pack')['zip'])) as archive:
        assert 'художнику/002.png' not in archive.namelist()


def test_artist_tamper_invalid_digest_and_attempt_rejected(pack):
    _, root, _, png = pack
    state = delivery._read(root)
    row = state['rows']['002']
    digest = hashlib.sha256(png).hexdigest()
    folder = root / 'quarantine' / '002'
    folder.mkdir(parents=True)
    path = folder / f'0-{digest}.png'
    path.write_bytes(b'tampered')
    row['frames'] = {'0': digest}
    delivery._write(root, state)
    with pytest.raises(ValueError, match='изменён'):
        secure_report.build('pack')
    row['frames'] = {'0': '../invalid'}
    delivery._write(root, state)
    with pytest.raises(ValueError, match='SHA-256'):
        secure_report.build('pack')
    row['frames'] = {}
    row['attempt'] = 2
    delivery._write(root, state)
    with pytest.raises(ValueError, match='попытка'):
        secure_report.build('pack')


def test_pending_and_extra_state_rejected(pack):
    _, root, _, _ = pack
    state = delivery._read(root)
    state['rows']['001']['status'] = 'pending'
    delivery._write(root, state)
    with pytest.raises(ValueError, match='не завершена'):
        secure_report.build('pack')
    state['rows']['001']['status'] = 'done'
    state['rows']['003'] = {'status': 'artist'}
    delivery._write(root, state)
    with pytest.raises(ValueError, match='ошибкой'):
        secure_report.build('pack')
