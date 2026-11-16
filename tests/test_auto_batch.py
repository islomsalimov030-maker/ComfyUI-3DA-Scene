"""Разбор пакета: таблица xlsx/csv (рус/англ заголовки), txt-вид с #модами, умолчания, ошибки строк, пути.
Запуск: python -m pytest tests/test_auto_batch.py -q"""
import shutil
import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from auto import batch as B  # noqa: E402

FIXTURE = ROOT / 'tests/fixtures/batch_table'


def png(path, color=(128, 128, 128)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new('RGB', (8, 8), color).save(path)


def by_id(parsed):
    return {r['row_id']: r for r in parsed['rows']}


def test_table_csv_ru_fixture():
    parsed = B.parse_batch(FIXTURE)
    rows = by_id(parsed)
    assert parsed['kind'] == 'table' and parsed['table'] == 'пакет.csv'
    assert list(rows) == ['001', '002', '003', '004', '005']
    r1 = rows['001']
    assert r1['errors'] == []
    assert Path(r1['photos'][0]).name == 'room.png' and Path(r1['photos'][1]).name == 'pose.png'
    assert r1['photos'][2:] == [None, None, None]
    assert r1['prompt'] == 'a woman stands by the window'
    # пустые ячейки → умолчания из _пакет.txt; умолчания нет → пусто
    assert r1['mods'] == {'character': 'character_a', 'face': '', 'prop': 'prop_a'}
    r2 = rows['002']
    assert r2['errors'] == [] and Path(r2['photos'][1]).as_posix().endswith('poses/pose2.png')
    assert Path(r2['photos'][2]).name == 'extra.png'
    assert r2['mods']['face'] == 'face_b' and r2['mods']['character'] == 'character_a'
    assert rows['003']['errors'] == ['нет фото 2 (поза)']
    assert rows['004']['errors'] == ['фото 2: файл не найден «missing.png»']
    # имя без учёта регистра и без расширения находится; пустой промпт — ошибка
    r5 = rows['005']
    assert Path(r5['photos'][0]).name == 'room.png' and Path(r5['photos'][1]).name == 'pose.png'
    assert r5['errors'] == ['пустой промпт']


def test_table_csv_en_comma_cp1251(tmp_path):
    png(tmp_path / 'a.png')
    png(tmp_path / 'b.png')
    (tmp_path / 'batch.csv').write_bytes(
        'ID,Photo1,Photo2,Prompt,Character,Face,Prop\n'
        '7,a.png,b.png,"a woman stands by the window, smiling",char_x,,prop_y\n'
        ',a.png,b.png,женщина сидит на стуле,,,\n'.encode('cp1251'))
    rows = B.parse_batch(tmp_path)['rows']
    assert [r['row_id'] for r in rows] == ['007', '002']
    assert rows[0]['prompt'] == 'a woman stands by the window, smiling'
    assert rows[0]['mods'] == {'character': 'char_x', 'face': '', 'prop': 'prop_y'}
    assert rows[1]['prompt'] == 'женщина сидит на стуле' and rows[1]['errors'] == []


def test_table_xlsx_mixed_headers(tmp_path):
    openpyxl = pytest.importorskip('openpyxl')
    png(tmp_path / 'room.png')
    png(tmp_path / 'img' / 'pose.png')
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(['№', 'Фото1', 'ФОТО 2', 'Промпт', 'Мод_Лицо', 'photo3', 'Комментарий'])
    ws.append([1, 'room.png', 'img/pose.png', 'a woman stands by the window', 'face_x', None, 'любая колонка'])
    ws.append([2.0, 'room.png', 'img\\pose.png', 'a woman sits on the sofa', None, 'nope.png', None])
    wb.save(tmp_path / 'пакет.xlsx')
    (tmp_path / '_batch.txt').write_text('#face: face_default\n', encoding='utf-8')
    parsed = B.parse_batch(tmp_path)
    rows = by_id(parsed)
    assert parsed['kind'] == 'table' and parsed['header'][-1] == 'Комментарий'
    assert rows['001']['mods']['face'] == 'face_x' and rows['001']['errors'] == []
    assert rows['002']['mods']['face'] == 'face_default'
    assert Path(rows['002']['photos'][1]).name == 'pose.png'
    assert rows['002']['errors'] == ['фото 3: файл не найден «nope.png»']


def test_xlsx_without_openpyxl(tmp_path, monkeypatch):
    (tmp_path / 'пакет.xlsx').write_bytes(b'PK')
    monkeypatch.setitem(sys.modules, 'openpyxl', None)
    parsed = B.parse_batch(tmp_path)
    assert parsed['rows'] == []
    assert any('сохраните таблицу как CSV' in e for e in parsed['errors'])


def test_txt_kind_with_mods_and_defaults(tmp_path):
    png(tmp_path / '001_1.png')
    png(tmp_path / '001_2.jpg')
    png(tmp_path / '001_3.webp')
    (tmp_path / '001.txt').write_text('#лицо: face_c\n#Character: char_x\n\na woman stands by the window\n#не мод, часть текста\n',
                                      encoding='utf-8')
    png(tmp_path / '002_1.png')
    (tmp_path / '002.txt').write_text('a woman sits on a chair', encoding='utf-8')
    png(tmp_path / '003_1.png')
    png(tmp_path / '003_2.png')
    (tmp_path / '_пакет.txt').write_text('#предмет: prop_d\n#лицо: face_default\n', encoding='utf-8')
    parsed = B.parse_batch(tmp_path)
    rows = by_id(parsed)
    assert parsed['kind'] == 'txt' and list(rows) == ['001', '002', '003']
    r1 = rows['001']
    assert r1['errors'] == [] and r1['photos'][2].endswith('001_3.webp') and r1['photos'][3] is None
    assert r1['prompt'] == 'a woman stands by the window\n#не мод, часть текста'
    assert r1['mods'] == {'character': 'char_x', 'face': 'face_c', 'prop': 'prop_d'}
    assert rows['002']['errors'] == ['нет фото 2 (поза)']
    assert rows['002']['mods']['face'] == 'face_default'
    assert rows['003']['errors'] == ['нет файла промпта 003.txt', 'пустой промпт']
    assert parsed['source'][0][0] == '1' and parsed['header'][0] == '№'


def test_path_outside_batch_rejected(tmp_path):
    base = tmp_path / 'pack'
    png(base / 'room.png')
    png(tmp_path / 'outside.png')
    (base / 'пакет.csv').write_text('n;фото1;фото2;промпт\n1;room.png;../outside.png;a woman stands by the window\n',
                                    encoding='utf-8')
    row = B.parse_batch(base)['rows'][0]
    assert row['photos'][1] is None
    assert row['errors'] == ['фото 2: путь вне папки пакета «../outside.png»']
    for bad in ('../x.png', 'a/../../x.png', '/etc/x.png', 'C:/x.png', 'C:x.png', '..\\x.png', ''):
        with pytest.raises(ValueError):
            B.safe_rel(base, bad)
    assert B.safe_rel(base, 'sub\\a.png') == (base / 'sub' / 'a.png').resolve()
    for bad in ('a/b', '..', '.hidden', '', 'a\\b'):
        with pytest.raises(ValueError):
            B.safe_name(bad)
    assert B.safe_name('Folder_20261117_1030') == 'Folder_20261117_1030'


def test_duplicate_numbers_and_summary(tmp_path):
    shutil.copytree(FIXTURE, tmp_path / 'p')
    with open(tmp_path / 'p' / 'пакет.csv', 'a', encoding='utf-8') as f:
        f.write('1;room.png;pose.png;;a woman stands by the door;;;\n')
    parsed = B.parse_batch(tmp_path / 'p')
    rows = by_id(parsed)
    assert '001_2' in rows and rows['001_2']['errors'] == ['номер строки повторяется: 1']
    text = B.summary(parsed, 'p')
    assert 'Строк: 6, с ошибками: 4' in text
    assert 'Мод персонаж: character_a (6)' in text and 'Мод лицо: face_b (1)' in text


def test_empty_folder(tmp_path):
    parsed = B.parse_batch(tmp_path)
    assert parsed['rows'] == [] and parsed['errors']
    assert B.parse_batch(tmp_path / 'nope')['errors'] == ['папка пакета не найдена: nope']
