"""Отчёт пакета: исходная таблица + `статус`, `причина`, `файл`; раскладка итогов и ZIP.

`output/auto/<пакет>/готово/` — финальные кадры строк со статусом done,
`output/auto/<пакет>/художнику/<строка>/` — файлы строки и `причина.txt` (status artist),
`отчёт.xlsx` (без openpyxl — `отчёт.csv`), `<пакет>.zip` — всё, кроме state.json и временных.
"""
import csv
import os
import shutil
import zipfile
from pathlib import Path

from .batch import IMAGE_EXT, MOD_KEYS, MOD_RU
from .state import STATE_FILE, STATUS_RU, counts

DONE_DIR, ARTIST_DIR = 'готово', 'художнику'
REPORT_NAME = 'отчёт'


def final_files(row_state):
    """Файлы последней фазы, в которой что-то сохранено."""
    files = row_state.get('files', {})
    phases = sorted((k for k, v in files.items() if v), key=lambda k: (not k.isdigit(), int(k) if k.isdigit() else 0, k))
    return list(files[phases[-1]]) if phases else []


def all_files(row_state):
    return [f for _, v in sorted(row_state.get('files', {}).items()) for f in v]


def reset_dir(path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def layout(out_dir, parsed, state):
    """Разложить итоги по `готово/` и `художнику/`. → {row_id: путь для колонки «файл»}."""
    out_dir = Path(out_dir)
    done_dir, artist_dir = out_dir / DONE_DIR, out_dir / ARTIST_DIR
    reset_dir(done_dir)
    reset_dir(artist_dir)
    where = {}
    rows = {r['row_id']: r for r in parsed['rows']}
    for rid in state.get('order', []):
        st = state['rows'][rid]
        if st['status'] == 'done':
            files = [f for f in final_files(st) if Path(f).suffix.lower() in IMAGE_EXT] or final_files(st)
            names = []
            for k, rel in enumerate(files, 1):
                src = out_dir / rel
                if not src.is_file():
                    continue
                name = f'{rid}{src.suffix}' if len(files) == 1 else f'{rid}_{k}{src.suffix}'
                shutil.copy2(src, done_dir / name)
                names.append(f'{DONE_DIR}/{name}')
            where[rid] = ', '.join(names)
        elif st['status'] == 'artist':
            dst = artist_dir / rid
            dst.mkdir()
            for rel in all_files(st):
                if (out_dir / rel).is_file():
                    shutil.copy2(out_dir / rel, dst / Path(rel).name)
            row = rows.get(rid, {})
            text = [f'Строка: {rid}', f'Причина: {st.get("reason") or "—"}', f'Промпт: {row.get("prompt", "")}']
            text += [f'Мод {MOD_RU[k]}: {row.get("mods", {}).get(k) or "—"}' for k in MOD_KEYS]
            text += [f'Фото {k}: {Path(p).name if p else "—"}' for k, p in enumerate(row.get('photos', []), 1)]
            (dst / 'причина.txt').write_text('\n'.join(text) + '\n', encoding='utf-8')
            where[rid] = f'{ARTIST_DIR}/{rid}/'
    return where


def table_rows(parsed, state, where):
    header = list(parsed['header']) + ['статус', 'причина', 'файл']
    by_src = {r['src']: r for r in parsed['rows']}
    out = []
    for i, values in enumerate(parsed['source']):
        row = by_src.get(i)
        st = state['rows'].get(row['row_id']) if row else None
        status = STATUS_RU.get(st['status'], st['status']) if st else ''
        out.append(list(values) + [status, (st or {}).get('reason', ''), where.get(row['row_id'], '') if row else ''])
    return header, out


def write_report(out_dir, header, rows):
    """XLSX, если есть openpyxl, иначе CSV (UTF-8 с BOM, `;` — открывается в Excel)."""
    out_dir = Path(out_dir)
    try:
        import openpyxl
    except ImportError:
        openpyxl = None
    for old in (out_dir / f'{REPORT_NAME}.xlsx', out_dir / f'{REPORT_NAME}.csv'):
        if old.exists():
            old.unlink()
    if openpyxl is not None:
        path = out_dir / f'{REPORT_NAME}.xlsx'
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = 'отчёт'
        ws.append(header)
        for row in rows:
            ws.append(row)
        ws.freeze_panes = 'A2'
        wb.save(path)
    else:
        path = out_dir / f'{REPORT_NAME}.csv'
        with open(path, 'w', encoding='utf-8-sig', newline='') as f:
            w = csv.writer(f, delimiter=';')
            w.writerow(header)
            w.writerows(rows)
    return path


def make_zip(out_dir, name):
    """`<пакет>.zip` из папки пакета: без state.json, временных файлов и самого архива."""
    out_dir = Path(out_dir)
    path = out_dir / f'{name}.zip'
    tmp = out_dir / f'{name}.zip.tmp'
    with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(out_dir.rglob('*')):
            rel = p.relative_to(out_dir).as_posix()
            if (not p.is_file() or p == path or p == tmp or rel.startswith(STATE_FILE)
                    or p.suffix.lower() in ('.tmp', '.part')):
                continue
            zf.write(p, f'{name}/{rel}')
    os.replace(tmp, path)
    return path


def summary_text(state):
    c = counts(state)
    working = sum(v for k, v in c.items() if k not in ('done', 'artist', 'error'))
    text = f'готово {c["done"]} / художнику {c["artist"]} / ошибки {c["error"]}'
    return text + (f' / в работе {working}' if working else '')


def build(out_dir, parsed, state, name):
    """Отчёт + раскладка + ZIP. → {summary, counts, report, zip}."""
    where = layout(out_dir, parsed, state)
    header, rows = table_rows(parsed, state, where)
    report = write_report(out_dir, header, rows)
    archive = make_zip(out_dir, name)
    return {'summary': summary_text(state), 'counts': counts(state), 'report': str(report), 'zip': str(archive)}
