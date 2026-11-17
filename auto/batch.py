"""Разбор папки пакета (один персонаж) в список строк.

Два вида входа, оба дают один и тот же список `rows`:
- таблица `*.xlsx` / `*.csv` + фото (основной);
- txt: `001_1.png` (локация), `001_2.jpg` (поза), `001_3..5.*` (доп.), `001.txt` (промпт с `#модами`).
Умолчания пакета — `_пакет.txt` / `_batch.txt` (`#ключ: значение`).
Ошибка строки не роняет пакет: строка получает `errors` (по-русски), остальные идут.
Без torch и ComfyUI — модуль используется и в тестах, и в маршрутах сервера.
"""
import csv
import io
import re
from pathlib import Path, PurePosixPath

IMAGE_EXT = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
TABLE_EXT = ('.xlsx', '.csv')
UPLOAD_EXT = IMAGE_EXT + TABLE_EXT + ('.txt',)
MOD_KEYS = ('character', 'face', 'prop')
MOD_RU = {'character': 'персонаж', 'face': 'лицо', 'prop': 'предмет'}
DEFAULTS_FILES = ('_пакет.txt', '_batch.txt')
N_PHOTOS = 5

# Заголовки таблицы и ключи `#` в txt: без регистра, ё=е, без пробелов/подчёркиваний/дефисов/точек и скобок.
ALIASES = {'n': ('№', 'n', 'id', 'номер', 'no'),
           'prompt': ('промпт', 'prompt'),
           'character': ('модперсонаж', 'character', 'персонаж', 'modcharacter'),
           'face': ('модлицо', 'face', 'лицо', 'modface'),
           'prop': ('модпредмет', 'prop', 'предмет', 'modprop')}
for _k in range(1, N_PHOTOS + 1):
    ALIASES[f'photo{_k}'] = (f'фото{_k}', f'photo{_k}')
KEY_OF = {alias: key for key, names in ALIASES.items() for alias in names}

BATCH_NAME = re.compile(r'^[\w\-. ()]{1,120}$')


def norm_key(text):
    text = re.sub(r'\(.*?\)', '', str(text or '')).lower().replace('ё', 'е')
    return re.sub(r'[\s_\-.:#]+', '', text)


def safe_name(name):
    """Имя пакета: одна папка, без разделителей и `..`. Иначе ValueError."""
    name = str(name or '').strip()
    if not BATCH_NAME.match(name) or name.strip('. ') == '' or name.startswith('.'):
        raise ValueError(f'недопустимое имя пакета «{name}»')
    return name


def safe_rel(base, rel):
    """Путь `rel` внутри `base`: без `..`, без абсолютных путей и дисков. Иначе ValueError."""
    raw = str(rel or '').replace('\\', '/').strip()
    parts = [p for p in raw.split('/') if p not in ('', '.')]
    if not parts or raw.startswith('/') or ':' in raw or any(p == '..' for p in parts):
        raise ValueError(f'путь вне папки пакета «{rel}»')
    base = Path(base).resolve()
    path = (base / PurePosixPath(*parts)).resolve()
    if path != base and base not in path.parents:
        raise ValueError(f'путь вне папки пакета «{rel}»')
    return path


def cell_text(value):
    if value is None:
        return ''
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value).strip()


def parse_hash_lines(text, all_lines=False):
    """`#ключ: значение` в начале текста → (моды, остальной текст). all_lines — файл умолчаний (`#` необязателен)."""
    mods, rest, head = {}, [], True
    for line in str(text or '').lstrip('﻿').splitlines():
        stripped = line.strip()
        if head and (stripped.startswith('#') or (all_lines and ':' in stripped)):
            match = re.match(r'#?\s*([^:]+):\s*(.*)$', stripped)
            key = KEY_OF.get(norm_key(match.group(1))) if match else None
            if key in MOD_KEYS:
                mods[key] = match.group(2).strip()
            continue
        if head and not stripped:
            continue
        head = all_lines
        rest.append(line)
    return mods, '\n'.join(rest).strip()


def read_text(path):
    data = Path(path).read_bytes()
    for enc in ('utf-8-sig', 'cp1251'):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode('utf-8', 'replace')


def read_table(path):
    """→ (заголовок, строки) как списки строк. xlsx без openpyxl — ValueError с подсказкой."""
    path = Path(path)
    if path.suffix.lower() == '.xlsx':
        try:
            import openpyxl
        except ImportError:
            raise ValueError('для xlsx нужен openpyxl — сохраните таблицу как CSV (UTF-8) и загрузите снова')
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        try:
            table = [[cell_text(v) for v in row] for row in wb.active.iter_rows(values_only=True)]
        finally:
            wb.close()
    else:
        text = read_text(path)
        # Разделитель — по строке заголовков (в промптах бывают запятые, в заголовках — нет).
        first = next((line for line in text.splitlines() if line.strip()), '')
        delimiter = max(';,\t', key=first.count) if any(c in first for c in ';,\t') else ','
        table = [[cell_text(v) for v in row] for row in csv.reader(io.StringIO(text), delimiter=delimiter)]
    table = [row for row in table if any(row)]
    if not table:
        return [], []
    width = max(len(r) for r in table)
    table = [row + [''] * (width - len(row)) for row in table]
    return table[0], table[1:]


class Files:
    """Файлы пакета: поиск без учёта регистра и без расширения (для имён из ячеек)."""
    def __init__(self, base):
        self.base = Path(base).resolve()
        self.lower = {}
        for p in self.base.rglob('*'):
            if p.is_file():
                self.lower.setdefault(p.relative_to(self.base).as_posix().lower(), p)

    def find(self, name):
        path = safe_rel(self.base, name)
        if path.is_file():
            return path
        rel = path.relative_to(self.base).as_posix().lower()
        if rel in self.lower:
            return self.lower[rel]
        if not Path(rel).suffix or Path(rel).suffix not in IMAGE_EXT:
            for ext in IMAGE_EXT:
                if rel + ext in self.lower:
                    return self.lower[rel + ext]
        return None


def load_defaults(base):
    for p in sorted(Path(base).iterdir()):
        if p.is_file() and p.name.lower() in DEFAULTS_FILES:
            return parse_hash_lines(read_text(p), all_lines=True)[0], p.name
    return {}, None


def row_id_of(n, ordinal):
    n = str(n or '').strip()
    if not n:
        return f'{ordinal:03d}'
    if n.isdigit():
        return n.zfill(3)
    return re.sub(r'[^\w\-]+', '_', n).strip('_') or f'{ordinal:03d}'


def make_row(row_id, n, photos, prompt, mods, defaults, errors, src):
    mods = {k: (mods.get(k) or defaults.get(k) or '').strip() for k in MOD_KEYS}
    if not photos[0] and not any(e.startswith('фото 1') for e in errors):
        errors.append('нет фото 1 (локация)')
    if not photos[1] and not any(e.startswith('фото 2') for e in errors):
        errors.append('нет фото 2 (поза)')
    if not str(prompt or '').strip():
        errors.append('пустой промпт')
    return {'row_id': row_id, 'n': n, 'photos': [str(p) if p else None for p in photos],
            'prompt': str(prompt or '').strip(), 'mods': mods, 'errors': errors, 'src': src}


def pick_table(base):
    tables = sorted(p for p in Path(base).iterdir()
                    if p.is_file() and p.suffix.lower() in TABLE_EXT and not p.name.startswith(('~$', '.', '_')))
    preferred = [p for p in tables if p.stem.lower() in ('пакет', 'batch')]
    return (preferred or tables)[0] if tables else None, tables


def parse_table(base, table, defaults, out):
    header, body = read_table(table)
    keys = [KEY_OF.get(norm_key(h)) for h in header]
    out['header'], out['source'] = header, body
    if 'photo1' not in keys and 'prompt' not in keys:
        out['errors'].append(f'в таблице {table.name} нет колонок «фото1»/«промпт» — проверьте заголовки')
        return
    col = {k: keys.index(k) for k in set(keys) if k}
    files = Files(base)
    for i, values in enumerate(body):
        get = lambda k: values[col[k]] if k in col else ''  # noqa: E731
        errors, photos = [], []
        for k in range(1, N_PHOTOS + 1):
            name = get(f'photo{k}')
            path = None
            if name:
                try:
                    path = files.find(name)
                    if path is None:
                        errors.append(f'фото {k}: файл не найден «{name}»')
                except ValueError as exc:
                    errors.append(f'фото {k}: {exc}')
            photos.append(path)
        n = get('n')
        out['rows'].append(make_row(row_id_of(n, i + 1), n or str(i + 1), photos, get('prompt'),
                                    {k: get(k) for k in MOD_KEYS}, defaults, errors, i))


def parse_txt(base, defaults, out):
    photo_re = re.compile(r'^(\d+)_([1-5])(\.\w+)$')
    text_re = re.compile(r'^(\d+)\.txt$', re.I)
    numbers, photos, texts = set(), {}, {}
    for p in sorted(Path(base).iterdir()):
        if not p.is_file():
            continue
        m = photo_re.match(p.name)
        if m and m.group(3).lower() in IMAGE_EXT:
            key = (int(m.group(1)), int(m.group(2)))
            if key in photos:
                out['warnings'].append(f'два файла для {m.group(1)}_{m.group(2)}: взят {photos[key].name}')
            else:
                photos[key] = p
            numbers.add(int(m.group(1)))
            continue
        m = text_re.match(p.name)
        if m:
            texts[int(m.group(1))] = p
            numbers.add(int(m.group(1)))
    width = max([3] + [len(re.match(r'\d+', p.name).group()) for p in list(photos.values()) + list(texts.values())])
    out['header'] = ['№'] + [f'фото{k}' for k in range(1, N_PHOTOS + 1)] + ['промпт', 'мод_персонаж', 'мод_лицо', 'мод_предмет']
    for i, n in enumerate(sorted(numbers)):
        errors, mods, prompt = [], {}, ''
        if n in texts:
            mods, prompt = parse_hash_lines(read_text(texts[n]))
        else:
            errors.append(f'нет файла промпта {str(n).zfill(width)}.txt')
        row = make_row(str(n).zfill(width), str(n), [photos.get((n, k)) for k in range(1, N_PHOTOS + 1)],
                       prompt, mods, defaults, errors, i)
        out['rows'].append(row)
        out['source'].append([row['n']] + [Path(p).name if p else '' for p in row['photos']]
                             + [row['prompt']] + [mods.get(k, '') for k in MOD_KEYS])


def parse_batch(folder):
    """Папка → {dir, kind, table, header, source, rows, defaults, errors, warnings}."""
    base = Path(folder).resolve()
    out = {'dir': str(base), 'kind': None, 'table': None, 'header': [], 'source': [], 'rows': [],
           'defaults': {}, 'defaults_file': None, 'errors': [], 'warnings': []}
    if not base.is_dir():
        out['errors'].append(f'папка пакета не найдена: {base.name}')
        return out
    out['defaults'], out['defaults_file'] = load_defaults(base)
    table, tables = pick_table(base)
    if table is not None:
        out['kind'], out['table'] = 'table', table.name
        if len(tables) > 1:
            out['warnings'].append(f'в папке {len(tables)} таблиц, взята {table.name}')
        try:
            parse_table(base, table, out['defaults'], out)
        except Exception as exc:  # noqa: BLE001 — битая таблица: ошибка пакета, а не падение узла
            out['errors'].append(f'таблица {table.name}: {exc}')
    else:
        out['kind'] = 'txt'
        parse_txt(base, out['defaults'], out)
    seen = {}
    for row in out['rows']:
        if row['row_id'] in seen:
            seen[row['row_id']] += 1
            row['errors'].append(f'номер строки повторяется: {row["n"]}')
            row['row_id'] = f'{row["row_id"]}_{seen[row["row_id"]]}'
        else:
            seen[row['row_id']] = 1
    if not out['rows'] and not out['errors']:
        out['errors'].append('в пакете нет строк (нужна таблица xlsx/csv или файлы 001_1.png … 001.txt)')
    return out


def summary(parsed, name=None):
    """Короткая сводка для узла: сколько строк, сколько ошибок, какие моды."""
    rows = parsed['rows']
    bad = [r for r in rows if r['errors']]
    kind = f'таблица {parsed["table"]}' if parsed['kind'] == 'table' else 'txt-файлы'
    lines = [f'Пакет: {name or Path(parsed["dir"]).name} ({kind})',
             f'Строк: {len(rows)}, с ошибками: {len(bad)}']
    for key in MOD_KEYS:
        used = {}
        for r in rows:
            if r['mods'][key]:
                used[r['mods'][key]] = used.get(r['mods'][key], 0) + 1
        text = ', '.join(f'{v} ({c})' for v, c in sorted(used.items())) or '—'
        lines.append(f'Мод {MOD_RU[key]}: {text}')
    lines += [f'Ошибка пакета: {e}' for e in parsed['errors']]
    lines += [f'Внимание: {w}' for w in parsed['warnings']]
    lines += [f'{r["row_id"]}: {"; ".join(r["errors"])}' for r in bad[:20]]
    if len(bad) > 20:
        lines.append(f'… и ещё {len(bad) - 20} строк с ошибками')
    return '\n'.join(lines)
