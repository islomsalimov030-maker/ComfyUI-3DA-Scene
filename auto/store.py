"""Где лежат пакеты: вход `input/auto/<пакет>/`, выход `output/auto/<пакет>/`.

Корни берутся из ComfyUI (folder_paths); в тестах их подменяют через set_roots().
"""
import json
import os
import time
from pathlib import Path

from . import batch as batch_mod
from . import report as report_mod
from . import state as state_mod

MAX_FILE = 50 * 1024 * 1024
_roots = {}


def set_roots(input_root, output_root):
    _roots['input'], _roots['output'] = Path(input_root), Path(output_root)


def root(kind):
    if kind not in _roots:
        import folder_paths
        base = folder_paths.get_input_directory() if kind == 'input' else folder_paths.get_output_directory()
        return Path(base) / 'auto'
    return _roots[kind]


def inside(path, base):
    """Лежит ли path внутри base (после разрешения ссылок и `..`)."""
    path, base = Path(path).resolve(), Path(base).resolve()
    return path == base or base in path.parents


def check_dirs(name, folder, out_dir, allow_path=False):
    """Проверить папки пакета: выход — только `output/auto/<имя>`, вход — внутри `input/auto`
    (абсолютная папка вне его — только при allow_path, т. е. из узла). → (имя, вход, выход)."""
    name = batch_mod.safe_name(name)
    expected = root('output') / name
    if Path(out_dir).resolve() != expected.resolve():
        raise ValueError(f'папка выхода пакета должна быть {expected}')
    folder = Path(folder)
    if not inside(folder, root('input')) and not (allow_path and folder.is_absolute()):
        raise ValueError('папка пакета вне input/auto')
    return name, folder, expected


def resolve_name(value):
    """Только имя пакета (для маршрутов сервера): без JSON и путей. → (имя, вход, выход)."""
    value = str(value or '').strip()
    if not value:
        raise ValueError('не указан пакет')
    name = batch_mod.safe_name(value)
    if name != value:
        raise ValueError(f'недопустимое имя пакета: «{value}»')
    return name, root('input') / name, root('output') / name


def resolve(value, allow_path=True):
    """Значение виджета/выхода `batch` → (имя, папка входа, папка выхода).
    Принимает имя пакета, JSON с выхода «пакет: загрузка» или (allow_path) абсолютный путь к папке.
    Выход всегда `output/auto/<имя>`."""
    value = str(value or '').strip()
    if value.startswith('{'):
        data = json.loads(value)
        return check_dirs(data['name'], data['dir'], data['out_dir'], allow_path)
    if not value:
        raise ValueError('пакет не выбран — нажмите «Выбрать папку» или впишите имя пакета')
    if os.path.isabs(value):
        if not allow_path:
            raise ValueError('путь к папке здесь не принимается — только имя пакета')
        folder = Path(value)
        name = batch_mod.safe_name(folder.name)
    else:
        return resolve_name(value)
    return name, folder, root('output') / name


def open_batch(value):
    """Разобрать пакет и свести состояние. → (имя, parsed, state, out_dir)."""
    return open_dirs(*resolve(value))


def open_dirs(name, folder, out_dir):
    out_dir = Path(out_dir)
    parsed = batch_mod.parse_batch(folder)
    state = state_mod.sync(out_dir, parsed, name)
    return name, parsed, state, out_dir


def fingerprint(folder):
    """Изменился ли пакет: имена, размеры и время файлов."""
    folder = Path(folder)
    if not folder.is_dir():
        return 'missing'
    items = sorted((p.relative_to(folder).as_posix(), p.stat().st_size, p.stat().st_mtime_ns)
                   for p in folder.rglob('*') if p.is_file())
    return json.dumps(items)


def batches():
    """Список пакетов во входной папке, новые сверху."""
    base = root('input')
    out = []
    if base.is_dir():
        for p in base.iterdir():
            if not p.is_dir():
                continue
            files = [f for f in p.rglob('*') if f.is_file()]
            st = state_mod.load(root('output') / p.name) if (root('output') / p.name).is_dir() else None
            out.append({'batch': p.name, 'files': len(files), 'mtime': p.stat().st_mtime,
                        'rows': len(st['order']) if st else None,
                        'counts': state_mod.counts(st) if st else None})
    return sorted(out, key=lambda x: -x['mtime'])


def target_path(batch_name, rel):
    """Куда положить загруженный файл. ValueError — недопустимое имя/путь/тип."""
    name = batch_mod.safe_name(batch_name)
    folder = root('input') / name
    folder.mkdir(parents=True, exist_ok=True)
    path = batch_mod.safe_rel(folder, rel)
    if path.suffix.lower() not in batch_mod.UPLOAD_EXT:
        raise ValueError(f'тип файла не нужен пакету: «{rel}»')
    return path


class Sink:
    """Запись загружаемого файла через .part с лимитом MAX_FILE: при превышении — OverflowError, файла нет."""
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.tmp = self.path.with_name(self.path.name + f'.{time.time_ns()}.part')
        self.file = open(self.tmp, 'wb')
        self.size = 0

    def write(self, chunk):
        self.size += len(chunk)
        if self.size > MAX_FILE:
            self.abort()
            raise OverflowError(f'файл больше {MAX_FILE // (1024 * 1024)} МБ: {self.path.name}')
        self.file.write(chunk)

    def commit(self):
        self.file.close()
        os.replace(self.tmp, self.path)
        return self.size

    def abort(self):
        self.file.close()
        if self.tmp.exists():
            self.tmp.unlink()


def write_stream(path, chunks):
    sink = Sink(path)
    try:
        for chunk in chunks:
            sink.write(chunk)
    except BaseException:
        sink.abort()
        raise
    return sink.commit()


def build(name, folder, out_dir):
    """Отчёт + раскладка + ZIP по свежему разбору папки и состоянию."""
    name, parsed, state, out_dir = open_dirs(name, folder, out_dir)
    return report_mod.build(out_dir, parsed, state, name)


def build_report(value):
    return build(*resolve(value))


def zip_path(value, rebuild_if_stale=True):
    """ZIP пакета; собирается заново, если его нет или состояние новее архива.
    value — значение для resolve() или готовый кортеж (имя, вход, выход)."""
    name, folder, out_dir = value if isinstance(value, tuple) else resolve(value)
    path = out_dir / f'{name}.zip'
    state_file = out_dir / state_mod.STATE_FILE
    if rebuild_if_stale and (not path.is_file() or not state_file.is_file()
                             or state_file.stat().st_mtime > path.stat().st_mtime):
        build(name, folder, out_dir)
    return path
