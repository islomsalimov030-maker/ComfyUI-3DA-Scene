"""Состояние пакета на диске: `output/auto/<пакет>/state.json`.

По строке: {status, phase_done: {фаза: время}, reason, attempts, files: {фаза: [пути от папки пакета]}}.
Запись атомарная (tmp + os.replace): при обрыве на диске остаётся прежний целый файл,
повторный запуск продолжает с места (строки с пройденной фазой пропускаются).
"""
import json
import os
import threading
import time
from pathlib import Path

STATUSES = ('new', 'parsed', 'placed', 'generated', 'checked', 'done', 'artist', 'error')
STATUS_RU = {'new': 'новая', 'parsed': 'разобрана', 'placed': 'поставлена', 'generated': 'сгенерирована',
             'checked': 'проверена', 'done': 'готово', 'artist': 'художнику', 'error': 'ошибка'}
FINAL = ('done', 'artist', 'error')
STATE_FILE = 'state.json'

_locks = {}
_locks_guard = threading.Lock()


def _lock(out_dir):
    key = str(Path(out_dir).resolve())
    with _locks_guard:
        return _locks.setdefault(key, threading.RLock())


def now():
    return time.strftime('%Y-%m-%d %H:%M:%S')


def new_row(n=''):
    return {'n': n, 'status': 'new', 'phase_done': {}, 'reason': '', 'attempts': 0, 'files': {}}


def load(out_dir):
    """Состояние или None, если пакет ещё не запускался."""
    path = Path(out_dir) / STATE_FILE
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding='utf-8'))


def save(out_dir, state):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state['updated'] = now()
    tmp = out_dir / (STATE_FILE + f'.{os.getpid()}.{threading.get_ident()}.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(state, f, ensure_ascii=False, indent=1)
        f.flush()
        os.fsync(f.fileno())
    for attempt in range(20):  # Windows: файл может держать читатель/антивирус — короткие повторы
        try:
            os.replace(tmp, out_dir / STATE_FILE)
            return state
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.05)


def sync(out_dir, parsed, name=''):
    """Свести состояние с разобранным пакетом: новые строки добавить, прежний прогресс сохранить.
    Ошибка разбора → status=error с причиной; исправленная строка (ошибки ушли) → снова parsed."""
    with _lock(out_dir):
        for tmp in Path(out_dir).glob(STATE_FILE + '.*.tmp'):  # хвосты оборванной записи
            try:
                tmp.unlink()
            except OSError:
                pass
        old = load(out_dir)
        before = json.dumps(old, ensure_ascii=False, sort_keys=True)
        state = old or {'batch': name, 'created': now(), 'rows': {}, 'order': []}
        state['batch'] = name or state.get('batch', '')
        rows = state['rows']
        for row in parsed['rows']:
            rid = row['row_id']
            st = rows.setdefault(rid, new_row(row['n']))
            st['n'] = row['n']
            if row['errors']:
                st['status'], st['reason'] = 'error', '; '.join(row['errors'])
                st['parse_error'] = True
            elif st['status'] == 'new' or st.pop('parse_error', False):
                st['status'], st['reason'] = 'parsed', ''
        state['order'] = [r['row_id'] for r in parsed['rows']]
        state['errors'] = list(parsed.get('errors', []))
        if old is not None and json.dumps(state, ensure_ascii=False, sort_keys=True) == before:
            return state  # без изменений — не трогать файл (его время — признак свежести ZIP)
        return save(out_dir, state)


def skip(row_state, phase):
    """Строка уже прошла эту фазу или в статусе error."""
    if row_state is None:
        return True
    return row_state.get('status') == 'error' or str(phase) in row_state.get('phase_done', {})


def pending(state, phase):
    """row_id строк, которым ещё нужна фаза (в порядке пакета)."""
    return [rid for rid in state.get('order', []) if not skip(state['rows'].get(rid), phase)]


def mark(out_dir, row_id, phase, status=None, reason=None, files=None):
    """Фаза строки пройдена: файлы, статус, причина, счётчик попыток."""
    if status is not None and status not in STATUSES:
        raise ValueError(f'неизвестный статус «{status}» (допустимы: {", ".join(STATUSES)})')
    with _lock(out_dir):
        state = load(out_dir)
        if state is None or row_id not in state['rows']:
            raise KeyError(f'строки {row_id} нет в состоянии пакета')
        st = state['rows'][row_id]
        st['attempts'] = int(st.get('attempts', 0)) + 1
        st['phase_done'][str(phase)] = now()
        if files is not None:
            st['files'][str(phase)] = list(files)
        if status:
            st['status'] = status
            st.pop('parse_error', None)
        if reason is not None:
            st['reason'] = reason
        save(out_dir, state)
        return st


def counts(state):
    out = {s: 0 for s in STATUSES}
    for rid in state.get('order', []):
        out[state['rows'][rid]['status']] += 1
    return out
