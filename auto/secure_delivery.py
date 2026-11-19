"""Изолированная выдача: только проверенный снимок и явный белый список.

Не использовать совместно с legacy AutoReport и /3da/auto ZIP-маршрутом.
Драйвер обязан вызывать план, сохранение кадра и оценку последовательно.
"""
import hashlib
import io
import json
import os
import re
import threading
import uuid
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from . import judge, store

_LOCKS = {}
_GUARD = threading.Lock()
MAX_SEED = (1 << 63) - 1


def _lock(root):
    with _GUARD:
        return _LOCKS.setdefault(str(root.resolve()), threading.RLock())


def _root(batch):
    name, _, _ = store.resolve(batch)
    # Отдельный корень: legacy архиватор не видит карантин и результаты.
    root = store.root('output').parent / 'auto_secure' / name
    for path in (root.parent, root):
        if path.is_symlink() or getattr(path, 'is_junction', lambda: False)():
            raise ValueError('ссылки в безопасном корне запрещены')
    root.mkdir(parents=True, exist_ok=True)
    return root


def _safe_path(root, *parts):
    path = root.joinpath(*parts)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('путь выходит из безопасного корня')
    current = root
    for part in path.relative_to(root).parts:
        current = current / part
        if current.is_symlink() or getattr(current, 'is_junction', lambda: False)():
            raise ValueError('ссылки внутри безопасного корня запрещены')
    return path


def _batch_data(batch):
    # JSON из узла загрузки может содержать устаревший снимок строк.
    from .nodes import batch_data
    _, folder, _ = store.resolve(batch)
    return batch_data(str(folder.resolve()))


def _identity(source):
    # Имена файлов недостаточны: изображение может быть заменено на месте.
    photos = []
    for name in source.get('photos', []):
        if name:
            with open(name, 'rb') as f:
                digest = hashlib.file_digest(f, 'sha256').hexdigest()
            photos.append(digest)
        else:
            photos.append(None)
    payload = json.dumps({'row': source, 'photo_hashes': photos}, ensure_ascii=False,
                         sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def _row(batch, row_index):
    data, st = _batch_data(batch)
    if type(row_index) is not int or not 0 <= row_index < len(data['rows']):
        raise ValueError('неверный индекс строки')
    source = data['rows'][row_index]
    rid = source['row_id']
    if rid not in st['rows'] or st['rows'][rid].get('status') == 'error' or source.get('errors'):
        raise ValueError('строка отсутствует или содержит ошибку')
    if not re.fullmatch(r'[\w-]+', rid):
        raise ValueError('небезопасный идентификатор строки')
    return rid, _identity(source)


def _read(root):
    path = _safe_path(root, 'secure_state.json')
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'rows': {}}


def _write(root, data):
    # Никогда не оставляем старый ZIP после изменения статусов.
    (_safe_path(root, 'delivery.zip')).unlink(missing_ok=True)
    tmp = _safe_path(root, f'.state-{os.getpid()}-{threading.get_ident()}.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, _safe_path(root, 'secure_state.json'))


def _check_identity(root, row, identity):
    if not row or row.get('identity') != identity:
        _safe_path(root, 'delivery.zip').unlink(missing_ok=True)
        raise ValueError('содержимое входной строки изменилось; нужен новый пакет')


def plan(batch, row_index, seed):
    if type(seed) is not int or not 0 <= seed <= MAX_SEED:
        raise ValueError('недопустимый seed')
    root = _root(batch)
    with _lock(root):
        rid, identity = _row(batch, row_index)
        data = _read(root)
        row = data['rows'].get(rid)
        if row is None or row.get('identity') != identity or not row.get('plan_id'):
            row = {'identity': identity, 'plan_id': uuid.uuid4().hex, 'base_seed': seed,
                   'attempt': 0, 'status': 'pending', 'reason': '', 'frames': {}}
            data['rows'][rid] = row
            _write(root, data)
        if seed != row['base_seed']:
            raise ValueError('seed первой попытки уже зафиксирован')
        if row['status'] in ('done', 'artist'):
            return row['attempt'], (seed + row['attempt']) % (MAX_SEED + 1), False
        return row['attempt'], (seed + row['attempt']) % (MAX_SEED + 1), True


def _png(images):
    arr = images.detach().cpu().numpy() if hasattr(images, 'detach') else np.asarray(images)
    if arr.ndim != 4 or arr.shape[0] != 1 or arr.shape[-1] != 3 or not np.isfinite(arr).all():
        raise ValueError('ожидается один конечный RGB IMAGE')
    arr = np.clip(arr[0] * 255, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, 'RGB').save(buf, format='PNG')
    return buf.getvalue()


def _plan_token(root, rid, row, attempt, seed):
    return (str(root.resolve()), rid, row['identity'], row['plan_id'], attempt, seed)


def stage(batch, row_index, images, attempt, seed, expected_identity=None, plan_token=None):
    root = _root(batch)
    payload = _png(images)
    digest = hashlib.sha256(payload).hexdigest()
    with _lock(root):
        rid, identity = _row(batch, row_index)
        data = _read(root)
        row = data['rows'].get(rid)
        _check_identity(root, row, identity)
        # Кадр, начатый для прежнего источника, не привязываем к новому plan.
        if expected_identity is not None and expected_identity != identity:
            raise ValueError('содержимое входной строки изменилось во время генерации')
        if row['status'] != 'pending':
            raise ValueError('сначала запросите активную попытку')
        if type(attempt) is not int or attempt != row['attempt'] or type(seed) is not int or seed != (row['base_seed'] + attempt) % (MAX_SEED + 1):
            raise ValueError('устаревшая попытка или неверный seed')
        if plan_token is not None and plan_token != _plan_token(root, rid, row, attempt, seed):
            raise ValueError('устаревший или неверный токен планирования')
        previous = row['frames'].get(str(attempt))
        if previous and previous != digest:
            raise ValueError('кадр попытки уже зафиксирован')
        quarantine = _safe_path(root, 'quarantine', rid)
        quarantine.mkdir(parents=True, exist_ok=True)
        path = quarantine / f'{attempt}-{digest}.png'
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('содержимое карантина изменено')
        if not path.exists():
            tmp = path.with_suffix('.tmp')
            tmp.write_bytes(payload)
            os.replace(tmp, path)
        row['frames'][str(attempt)] = digest
        _write(root, data)
    import torch
    with Image.open(io.BytesIO(payload)) as image:
        frozen = torch.from_numpy(np.asarray(image.convert('RGB'), dtype=np.float32).copy() / 255.0)[None]
    return frozen, (batch, row_index, attempt, digest, identity)


def restore(batch, row_index):
    """Восстановить ровно текущий удержанный кадр и билет для фазы зрения."""
    root = _root(batch)
    with _lock(root):
        rid, identity = _row(batch, row_index)
        row = _read(root)['rows'].get(rid)
        _check_identity(root, row, identity)
        attempt = row.get('attempt')
        if row.get('status') != 'pending' or type(attempt) is not int or attempt not in (0, 1):
            raise ValueError('нет активного кадра для проверки')
        digest = row.get('frames', {}).get(str(attempt))
        if not isinstance(digest, str) or not re.fullmatch(r'[0-9a-f]{64}', digest):
            raise ValueError('кадр попытки не сохранён')
        path = _safe_path(root, 'quarantine', rid, f'{attempt}-{digest}.png')
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != digest:
            raise ValueError('кадр в карантине изменён')
        import torch
        with Image.open(io.BytesIO(payload)) as image:
            frozen = torch.from_numpy(np.asarray(image.convert('RGB'), dtype=np.float32).copy() / 255.0)[None]
        return frozen, (batch, row_index, attempt, digest, identity)


def adjudicate(ticket, text):
    batch, row_index, attempt, digest, ticket_identity = ticket
    root = _root(batch)
    with _lock(root):
        rid, identity = _row(batch, row_index)
        data = _read(root)
        row = data['rows'].get(rid)
        _check_identity(root, row, identity)
        if ticket_identity != identity:
            raise ValueError('устаревшая входная строка')
        if row['attempt'] != attempt or row['status'] != 'pending' or row['frames'].get(str(attempt)) != digest:
            raise ValueError('устаревший или неверный кадр')
        path = _safe_path(root, 'quarantine', rid, f'{attempt}-{digest}.png')
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('кадр в карантине изменён')
        try:
            verdict, reason, ok = judge.decide(text, attempt)
        except Exception:
            verdict, reason, ok = 'artist', 'Ошибка проверки результата', False
        if verdict == 'ok' and ok is True:
            row.update(status='done', reason=reason, approved=digest)
        elif verdict == 'retry' and attempt == 0 and ok is False:
            row.update(status='pending', reason=reason, attempt=1)
        else:
            row.update(status='artist', reason=reason or 'Требуется проверка художником')
        _write(root, data)
        return row['status'], row['reason'], row['attempt']


def run_row(batch, row_index, seed, generate, inspect):
    """Последовательный CPU/локальный драйвер одной строки.

    generate(seed, attempt) -> IMAGE; inspect(проверяемый IMAGE, attempt) -> JSON text.
    Callback inspect должен синхронно вызвать локальную VLM на переданном IMAGE.
    Повторный вызов после сбоя генерации не вызывает её повторно: artist.
    Блокировка внутрипроцессная: запускать только из одного процесса.
    """
    root = _root(batch)
    with _lock(root):
        rid, identity = _row(batch, row_index)
        while True:
            attempt, actual_seed, active = plan(batch, row_index, seed)
            if not active:
                row = _read(root)['rows'][rid]
                return row['status'], row['reason'], row['attempt']
            data = _read(root)
            row = data['rows'][rid]
            _check_identity(root, row, identity)
            digest = row['frames'].get(str(attempt))
            try:
                if digest:
                    import torch
                    path = _safe_path(root, 'quarantine', rid, f'{attempt}-{digest}.png')
                    payload = path.read_bytes()
                    if hashlib.sha256(payload).hexdigest() != digest:
                        raise ValueError('кадр изменён')
                    with Image.open(io.BytesIO(payload)) as image:
                        frozen = torch.from_numpy(np.asarray(image.convert('RGB'), dtype=np.float32).copy() / 255.0)[None]
                    ticket = (batch, row_index, attempt, digest, identity)
                else:
                    started = row.setdefault('started', [])
                    if attempt in started:
                        raise ValueError('предыдущая генерация оборвалась; повтор удержан')
                    started.append(attempt)
                    _write(root, data)  # до вызова генератора: не более двух вызовов после рестарта
                    image = generate(actual_seed, attempt)
                    frozen, ticket = stage(batch, row_index, image, attempt, actual_seed, identity)
                result = adjudicate(ticket, inspect(frozen, attempt))
            except Exception:
                current_rid, current_identity = _row(batch, row_index)
                data = _read(root)
                row = data['rows'][rid]
                # Источник сменился: не пишем статус прежней попытки в новую строку.
                _check_identity(root, row, identity)
                if current_rid != rid or current_identity != identity:
                    raise ValueError('содержимое входной строки изменилось во время генерации')
                row.update(status='artist', reason='Ошибка генерации или проверки; кадр удержан')
                _write(root, data)
                return row['status'], row['reason'], row['attempt']
            if result[0] != 'pending':
                return result


def archive(batch):
    root = _root(batch)
    with _lock(root):
        _safe_path(root, 'delivery.zip').unlink(missing_ok=True)
        data = _read(root)
        if not data['rows'] or any(r['status'] not in ('done', 'artist') for r in data['rows'].values()):
            raise ValueError('есть незавершённые строки')
        # Проверяем все строки исходного пакета, а не только запущенные драйвером.
        batch_info, _ = _batch_data(batch)
        expected = {row['row_id'] for row in batch_info['rows'] if not row.get('errors')}
        if set(data['rows']) != expected:
            raise ValueError('обработаны не все строки пакета')
        for source in batch_info['rows']:
            if not source.get('errors'):
                _check_identity(root, data['rows'][source['row_id']], _identity(source))
        target = _safe_path(root, 'delivery.zip')
        tmp = _safe_path(root, '.delivery.zip.tmp')
        with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as z:
            manifest = [{'row_id': row['row_id'], 'status': 'error', 'reason': '; '.join(row['errors'])}
                        for row in batch_info['rows'] if row.get('errors')]
            for rid, row in sorted(data['rows'].items()):
                status = row['status']
                item = {'row_id': rid, 'status': status, 'reason': row['reason']}
                if status == 'done':
                    digest = row.get('approved', '')
                    if not re.fullmatch(r'[0-9a-f]{64}', digest):
                        raise ValueError('нет подтверждённого кадра')
                    path = _safe_path(root, 'quarantine', rid, f'{row["attempt"]}-{digest}.png')
                    payload = path.read_bytes()
                    if hashlib.sha256(payload).hexdigest() != digest:
                        raise ValueError('подтверждённый кадр изменён')
                    name = f'готово/{rid}.png'
                    z.writestr(name, payload)
                    item['file'] = name
                manifest.append(item)
            z.writestr('manifest.json', json.dumps(manifest, ensure_ascii=False, indent=2))
        os.replace(tmp, target)
        return str(target)


class SecurePlan:
    CATEGORY = '3DA/auto'
    FUNCTION = 'execute'
    RETURN_TYPES = ('INT', 'INT', 'BOOLEAN', '3DA_SECURE_PLAN')
    RETURN_NAMES = ('attempt', 'seed', 'run', 'plan_token')
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'batch': ('STRING', {'forceInput': True}), 'row_index': ('INT', {'default': 0, 'min': 0}), 'seed': ('INT', {'default': 0, 'min': 0, 'max': MAX_SEED})}}
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float('nan')
    def execute(self, batch, row_index, seed):
        root = _root(batch)
        with _lock(root):
            attempt, actual_seed, active = plan(batch, row_index, seed)
            rid, identity = _row(batch, row_index)
            row = _read(root)['rows'][rid]
            _check_identity(root, row, identity)
            token = _plan_token(root, rid, row, attempt, actual_seed)
            return attempt, actual_seed, active, token


class SecureStage:
    CATEGORY = '3DA/auto'
    FUNCTION = 'execute'
    RETURN_TYPES = ('IMAGE', '3DA_SECURE_FRAME')
    RETURN_NAMES = ('image_for_vision', 'ticket')
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'batch': ('STRING', {'forceInput': True}), 'row_index': ('INT', {'default': 0, 'min': 0}), 'image': ('IMAGE',), 'attempt': ('INT', {'forceInput': True}), 'seed': ('INT', {'forceInput': True}), 'plan_token': ('3DA_SECURE_PLAN', {'forceInput': True})}}
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float('nan')
    def execute(self, batch, row_index, image, attempt, seed, plan_token):
        if not isinstance(plan_token, tuple) or len(plan_token) != 6:
            raise ValueError('требуется токен узла планирования')
        return stage(batch, row_index, image, attempt, seed, plan_token=plan_token)


class SecureRestore:
    CATEGORY = '3DA/auto'
    FUNCTION = 'execute'
    RETURN_TYPES = ('IMAGE', '3DA_SECURE_FRAME')
    RETURN_NAMES = ('image_for_vision', 'ticket')
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'batch': ('STRING', {'forceInput': True}), 'row_index': ('INT', {'default': 0, 'min': 0})}}
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float('nan')
    def execute(self, batch, row_index):
        return restore(batch, row_index)


class SecureJudge:
    CATEGORY = '3DA/auto'
    FUNCTION = 'execute'
    OUTPUT_NODE = True
    RETURN_TYPES = ('STRING', 'STRING', 'INT')
    RETURN_NAMES = ('status', 'reason', 'attempt')
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'ticket': ('3DA_SECURE_FRAME',), 'text': ('STRING', {'forceInput': True})}}
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float('nan')
    def execute(self, ticket, text):
        return adjudicate(ticket, text)


class SecureArchive:
    CATEGORY = '3DA/auto'
    FUNCTION = 'execute'
    OUTPUT_NODE = True
    RETURN_TYPES = ('STRING',)
    RETURN_NAMES = ('zip',)
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'batch': ('STRING', {'forceInput': True})}}
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float('nan')
    def execute(self, batch):
        return (archive(batch),)
