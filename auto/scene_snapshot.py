"""Переносимый снимок сцены; интеграция с узлами и хранилищем намеренно отсутствует.

Границей доверия служит отдельно сохранённый digest, а не содержимое контейнера.
Модуль работает только с байтами: сам он не открывает пути и не десериализует код.
"""
import base64
import binascii
import hashlib
import hmac
import json
import math
import re
from types import MappingProxyType

_VERSION = 1
_MAX_BLOB = 64 * 1024 * 1024
_MAX_TOTAL = 256 * 1024 * 1024
_MAX_CONTAINER = 360 * 1024 * 1024
_MAX_ENTRIES = 128
_NAME = re.compile(r'[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}\Z', re.ASCII)
_HASH = re.compile(r'[0-9a-f]{64}\Z', re.ASCII)


def _json(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(',', ':'),
                      allow_nan=False).encode('ascii')


def _metadata(value, depth=0):
    if depth > 16:
        raise ValueError('слишком глубокие метаданные')
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    if type(value) is list:
        for item in value:
            _metadata(item, depth + 1)
        return
    if type(value) is dict and all(type(k) is str for k in value):
        for item in value.values():
            _metadata(item, depth + 1)
        return
    raise ValueError('метаданные должны содержать только JSON-значения')


def _digest(value):
    return hashlib.sha256(value).hexdigest()


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('повторяющийся ключ JSON')
        result[key] = value
    return result


def create(row_identity, selection, blobs):
    """Вернуть (неизменяемые bytes контейнера, SHA-256 контейнера).

    row_identity — SHA-256 текущей строки, вычисленный доверенным драйвером;
    selection — JSON-метаданные выбора; blobs — непустой dict имени и bytes.
    Имена — идентификаторы слотов, не пути. Состав и формат слотов задаёт драйвер.
    """
    if type(row_identity) is not str or not _HASH.fullmatch(row_identity):
        raise ValueError('неверная identity строки')
    if type(selection) is not dict:
        raise ValueError('selection должен быть объектом')
    _metadata(selection)
    if type(blobs) is not dict or not 0 < len(blobs) <= _MAX_ENTRIES:
        raise ValueError('неверный набор данных сцены')
    entries = {}
    total = 0
    for name, data in blobs.items():
        if type(name) is not str or not _NAME.fullmatch(name) or name in ('.', '..'):
            raise ValueError('неверное имя слота')
        if type(data) is not bytes or len(data) > _MAX_BLOB:
            raise ValueError('неверные данные слота')
        total += len(data)
        if total > _MAX_TOTAL:
            raise ValueError('снимок слишком велик')
        entries[name] = {'size': len(data), 'sha256': _digest(data),
                         'data': base64.b64encode(data).decode('ascii')}
    payload = _json({'version': _VERSION, 'row_identity': row_identity,
                     'selection': selection, 'blobs': entries})
    if len(payload) > _MAX_CONTAINER:
        raise ValueError('снимок слишком велик')
    return payload, _digest(payload)


def restore(payload, expected_digest, expected_row_identity):
    """Проверить контейнер по независимым ожидаемым identity и вернуть копии.

    Результат — (selection, неизменяемое отображение slot -> bytes). Проверка
    после восстановления пути/файла и защита digest от подмены — задача драйвера.
    """
    if type(payload) is not bytes or len(payload) > _MAX_CONTAINER:
        raise ValueError('неверный контейнер')
    for value in (expected_digest, expected_row_identity):
        if type(value) is not str or not _HASH.fullmatch(value):
            raise ValueError('неверная ожидаемая identity')
    if not hmac.compare_digest(_digest(payload), expected_digest):
        raise ValueError('контейнер изменён')
    try:
        obj = json.loads(payload, object_pairs_hook=_unique,
                         parse_constant=lambda _: (_ for _ in ()).throw(ValueError('неверное число')))
        if type(obj) is not dict or set(obj) != {'version', 'row_identity', 'selection', 'blobs'}:
            raise ValueError('неверная схема')
        if type(obj['version']) is not int or obj['version'] != _VERSION:
            raise ValueError('неподдерживаемая версия')
        if not hmac.compare_digest(obj['row_identity'], expected_row_identity):
            raise ValueError('неверная identity строки')
        if type(obj['selection']) is not dict:
            raise ValueError('неверный выбор')
        _metadata(obj['selection'])
        entries = obj['blobs']
        if type(entries) is not dict or not 0 < len(entries) <= _MAX_ENTRIES:
            raise ValueError('неверный набор слотов')
        result, total = {}, 0
        for name, entry in entries.items():
            if not _NAME.fullmatch(name) or name in ('.', '..'):
                raise ValueError('неверное имя слота')
            if type(entry) is not dict or set(entry) != {'size', 'sha256', 'data'}:
                raise ValueError('неверная запись слота')
            size, digest, encoded = entry['size'], entry['sha256'], entry['data']
            if type(size) is not int or not 0 <= size <= _MAX_BLOB:
                raise ValueError('неверный размер')
            if type(digest) is not str or not _HASH.fullmatch(digest) or type(encoded) is not str:
                raise ValueError('неверная запись данных')
            if len(encoded) != 4 * ((size + 2) // 3):
                raise ValueError('неверная длина данных')
            total += size
            if total > _MAX_TOTAL:
                raise ValueError('снимок слишком велик')
            data = base64.b64decode(encoded, validate=True)
            if len(data) != size or not hmac.compare_digest(_digest(data), digest):
                raise ValueError('данные слота изменены')
            result[name] = data
        if _json(obj) != payload:
            raise ValueError('неканонический контейнер')
        return obj['selection'], MappingProxyType(result)
    except (TypeError, KeyError, UnicodeError, json.JSONDecodeError, binascii.Error, RecursionError) as exc:
        raise ValueError('повреждённый контейнер') from exc
