"""Безопасный XLSX и ZIP из завершённого secure_delivery state.

В ZIP попадают отчёт, manifest и одобренные PNG строк done с проверенным SHA-256.
Строки artist учитываются в отчёте. Их изображения остаются в карантине.
"""
import hashlib
import io
import json
import re
import zipfile
from . import batch as batch_mod
from . import secure_delivery, store

REPORT_NAME = 'отчёт.xlsx'


def _text(value):
    """Все пользовательские значения в XLSX остаются текстом, никогда формулой."""
    if value is None:
        return ''
    return str(value)


def _safe_png(root, rid, row):
    attempt = row.get('attempt')
    if type(attempt) is not int or attempt not in (0, 1):
        raise ValueError(f'неверная попытка строки {rid}')
    frames = row.get('frames')
    if not isinstance(frames, dict):
        raise ValueError(f'неверный список кадров строки {rid}')
    digest = frames.get(str(attempt))
    if digest is None and str(attempt) not in frames and row.get('status') == 'artist':
        # Генерация могла завершиться ошибкой до сохранения текущего кадра.
        return None
    if not isinstance(digest, str) or not re.fullmatch(r'[0-9a-f]{64}', digest):
        raise ValueError(f'нет проверенного SHA-256 для строки {rid}')
    if row.get('status') == 'done' and row.get('approved') != digest:
        raise ValueError(f'подтверждение кадра строки {rid} не совпадает')
    path = secure_delivery._safe_path(root, 'quarantine', rid, f'{attempt}-{digest}.png')
    payload = path.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != digest or row.get('frames', {}).get(str(attempt)) != digest:
        raise ValueError(f'проверенный кадр строки {rid} изменён')
    return payload if row['status'] == 'done' else None


def _xlsx(header, rows, batch_errors):
    import openpyxl
    from openpyxl.styles import Font

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = 'отчёт'
    ws.append([_text(x) for x in header])
    for row in rows:
        ws.append([_text(x) for x in row])
    ws.freeze_panes = 'A2'
    errors_ws = wb.create_sheet('ошибки пакета')
    errors_ws.append(['ошибка'])
    for error in batch_errors:
        errors_ws.append([_text(error)])
    for sheet in wb:
        for cells in sheet.iter_rows():
            for cell in cells:
                cell.font = Font(name='Arial', size=10, bold=(cell.row == 1))
                # Explicit string type prevents formula interpretation of external text.
                cell.data_type = 's'
    output = io.BytesIO()
    wb.save(output)
    wb.close()
    return output.getvalue()


def build(batch):
    """Собрать безопасную выгрузку; вернуть {'xlsx': bytes, 'zip': bytes, ...}.

    Текущая таблица и состояние перечитываются на каждый вызов. Все исходные
    валидные строки должны иметь совпадающий secure identity, а строки с ошибками
    должны отсутствовать в secure state. Неизвестное или устаревшее состояние
    приводит к отказу, не к частичной выгрузке.

    Только один процесс-драйвер; входы во время build должны быть неизменяемыми.
    Повторная проверка перед возвратом обнаруживает изменения в ходе сборки,
    но не является межпроцессной блокировкой или полной защитой от TOCTOU.
    """
    root = secure_delivery._root(batch)
    with secure_delivery._lock(root):
        data, _ = secure_delivery._batch_data(batch)
        _, folder, _ = store.resolve(batch)
        parsed = batch_mod.parse_batch(folder)
        source_rows = data.get('rows')
        if parsed.get('rows') != source_rows:
            raise ValueError('входной пакет изменился во время чтения')
        secure_state = secure_delivery._read(root)
        entries = secure_state.get('rows')
        if not isinstance(source_rows, list) or not isinstance(entries, dict):
            raise ValueError('некорректное состояние безопасной выдачи')

        valid = []
        errors = []
        source_ids = set()
        for source in source_rows:
            rid = source.get('row_id')
            if not isinstance(rid, str) or not re.fullmatch(r'[\w-]+', rid) or rid in source_ids:
                raise ValueError('недопустимый или повторяющийся идентификатор строки')
            source_ids.add(rid)
            if source.get('errors'):
                errors.append((source, '; '.join(map(_text, source['errors']))))
                if rid in entries:
                    raise ValueError(f'строка с ошибкой присутствует в безопасном состоянии: {rid}')
                continue
            saved = entries.get(rid)
            if not saved or saved.get('identity') != secure_delivery._identity(source):
                raise ValueError(f'содержимое входной строки изменилось или не подтверждено: {rid}')
            if saved.get('status') not in ('done', 'artist'):
                raise ValueError(f'строка не завершена: {rid}')
            valid.append((source, saved))
        if set(entries) != {source['row_id'] for source, _ in valid}:
            raise ValueError('состав безопасного состояния не совпадает с пакетом')
        if len(valid) + len(errors) != len(source_rows):
            raise ValueError('не все строки пакета учтены')

        by_id = {source['row_id']: (source, saved) for source, saved in valid}
        err_by_id = {source['row_id']: (source, reason) for source, reason in errors}
        ordered_ids = [source['row_id'] for source in source_rows]

        header = list(parsed.get('header') or [])
        if not header and parsed.get('source'):
            width = max((len(row) for row in parsed['source']), default=0)
            header = [f'колонка {index + 1}' for index in range(width)]
        if not header:
            header = ['№', 'промпт']
        header = [_text(item) for item in header] + ['статус', 'причина', 'файл']
        table = []
        manifest = []
        approved_pngs = []
        src_by_id = {source['row_id']: source for source in source_rows}
        for rid in ordered_ids:
            source = src_by_id[rid]
            raw = list((parsed.get('source') or [])[source.get('src', -1)]) if (
                type(source.get('src')) is int and 0 <= source.get('src') < len(parsed.get('source') or [])) else [source.get('n', rid), source.get('prompt', '')]
            if rid in err_by_id:
                status = 'ошибка'
                reason = err_by_id[rid][1]
                file_name = ''
                entry = {'row_id': rid, 'status': 'error', 'reason': reason}
            else:
                saved = by_id[rid][1]
                status = 'готово' if saved['status'] == 'done' else 'художнику'
                reason = _text(saved.get('reason', ''))
                file_name = ''
                entry = {'row_id': rid, 'status': saved['status'], 'reason': reason}
                payload = _safe_png(root, rid, saved)
                if payload is not None:
                    file_name = f'готово/{rid}.png'
                    approved_pngs.append((file_name, payload))
                    entry['file'] = file_name
            raw += [''] * max(0, len(header) - 3 - len(raw))
            table.append(raw[:len(header) - 3] + [status, reason, file_name])
            manifest.append(entry)

        xlsx_payload = _xlsx(header, table, parsed.get('errors', []))
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(REPORT_NAME, xlsx_payload)
            archive.writestr('manifest.json', json.dumps(manifest, ensure_ascii=False, indent=2))
            for name, payload in approved_pngs:
                archive.writestr(name, payload)
        # Ничего не публикуем, если источник/state изменились за время сборки.
        latest, _ = secure_delivery._batch_data(batch)
        if latest.get('rows') != source_rows or batch_mod.parse_batch(folder) != parsed:
            raise ValueError('входной пакет изменился во время сборки отчёта')
        for source, saved in valid:
            if secure_delivery._identity(source) != saved['identity']:
                raise ValueError('содержимое фото изменилось во время сборки отчёта')
        if secure_delivery._read(root) != secure_state:
            raise ValueError('безопасное состояние изменилось во время сборки отчёта')
        return {'xlsx': xlsx_payload, 'zip': zip_buffer.getvalue(),
                'rows': len(source_rows),
                'approved': sum(1 for _, saved in valid if saved['status'] == 'done')}
