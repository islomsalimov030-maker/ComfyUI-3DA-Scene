"""Маршруты сервера `/3da/auto/*` (доступны и как `/api/3da/auto/*`).

POST upload       multipart: `batch`, затем пары `path` + файл (или файл с относительным именем) → input/auto/<пакет>/
GET  batches      список пакетов
GET  state?batch= (только имя пакета) состояние пакета (разбор папки и сведение state.json)
GET  zip?batch=   (только имя пакета) ZIP пакета (собирается, если устарел)
"""
import asyncio
from urllib.parse import quote

from aiohttp import web

from . import store
from .state import STATUSES


def error(text, status=400):
    return web.json_response({'ok': False, 'error': str(text)}, status=status)


async def upload(request):
    reader = await request.multipart()
    batch, path, saved, errors, status = None, None, [], [], 200
    while True:
        part = await reader.next()
        if part is None:
            break
        if part.filename is None:
            value = await part.text()
            if part.name == 'batch':
                batch = value
            elif part.name == 'path':
                path = value
            continue
        rel, path = path or part.filename, None
        try:
            if batch is None:
                raise ValueError('поле batch должно идти перед файлами')
            sink = store.Sink(store.target_path(batch, rel))
        except ValueError as exc:
            errors.append(f'{rel}: {exc}')
            status = 400
            await part.release()
            continue
        try:
            while chunk := await part.read_chunk(1 << 16):
                sink.write(chunk)
            size = sink.commit()
            saved.append({'path': rel, 'size': size})
        except OverflowError as exc:
            errors.append(str(exc))
            status = 413
            await part.release()
        except BaseException:
            sink.abort()
            raise
    if batch is None:
        return error('нет поля batch')
    return web.json_response({'ok': not errors, 'batch': batch, 'saved': saved, 'errors': errors},
                             status=status if errors else 200)


async def batches(request):
    return web.json_response({'batches': await asyncio.to_thread(store.batches)})


def state_json(value):
    name, parsed, state, out_dir = store.open_dirs(*store.resolve_name(value))
    return {'batch': name, 'rows_count': len(parsed['rows']), 'statuses': list(STATUSES),
            'rows': [dict(state['rows'][rid], row_id=rid) for rid in state['order']],
            'errors': parsed['errors'], 'updated': state.get('updated')}


async def state(request):
    try:
        return web.json_response(await asyncio.to_thread(state_json, request.query.get('batch', '')))
    except (ValueError, KeyError) as exc:
        return error(exc)


async def zip_file(request):
    try:
        value = request.query.get('batch', '')
        path = await asyncio.to_thread(store.zip_path, store.resolve_name(value))
    except (ValueError, KeyError) as exc:
        return error(exc)
    if not path.is_file():
        return error('архив не собран', 404)
    disposition = f"attachment; filename=\"batch.zip\"; filename*=UTF-8''{quote(path.name)}"
    return web.FileResponse(path, headers={'Content-Disposition': disposition})


def register(routes):
    routes.post('/3da/auto/upload')(upload)
    routes.get('/3da/auto/batches')(batches)
    routes.get('/3da/auto/state')(state)
    routes.get('/3da/auto/zip')(zip_file)


def register_server():
    """Подключить маршруты к PromptServer (без ComfyUI — тихо пропустить)."""
    try:
        from server import PromptServer
        routes = PromptServer.instance.routes
    except Exception:  # noqa: BLE001 — вне ComfyUI (тесты, check_compat)
        return False
    register(routes)
    return True
