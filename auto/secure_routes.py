"""Отдельная публичная выдача завершённого secure_report.

Не читает legacy ZIP и не публикует файлы на диске. Каждый запрос заново
проверяет текущий пакет и возвращает только результат secure_report.build.
Доступ защищается средствами самого сервера ComfyUI, не этим модулем.
"""
import asyncio
from aiohttp import web
from . import secure_report, store


def _build(name):
    name, folder, _ = store.resolve_name(name)
    # Публичный API принимает только реальные папки непосредственно в input/auto.
    base = store.root('input')
    for path in (base, folder):
        if path.is_symlink() or getattr(path, 'is_junction', lambda: False)():
            raise ValueError('ссылки запрещены')
    if not folder.is_dir() or folder.resolve().parent != base.resolve():
        raise ValueError('пакет недоступен')
    result = secure_report.build(name)
    if not result['rows']:
        raise ValueError('пакет пуст')
    return result


async def _download(request, kind):
    try:
        result = await asyncio.to_thread(_build, request.query.get('batch', ''))
    except Exception:  # Не раскрываем пути и подробности состояния в HTTP.
        return web.json_response({'ok': False, 'error': 'безопасная выдача недоступна'},
                                 status=400, headers={'Cache-Control': 'no-store'})
    mime = ('application/zip' if kind == 'zip' else
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    return web.Response(body=result[kind], content_type=mime, headers={
        'Content-Disposition': f'attachment; filename="report.{kind}"',
        'Cache-Control': 'no-store, private',
        'X-Content-Type-Options': 'nosniff',
    })


async def xlsx(request):
    return await _download(request, 'xlsx')


async def zip_file(request):
    return await _download(request, 'zip')


def register(routes):
    routes.get('/3da/auto/secure/xlsx', allow_head=False)(xlsx)
    routes.get('/3da/auto/secure/zip', allow_head=False)(zip_file)


def register_server():
    try:
        from server import PromptServer
        routes = PromptServer.instance.routes
    except Exception:
        return False
    register(routes)
    return True
