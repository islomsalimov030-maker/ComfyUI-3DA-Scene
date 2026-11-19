"""Thin ComfyUI bridge to the farm CLI; no geometry/model deps in ComfyUI env."""
# pytest из папки с дефисом (wt-e4, ComfyUI-3DA-Scene) грузит этот файл как отдельный модуль без пакета —
# тогда регистрировать нечего; ComfyUI всегда импортирует его как пакет.
if __package__:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

    # Пакетный вход (3DA/auto): отдельный модуль; его сбой не мешает старым узлам.
    try:
        from .auto.nodes import NODE_CLASS_MAPPINGS as _AUTO_NODES, NODE_DISPLAY_NAME_MAPPINGS as _AUTO_NAMES
        from .auto.routes import register_server as _register_auto_routes
        NODE_CLASS_MAPPINGS.update(_AUTO_NODES)
        NODE_DISPLAY_NAME_MAPPINGS.update(_AUTO_NAMES)
        _register_auto_routes()
    except Exception as _exc:  # noqa: BLE001
        print(f'[3DA] пакетный вход (auto) не загружен: {_exc}')

    # Независимая регистрация: старые маршруты и узлы не меняются.
    try:
        from .auto.secure_routes import register_server as _register_secure_routes
        _register_secure_routes()
    except Exception as _exc:  # noqa: BLE001
        print(f'[3DA] безопасная выдача не загружена: {_exc}')

WEB_DIRECTORY = './web'

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
