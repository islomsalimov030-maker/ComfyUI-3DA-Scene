"""Публичная выдача: CPU-only, синтетический пакет и локальный HTTP."""
import asyncio
import io
import json
import zipfile

import numpy as np
import openpyxl
import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from test_secure_report import pack  # noqa: F401
from auto import secure_delivery as delivery, secure_routes


def test_public_delivery_end_to_end(pack):
    folder, root, write_rows, _ = pack
    # Используем настоящий план, карантин, ticket, проверку и retry-драйвер.
    delivery._write(root, {'rows': {}})
    calls = []
    def generate(seed, attempt):
        calls.append((seed, attempt))
        return np.full((1, 2, 2, 3), 0.25, dtype=np.float32)
    def inspect(frame, attempt):
        assert tuple(frame.shape) == (1, 2, 2, 3)
        assert abs(float(frame[0, 0, 0, 0]) - 63 / 255) < 1e-6
        return json.dumps({'one_person': True, 'pose_matches': True,
                           'on_expected_support': True, 'anatomy_ok': True,
                           'prop_ok': None, 'notes': ''})
    assert delivery.run_row('pack', 0, 10, generate, inspect)[0] == 'done'
    assert delivery.run_row('pack', 1, 20, generate, lambda *_: 'invalid')[0] == 'artist'
    assert calls == [(10, 0), (20, 0)]

    async def check():
        app = web.Application()
        routes = web.RouteTableDef()
        secure_routes.register(routes)
        app.add_routes(routes)
        async with TestClient(TestServer(app)) as client:
            for kind in ('xlsx', 'zip'):
                response = await client.get('/3da/auto/secure/' + kind, params={'batch': 'pack'})
                assert response.status == 200
                assert 'no-store' in response.headers['Cache-Control']
                payload = await response.read()
                if kind == 'xlsx':
                    wb = openpyxl.load_workbook(io.BytesIO(payload))
                    assert [wb.active.cell(i, 6).value for i in (2, 3, 4)] == ['готово', 'художнику', 'ошибка']
                    wb.close()
                else:
                    with zipfile.ZipFile(io.BytesIO(payload)) as z:
                        assert set(z.namelist()) == {'отчёт.xlsx', 'manifest.json', 'готово/001.png'}
                        assert [x['status'] for x in json.loads(z.read('manifest.json'))] == ['done', 'artist', 'error']
            for value in ('../pack', str(folder), '{"name":"pack"}', 'missing'):
                response = await client.get('/3da/auto/secure/zip', params={'batch': value})
                assert response.status == 400
                assert (await response.json())['ok'] is False
            # Повреждение одобренного кадра никогда не отдаёт старый архив.
            row = delivery._read(root)['rows']['001']
            approved = root / 'quarantine' / '001' / f"{row['attempt']}-{row['approved']}.png"
            original = approved.read_bytes()
            approved.write_bytes(b'tampered')
            response = await client.get('/3da/auto/secure/zip', params={'batch': 'pack'})
            assert response.status == 400
            approved.write_bytes(original)
            # Незавершённая строка блокирует всю публичную выдачу.
            state = delivery._read(root)
            state['rows']['002']['status'] = 'pending'
            delivery._write(root, state)
            response = await client.get('/3da/auto/secure/xlsx', params={'batch': 'pack'})
            assert response.status == 400
            state['rows']['002']['status'] = 'artist'
            delivery._write(root, state)
            # Изменённая строка не получает ранее одобренный кадр.
            write_rows('changed')
            for kind in ('zip', 'xlsx'):
                response = await client.get('/3da/auto/secure/' + kind, params={'batch': 'pack'})
                assert response.status == 400
            # Переобработка изменённой строки порождает новую привязку.
            assert delivery.run_row('pack', 0, 10, generate, inspect)[0] == 'done'
            response = await client.get('/3da/auto/secure/zip', params={'batch': 'pack'})
            assert response.status == 200
    asyncio.run(check())
