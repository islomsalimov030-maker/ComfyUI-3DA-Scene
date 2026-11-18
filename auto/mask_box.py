"""Прямоугольная маска по силуэту выбранного кандидата."""
import base64
import json
import math
import zlib

import numpy as np
import torch
from PIL import Image, ImageDraw


class MaskBox:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'plate': ('IMAGE',), 'candidates_json': ('STRING', {'forceInput': True})},
                'optional': {
                    'silhouette': ('MASK', {'tooltip': 'Силуэт выбранного кандидата из 3DAPlaceCandidates'}),
                    'passport': ('STRING', {'forceInput': True, 'tooltip': 'JSON паспорта с ростом персонажа, если известен'}),
                    'pick': ('INT', {'default': -1, 'min': -1, 'max': 1000}),
                    'margin': ('FLOAT', {'default': 0.15, 'min': 0., 'max': 2., 'step': 0.01}),
                    'feather_px': ('INT', {'default': 24, 'min': 0, 'max': 1024}),
                    'min_frac': ('FLOAT', {'default': 0.05, 'min': 0., 'max': 1., 'step': 0.01}),
                    'max_frac': ('FLOAT', {'default': 1., 'min': 0.01, 'max': 1., 'step': 0.01})}}

    RETURN_TYPES = ('MASK', 'STRING', 'IMAGE')
    RETURN_NAMES = ('mask', 'box', 'preview')
    FUNCTION = 'execute'
    CATEGORY = '3DA/auto'

    def execute(self, plate, candidates_json, silhouette=None, pick=-1, margin=0.15, feather_px=24,
                min_frac=0.05, max_frac=1.0, passport=''):
        image = plate.detach().cpu().float().numpy()
        h, w = image.shape[1:3]
        warning = ''
        candidate, data = None, {}
        try:
            data = json.loads(candidates_json)
            selected = data.get('chosen') if int(pick) == -1 else int(pick)
            candidate = next((c for c in data['candidates'] if c.get('index') == selected), None)
            if candidate is None:
                raise ValueError('кандидат не найден')
            encoded = data.get('silhouettes', {}).get(str(selected))
            if encoded is not None:
                shape = encoded['shape']
                if (not isinstance(shape, list) or len(shape) != 2 or
                        any(not isinstance(v, int) or v <= 0 or v > 16384 for v in shape) or
                        shape[0] * shape[1] > 16777216):
                    raise ValueError('некорректный размер силуэта')
                packed = base64.b64decode(encoded['zlib_base64'], validate=True)
                decoder = zlib.decompressobj()
                raw = decoder.decompress(packed, shape[0] * shape[1] + 1)
                if len(raw) != shape[0] * shape[1] or not decoder.eof or decoder.unused_data:
                    raise ValueError('некорректные данные силуэта')
                arr = np.frombuffer(raw, dtype=np.uint8).reshape(shape).astype(np.float32) / 255
            else:
                if selected != data.get('chosen'):
                    raise ValueError('силуэт доступен только для выбранного кандидата; измените pick в 3DAPlaceCandidates')
                requested = data.get('pick', -1)
                if isinstance(requested, int) and requested >= 0 and requested != selected:
                    raise ValueError('3DAPlaceCandidates не выполнил запрошенный pick; возвращён другой кандидат')
                if silhouette is None:
                    raise ValueError('силуэт не подключён: candidates_json не содержит bbox силуэта')
                arr = silhouette.detach().cpu().float().numpy()
            if arr.ndim == 3:
                arr = arr[0]
            if arr.ndim != 2 or not np.isfinite(arr).all() or arr.max() <= 0:
                raise ValueError('силуэт пуст или некорректен')
            if arr.shape[1] * h != arr.shape[0] * w:
                raise ValueError('пропорции силуэта и локации не совпадают')
            resized = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)).resize((w, h), Image.Resampling.NEAREST)
            ys, xs = np.where(np.asarray(resized) >= 128)
            if not len(xs):
                raise ValueError('силуэт пуст после порога')
            x0, x1, y0, y1 = float(xs.min()), float(xs.max() + 1), float(ys.min()), float(ys.max() + 1)
        except Exception as exc:  # noqa: BLE001 — узел должен возвращать запасной прямоугольник
            warning = str(exc)
            try:
                pp = json.loads(passport) if isinstance(passport, str) and passport.strip() else passport
            except (ValueError, TypeError):
                pp = None
            if not isinstance(pp, dict):
                pp = data.get('passport', {}) if isinstance(data, dict) else {}
            if not isinstance(pp, dict):
                pp = {}
            character = pp.get('character', {})
            if not isinstance(character, dict):
                character = {}
            heights = [(candidate or {}).get('height_m'), character.get('height_m'), pp.get('height_m'), 2.3]
            height_m = 2.3
            for height in heights:
                try:
                    value = float(height)
                    if math.isfinite(value) and value > 0:
                        height_m = value
                        break
                except (ValueError, TypeError, OverflowError):
                    continue
            # Нет камеры/масштаба: рост даёт только грубую оценку, не перспективный расчёт.
            bh = h * min(0.8, max(0.2, 0.55 * height_m / 2.3))
            bw = bh * 0.4
            x0, x1, y0, y1 = (w - bw) / 2, (w + bw) / 2, h * 0.9 - bh, h * 0.9
        def number(value, default):
            try:
                value = float(value)
                return value if math.isfinite(value) else default
            except (ValueError, TypeError, OverflowError):
                return default

        m = min(2., max(0., number(margin, 0.15)))
        bw, bh = x1 - x0, y1 - y0
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        lo = min(1., max(0., number(min_frac, 0.05)))
        hi = min(1., max(lo, number(max_frac, 1.0)))
        bw = min(w * hi, max(w * lo, bw * (1 + 2 * m)))
        bh = min(h * hi, max(h * lo, bh * (1 + 2 * m)))
        x0, x1 = max(0, math.floor(cx - bw / 2)), min(w, math.ceil(cx + bw / 2))
        y0, y1 = max(0, math.floor(cy - bh / 2)), min(h, math.ceil(cy + bh / 2))
        x0, y0 = min(w - 1, x0), min(h - 1, y0)
        x1, y1 = max(x0 + 1, x1), max(y0 + 1, y1)
        mask = np.zeros((h, w), dtype=np.float32)
        feather = int(min(1024, max(0, number(feather_px, 24))))
        if feather:
            xx = np.minimum(np.arange(x1 - x0) + 1, np.arange(x1 - x0, 0, -1)) / feather
            yy = np.minimum(np.arange(y1 - y0) + 1, np.arange(y1 - y0, 0, -1)) / feather
            mask[y0:y1, x0:x1] = np.minimum(1, np.minimum(yy[:, None], xx[None, :]))
        else:
            mask[y0:y1, x0:x1] = 1
        info = {'x': x0, 'y': y0, 'w': x1 - x0, 'h': y1 - y0,
                'fraction': {'x': x0 / w, 'y': y0 / h, 'w': (x1 - x0) / w, 'h': (y1 - y0) / h},
                'warning': warning}
        previews = []
        for frame in image:
            preview = Image.fromarray((np.clip(frame[..., :3], 0, 1) * 255).astype(np.uint8))
            ImageDraw.Draw(preview).rectangle((x0, y0, x1 - 1, y1 - 1), outline=(255, 60, 20), width=max(1, min(w, h) // 256))
            previews.append(np.asarray(preview, dtype=np.float32) / 255.)
        if warning:
            print(f'[3DA] маска-прямоугольник: {warning}')
        return torch.from_numpy(mask[None]), json.dumps(info, ensure_ascii=False), torch.from_numpy(np.stack(previews))
