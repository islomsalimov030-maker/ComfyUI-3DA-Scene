"""Явная передача силуэта вместе с JSON результата постановки."""
import base64
import json
import zlib

import numpy as np


class MaskBoxPack:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'candidates_json': ('STRING', {'forceInput': True}), 'silhouette': ('MASK',)}}

    RETURN_TYPES = ('STRING',)
    RETURN_NAMES = ('candidates_json',)
    FUNCTION = 'execute'
    CATEGORY = '3DA/auto'

    def execute(self, candidates_json, silhouette):
        try:
            data = json.loads(candidates_json)
            selected = data['chosen']
            if not isinstance(selected, int) or isinstance(selected, bool):
                raise ValueError('некорректный выбранный кандидат')
            entry = next(c for c in data['candidates'] if c.get('index') == selected)
            if entry.get('status') == 'skipped_time' or str(entry.get('status', '')).startswith(('fallback', 'error')):
                raise ValueError('кандидат не рассчитан')
            requested = data.get('pick', -1)
            if isinstance(requested, int) and requested >= 0 and requested != selected:
                raise ValueError('постановка вернула другой pick')
            arr = silhouette.detach().cpu().float().numpy()
            if arr.ndim == 3 and arr.shape[0] == 1:
                arr = arr[0]
            if (arr.ndim != 2 or not arr.size or arr.size > 16777216 or max(arr.shape) > 16384 or
                    not np.isfinite(arr).all() or arr.max() <= 0):
                raise ValueError('некорректный силуэт')
            raw = np.rint(np.clip(arr, 0, 1) * 255).astype(np.uint8)
            data['silhouettes'] = {str(selected): {'shape': list(raw.shape),
                'zlib_base64': base64.b64encode(zlib.compress(raw.tobytes())).decode('ascii')}}
            return (json.dumps(data, ensure_ascii=False),)
        except Exception as exc:  # noqa: BLE001 — сохраняем предупреждающий fallback узла MaskBox
            return (json.dumps({'candidates': [], 'error': str(exc)}, ensure_ascii=False),)
