"""Постановка и выбор маски из одного набора кандидатов."""
import json

if __package__ == 'auto':  # pytest может загружать auto как верхнеуровневый пакет
    from nodes import PlaceCandidates
else:
    from ..nodes import PlaceCandidates
from .mask_box import MaskBox


class MaskBoxCandidates:
    @classmethod
    def INPUT_TYPES(cls):
        base = PlaceCandidates.INPUT_TYPES()
        required = {'plate': ('IMAGE',), **base['required']}
        optional = dict(base.get('optional', {}))
        optional.update({
            'margin': ('FLOAT', {'default': 0.15, 'min': 0., 'max': 2., 'step': 0.01}),
            'feather_px': ('INT', {'default': 24, 'min': 0, 'max': 1024}),
            'min_frac': ('FLOAT', {'default': 0.05, 'min': 0., 'max': 1., 'step': 0.01}),
            'max_frac': ('FLOAT', {'default': 1., 'min': 0.01, 'max': 1., 'step': 0.01}),
        })
        return {'required': required, 'optional': optional}

    RETURN_TYPES = ('MASK', 'STRING', 'IMAGE', 'STRING')
    RETURN_NAMES = ('mask', 'box', 'preview', 'candidates_json')
    FUNCTION = 'execute'
    CATEGORY = '3DA/auto'

    def execute(self, plate, pick=-1, margin=0.15, feather_px=24, min_frac=0.05, max_frac=1., **inputs):
        box_options = dict(margin=margin, feather_px=feather_px, min_frac=min_frac, max_frac=max_frac,
                           passport=inputs.get('passport', ''))
        first_json = '{}'
        try:
            place = PlaceCandidates()
            first = place.execute(pick=-1, **inputs)
            first_json = first[10]
            original = json.loads(first_json)
            chosen = original.get('chosen') if int(pick) == -1 else int(pick)
            entry = next((c for c in original['candidates'] if c.get('index') == chosen), None)
            if entry is None:
                raise ValueError('кандидат не найден')
            status = str(entry.get('status', ''))
            if status == 'skipped_time' or status.startswith(('fallback', 'error')):
                raise ValueError('кандидат не был успешно рассчитан')
            selected = first
            if chosen != original.get('chosen'):
                selected = place.execute(pick=chosen, **inputs)
                again = json.loads(selected[10])
                if (again.get('cache') != 'hit' or again.get('chosen') != chosen or
                        again.get('candidates') != original['candidates']):
                    raise ValueError('не удалось подтвердить выбор из того же набора в кэше')
            mask, box, preview = MaskBox().execute(plate, selected[10], selected[3], pick=chosen, **box_options)
            return mask, box, preview, selected[10]
        except Exception as exc:  # noqa: BLE001 — резерв вместо неподтверждённого силуэта
            mask, box, preview = MaskBox().execute(plate, '{"candidates": []}', **box_options)
            info = json.loads(box)
            info['warning'] = 'выбор силуэта: ' + str(exc)
            return mask, json.dumps(info, ensure_ascii=False), preview, first_json
