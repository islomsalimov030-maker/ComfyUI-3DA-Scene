"""Узлы пакетного входа (категория 3DA/auto): загрузка пакета, строка, сохранение, отчёт.

Ошибки не останавливают граф: узел пишет причину в текст/статус строки.
Драйвер фаз (web/autoBatch.js) ставит задачи «все строки фазы 1, потом фазы 2…»,
выставляя `row_index` и `phase` у узлов «строка» и «сохранить».
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from . import batch as batch_mod
from . import state as state_mod
from . import store
from . import report as report_mod
from .vision import VisionPick
from . import parse as parse_mod
from .mask_box import MaskBox
from .mask_box_candidates import MaskBoxCandidates
from .mask_box_pack import MaskBoxPack

CATEGORY = '3DA/auto'
N = batch_mod.N_PHOTOS


def parse_phases(text):
    out = []
    for part in str(text or '').replace(';', ',').split(','):
        part = part.strip()
        if part.isdigit() and int(part) not in out:
            out.append(int(part))
    return out or [1]


def blank():
    import torch
    return torch.zeros((1, 64, 64, 3), dtype=torch.float32)


def load_image(path):
    import torch
    with Image.open(path) as im:
        img = ImageOps.exif_transpose(im).convert('RGB')
    return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)[None]


def digest(*parts):
    return hashlib.sha1(json.dumps(parts, ensure_ascii=False, default=str).encode('utf-8')).hexdigest()


HINT_PHASE = 1   # E5: фаза разбора промпта — её текст (hint_json) «строка» отдаёт в следующих фазах


def saved_text(out_dir, row_state, phase):
    """Текст, сохранённый узлом «сохранить» для фазы строки (`p<фаза>_text.txt` из state.json); нет — ''."""
    try:
        for rel in (row_state or {}).get('files', {}).get(str(phase), []):
            if rel.endswith(f'p{phase}_text.txt'):
                return (Path(out_dir) / rel).read_text(encoding='utf-8')
    except OSError:
        pass
    return ''


def batch_data(value):
    """JSON с выхода «пакет: загрузка» (или имя пакета) → (данные пакета, состояние)."""
    value = str(value or '').strip()
    if value.startswith('{'):
        data = json.loads(value)
        if 'name' not in data:
            raise ValueError(data.get('error') or 'пакет не загружен')
        name, folder, out_dir = store.check_dirs(data['name'], data['dir'], data['out_dir'], allow_path=True)
        data = dict(data, name=name, dir=str(folder), out_dir=str(out_dir))
        st = state_mod.load(data['out_dir'])
        if st is None:
            raise ValueError('состояние пакета не найдено — перезапустите узел «пакет: загрузка»')
        return data, st
    name, parsed, st, out_dir = store.open_batch(value)
    return {'name': name, 'dir': parsed['dir'], 'out_dir': str(out_dir), 'rows': parsed['rows'], 'phases': []}, st


class AutoBatch:
    """Пакет из `input/auto/<пакет>/` (кнопка «Выбрать папку») → строки + состояние."""
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'batch': ('STRING', {'default': '', 'tooltip': 'Имя пакета в input/auto (ставит кнопка «Выбрать папку») или абсолютный путь к папке (кнопки «Запустить пакет» и «Скачать ZIP» работают только с именем пакета)'}),
            'phases': ('STRING', {'default': '1,2,3,4,5', 'tooltip': 'Фазы для кнопки «Запустить пакет», через запятую'})}}
    RETURN_TYPES = ('STRING', 'INT', 'STRING')
    RETURN_NAMES = ('batch', 'rows_count', 'summary')
    FUNCTION = 'execute'
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, batch, phases):
        try:
            name, folder, out_dir = store.resolve(batch)
            return digest(name, phases, store.fingerprint(folder))
        except Exception as exc:  # noqa: BLE001
            return digest('error', str(exc))

    def execute(self, batch, phases):
        try:
            name, parsed, st, out_dir = store.open_batch(batch)
            data = {'name': name, 'dir': parsed['dir'], 'out_dir': str(out_dir), 'kind': parsed['kind'],
                    'phases': parse_phases(phases),
                    'rows': [{k: v for k, v in r.items() if k != 'src'} for r in parsed['rows']]}
            text = batch_mod.summary(parsed, name)
            result = (json.dumps(data, ensure_ascii=False), len(parsed['rows']), text)
        except Exception as exc:  # noqa: BLE001 — показать причину на узле, не ронять граф
            text = f'Пакет не загружен: {exc}'
            result = (json.dumps({'error': text}, ensure_ascii=False), 0, text)
        return {'ui': {'text': [text]}, 'result': result}


class AutoRow:
    """Одна строка пакета: фото 1–5, промпт, моды; skip — фаза уже пройдена или строка с ошибкой."""
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'batch': ('STRING', {'forceInput': True}),
            'row_index': ('INT', {'default': 0, 'min': 0, 'max': 100000, 'tooltip': 'Номер строки с 0 (ставит драйвер фаз)'}),
            'phase': ('INT', {'default': 1, 'min': 1, 'max': 99, 'tooltip': 'Фаза (ставит драйвер фаз)'})}}
    # E5: hint_json — в конце (старые выходы не сдвигаются)
    RETURN_TYPES = ('IMAGE',) * N + ('INT', 'STRING', 'STRING', 'STRING', 'STRING', 'STRING', 'BOOLEAN', 'STRING')
    RETURN_NAMES = tuple(f'image{k}' for k in range(1, N + 1)) + (
        'n_images', 'prompt', 'mod_character', 'mod_face', 'mod_prop', 'row_id', 'skip', 'hint_json')
    FUNCTION = 'execute'
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, batch, row_index, phase):
        try:
            data, st = batch_data(batch)
            rid = st['order'][row_index] if row_index < len(st['order']) else None
            return digest(batch, row_index, phase, st['rows'].get(rid))
        except Exception as exc:  # noqa: BLE001
            return digest('error', str(exc))

    def execute(self, batch, row_index, phase):
        images, notes, row, skip, present, hint = [blank() for _ in range(N)], [], None, True, [], ''
        try:
            data, st = batch_data(batch)
            if not 0 <= row_index < len(data['rows']):
                raise IndexError(f'строки {row_index} нет: в пакете {len(data["rows"])} строк')
            row = data['rows'][row_index]
            skip = state_mod.skip(st['rows'].get(row['row_id']), phase)
            hint = saved_text(data['out_dir'], st['rows'].get(row['row_id']), HINT_PHASE)
            for k, path in enumerate(row['photos']):
                if not path:
                    continue
                try:
                    images[k] = load_image(path)
                    present.append(k + 1)
                except Exception as exc:  # noqa: BLE001
                    notes.append(f'фото {k + 1} не читается: {exc}')
        except Exception as exc:  # noqa: BLE001
            notes.append(str(exc))
        n_images = len(present)
        mods = (row or {}).get('mods', {})
        rid = (row or {}).get('row_id', '')
        missing = [k for k in range(1, N + 1) if k not in present]
        text = (f'строка {rid or "—"} · фаза {phase} · фото: {n_images}'
                + (f' (нет: {", ".join(map(str, missing))} — чёрные 64×64)' if missing else '')
                + (' · ПРОПУСК' if skip else '') + ''.join(f'\n{n}' for n in notes))
        return {'ui': {'text': [text]},
                'result': (*images, n_images, (row or {}).get('prompt', ''), mods.get('character', ''),
                           mods.get('face', ''), mods.get('prop', ''), rid, bool(skip), hint)}


class AutoSave:
    """Сохранить итог фазы строки в `output/auto/<пакет>/rows/<строка>/` и отметить фазу в state.json."""
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
                    'batch': ('STRING', {'forceInput': True}),
                    'row_index': ('INT', {'default': 0, 'min': 0, 'max': 100000}),
                    'phase': ('INT', {'default': 1, 'min': 1, 'max': 99})},
                'optional': {
                    'images': ('IMAGE',),
                    'text': ('STRING', {'forceInput': True}),
                    'status': ('STRING', {'default': '', 'tooltip': 'Новый статус строки: ' + ' | '.join(state_mod.STATUSES) + '; пусто — не менять'}),
                    'reason': ('STRING', {'default': '', 'tooltip': 'Причина (для художника / ошибки)'})}}
    RETURN_TYPES = ('STRING',)
    RETURN_NAMES = ('info',)
    FUNCTION = 'execute'
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float('nan')

    def execute(self, batch, row_index, phase, images=None, text=None, status='', reason=''):
        try:
            data, st = batch_data(batch)
            if not 0 <= row_index < len(data['rows']):
                raise IndexError(f'строки {row_index} нет: в пакете {len(data["rows"])} строк')
            rid = data['rows'][row_index]['row_id']
            if state_mod.skip(st['rows'].get(rid), phase):
                info = f'строка {rid} · фаза {phase}: пропуск (фаза уже пройдена или строка с ошибкой)'
            else:
                out_dir = Path(data['out_dir'])
                row_dir = out_dir / 'rows' / rid
                row_dir.mkdir(parents=True, exist_ok=True)
                for old in row_dir.glob(f'p{phase}_*'):
                    old.unlink()
                files = []
                if images is not None:
                    for k in range(images.shape[0]):
                        arr = np.clip(images[k].cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
                        path = row_dir / f'p{phase}_{k + 1}.png'
                        Image.fromarray(arr).save(path, compress_level=4)
                        files.append(path.relative_to(out_dir).as_posix())
                if text:
                    path = row_dir / f'p{phase}_text.txt'
                    path.write_text(str(text), encoding='utf-8')
                    files.append(path.relative_to(out_dir).as_posix())
                status = str(status or '').strip() or None
                row = state_mod.mark(out_dir, rid, phase, status=status, reason=(str(reason) if reason else None), files=files)
                info = (f'строка {rid} · фаза {phase}: сохранено файлов {len(files)}, статус '
                        f'{state_mod.STATUS_RU.get(row["status"], row["status"])}' + (f' — {row["reason"]}' if row.get('reason') else ''))
        except Exception as exc:  # noqa: BLE001
            info = f'не сохранено: {exc}'
        return {'ui': {'text': [info]}, 'result': (info,)}


class AutoReport:
    """Отчёт XLSX, раскладка `готово/` и `художнику/`, ZIP. Собирает, когда все строки закончены (или force)."""
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'batch': ('STRING', {'forceInput': True})},
                'optional': {'force': ('BOOLEAN', {'default': False, 'tooltip': 'Собрать отчёт и ZIP, даже если не все строки прошли все фазы'})}}
    RETURN_TYPES = ('STRING', 'STRING')
    RETURN_NAMES = ('summary', 'zip')
    FUNCTION = 'execute'
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float('nan')

    def execute(self, batch, force=False):
        zip_path = ''
        try:
            data, st = batch_data(batch)
            phases = [str(p) for p in data.get('phases') or []]
            unfinished = [rid for rid in st['order'] if st['rows'][rid]['status'] not in state_mod.FINAL
                          and not (phases and all(p in st['rows'][rid]['phase_done'] for p in phases))]
            if unfinished and not force:
                text = (report_mod.summary_text(st) + f'\nотчёт соберётся после последней строки (осталось {len(unfinished)})')
            else:
                result = store.build(data['name'], data['dir'], data['out_dir'])
                zip_path = result['zip']
                text = f'{result["summary"]}\nотчёт: {Path(result["report"]).name} · ZIP: {Path(zip_path).name}'
        except Exception as exc:  # noqa: BLE001
            text = f'отчёт не собран: {exc}'
        return {'ui': {'text': [text], 'zip': [zip_path]}, 'result': (text, zip_path)}


class ParseHint:
    """E5: ответ LLM о позе → hint_json для 3DAPlaceCandidates; битый ответ — запасной разбор по словам промпта."""
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'text': ('STRING', {'forceInput': True, 'tooltip': 'Ответ модели (JSON, можно в ```-блоке)'})},
                'optional': {'prompt': ('STRING', {'forceInput': True, 'tooltip': 'Исходный промпт строки — для запасного разбора'})}}
    RETURN_TYPES = ('STRING', 'BOOLEAN', 'STRING')
    RETURN_NAMES = ('hint_json', 'ok', 'note')
    FUNCTION = 'execute'
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    def execute(self, text, prompt=''):
        hint, ok, note = parse_mod.safe_parse(text, prompt)
        return {'ui': {'text': [hint + '\n' + note]}, 'result': (hint, ok, note)}


NODE_CLASS_MAPPINGS = {'3DAAutoBatch': AutoBatch, '3DAAutoRow': AutoRow, '3DAAutoSave': AutoSave, '3DAAutoReport': AutoReport,
                       '3DAVisionPick': VisionPick, '3DAParseHint': ParseHint, '3DAMaskBox': MaskBox,
                       '3DAMaskBoxCandidates': MaskBoxCandidates, '3DAMaskBoxPack': MaskBoxPack}
NODE_DISPLAY_NAME_MAPPINGS = {'3DAAutoBatch': '3DA · пакет: загрузка', '3DAAutoRow': '3DA · пакет: строка',
                              '3DAAutoSave': '3DA · пакет: сохранить', '3DAAutoReport': '3DA · пакет: отчёт',
                              '3DAVisionPick': '3DA · выбор кандидата (зрение)', '3DAParseHint': '3DA · разбор промпта',
                              '3DAMaskBox': '3DA · маска-прямоугольник',
                              '3DAMaskBoxCandidates': '3DA · маска-прямоугольник с постановкой',
                              '3DAMaskBoxPack': '3DA · передать силуэт кандидата'}
