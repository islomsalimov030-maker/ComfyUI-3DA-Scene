"""E6: дисковый кэш 3DAPlaceCandidates — повторный вызов с другим pick без пересчёта кандидатов.

Ключ — хэш входов узла без pick (его считает узел). Запись: <root>/<ключ>/summary.json (candidates_json без
выбора), sheet.npz (contact_sheet), cand_<номер>.npz (10 выходов 3DAPlacePose + отчёт кандидата).
Картинки и маски из PNG (значения k/255) хранятся как uint8 без потерь, прочие тензоры — как есть.
Запись идёт во временную папку и переименовывается в конце: недописанный кэш никогда не читается.
Любая ошибка кэша не мешает узлу — он просто считает заново.
"""
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np

VERSION = 'e6-cache-v2'
# сбой окружения (память GPU и т. п.), не свойство сцены: такую запись в кэш не кладём
TRANSIENT_MARKERS = ('out of memory', 'cuda')


def _torch():
    import torch
    return torch


def pack(value, arrays):
    """Выходы узла (тензоры, dict, list, строки, числа) → JSON-скелет + массивы для np.savez_compressed."""
    if isinstance(value, np.ndarray) or type(value).__name__ == 'Tensor':
        is_tensor = not isinstance(value, np.ndarray)
        a = value.detach().cpu().numpy() if is_tensor else value
        key = f'a{len(arrays)}'
        meta = {'__a': key, 'tensor': is_tensor, 'dtype': str(a.dtype)}
        if a.dtype in (np.float32, np.float64) and a.size and np.isfinite(a).all() and a.min() >= 0 and a.max() <= 1:
            q = np.rint(a * 255)
            if np.array_equal((q.astype(a.dtype) / 255).astype(a.dtype), a):
                arrays[key] = q.astype(np.uint8)
                meta['u8'] = True
                return meta
        arrays[key] = a
        return meta
    if isinstance(value, dict):
        if not all(isinstance(k, str) for k in value):
            raise TypeError('кэш: ключи dict — только строки')
        return {'__d': {k: pack(v, arrays) for k, v in value.items()}}
    if isinstance(value, (list, tuple)):
        return {'__l' if isinstance(value, list) else '__t': [pack(v, arrays) for v in value]}
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f'кэш: тип {type(value).__name__} не сохраняется')


def unpack(skel, arrays):
    if isinstance(skel, dict):
        if '__a' in skel:
            a = arrays[skel['__a']]
            if skel.get('u8'):
                a = (a.astype(skel['dtype']) / 255).astype(skel['dtype'])
            return _torch().from_numpy(np.ascontiguousarray(a)) if skel['tensor'] else a
        if '__d' in skel:
            return {k: unpack(v, arrays) for k, v in skel['__d'].items()}
        if '__l' in skel:
            return [unpack(v, arrays) for v in skel['__l']]
        if '__t' in skel:
            return tuple(unpack(v, arrays) for v in skel['__t'])
    return skel


def save_obj(path, value):
    arrays = {}
    skel = pack(value, arrays)
    np.savez_compressed(path, __skel=np.frombuffer(json.dumps(skel).encode('utf-8'), dtype=np.uint8), **arrays)


def load_obj(path):
    with np.load(path, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    return unpack(json.loads(arrays.pop('__skel').tobytes().decode('utf-8')), arrays)


class CandCache:
    """Один ключ: чтение готовой записи или запись новой (begin → put … → commit)."""

    def __init__(self, root, key, keep, tmp_age_s):
        self.root, self.key, self.keep, self.tmp_age_s = Path(root), key, int(keep), float(tmp_age_s)
        self.dir = self.root / key
        self.tmp = None

    # ------------------------------------------------------------ чтение
    def summary(self):
        path = self.dir / 'summary.json'
        if not path.is_file():
            return None
        data = json.loads(path.read_text(encoding='utf-8'))
        if data.get('cache_version') != VERSION:
            return None
        try:
            os.utime(self.dir)          # свежий — последним уйдёт при чистке
        except OSError:
            pass
        return data

    def candidate(self, index):
        """(outs, rep) кандидата с этим номером."""
        data = load_obj(self.dir / f'cand_{int(index)}.npz')
        return tuple(data['outs']), data['rep']

    def sheet(self):
        return load_obj(self.dir / 'sheet.npz')

    # ------------------------------------------------------------ запись
    def begin(self):
        self.root.mkdir(parents=True, exist_ok=True)
        self.tmp = self.root / f'tmp_{self.key}_{os.getpid()}_{int(time.time() * 1000)}'
        self.tmp.mkdir(parents=True)

    def put(self, index, outs, rep):
        save_obj(self.tmp / f'cand_{int(index)}.npz', {'outs': list(outs), 'rep': rep if isinstance(rep, dict) else {}})

    def commit(self, summary, sheet):
        save_obj(self.tmp / 'sheet.npz', sheet)
        (self.tmp / 'summary.json').write_text(json.dumps({**summary, 'cache_version': VERSION}, ensure_ascii=False,
                                                          default=str), encoding='utf-8')
        if self.dir.exists():
            shutil.rmtree(self.dir, ignore_errors=True)
        os.replace(self.tmp, self.dir)
        self.tmp = None
        self.prune()

    @staticmethod
    def transient_failure(entries):
        """E12: есть кандидат-fallback с ошибкой окружения (OOM/CUDA) — результат кэшировать нельзя:
        повторный вызов получил бы тот же fallback, хотя память уже свободна."""
        return any(e.get('status') == 'fallback'
                   and any(m in str(e.get('error') or '').lower() for m in TRANSIENT_MARKERS) for e in entries)

    def abort(self):
        if self.tmp is not None:
            shutil.rmtree(self.tmp, ignore_errors=True)
            self.tmp = None

    def prune(self):
        """Оставить keep последних записей; брошенные tmp_ старше tmp_age_s — тоже прочь."""
        now = time.time()
        done = []
        for p in self.root.iterdir():
            if not p.is_dir():
                continue
            if p.name.startswith('tmp_'):
                if now - p.stat().st_mtime > self.tmp_age_s:
                    shutil.rmtree(p, ignore_errors=True)
                continue
            done.append(p)
        done.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        for p in done[self.keep:]:
            shutil.rmtree(p, ignore_errors=True)
