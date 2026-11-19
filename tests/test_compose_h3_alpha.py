"""E14 (0.3.46): вклейка compose_h3/compose и «чистые тени» — внутри фигуры и предмета нет полупрозрачности.

Синтетика: комната-градиент, «тело» и «предмет» прямоугольниками в strength, кадр H3 с другими цветами
и «волосами» за силуэтом. Запуск: python -m pytest tests/test_compose_h3_alpha.py -q"""
import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
NODES = ROOT / 'nodes.py' if (ROOT / 'nodes.py').is_file() else ROOT / 'comfyui/ComfyUI-3DA/nodes.py'
sys.path[:0] = [str(ROOT), str(ROOT / 'tools/scene3d')]
spec = importlib.util.spec_from_file_location('threeda_nodes_compose_test', NODES)
N = importlib.util.module_from_spec(spec)
spec.loader.exec_module(N)
N.config = lambda: {'repo': str(ROOT)}

H, W = 128, 192
ROOM = torch.linspace(0.2, 0.8, W).view(1, 1, W, 1).expand(1, H, W, 3).clone()  # градиент-«комната»
BODY = (slice(40, 100), slice(60, 110))    # «тело»
PROP = (slice(80, 110), slice(120, 160))   # «предмет»
HAIR = (slice(30, 60), slice(105, 125))    # «волосы» — за силуэтом тела, H3 их нарисовал


def scene(prop_mode='h3', silhouette_hair=False):
    strength = torch.zeros(1, H, W)
    strength[0][BODY] = 1
    if prop_mode == 'h3':
        strength[0][PROP] = 1
    h3 = ROOM.clone()
    h3[0][BODY] = torch.tensor([.9, .3, .2])
    h3[0][PROP] = torch.tensor([.2, .5, .9])
    h3[0][HAIR] = torch.tensor([.1, .1, .1])  # волосы рисует H3, в strength их нет
    sil = torch.zeros(1, H, W)
    sil[0][BODY] = 1
    if silhouette_hair:
        sil[0][HAIR] = 1
    prop_alpha = torch.zeros(1, H, W)
    prop_alpha[0][PROP] = 1
    guides = {'validation': {'reasons': [], 'warnings': []}, 'plate': ROOM.clone(), 'playblast': h3.clone(),
              'init': h3.clone(), 'strength': strength, 'prompt': 'x', 'prop_mode': prop_mode,
              'prop_rgb': ROOM.clone(), 'prop_alpha': prop_alpha, 'clean_shadows': dict(person=sil, zone=None)}
    return h3, guides


def inside(mask_rect):
    return tuple(slice(1, None) if s.stop - s.start > 2 else s for s in mask_rect)  # без края


def test_compose_h3_body_prop_opaque():
    h3, guides = scene()
    final, support, _ = N.NativeComposite().compose_h3(h3.clone(), guides)
    for rect in (BODY, PROP):
        core = (slice(rect[0].start + 6, rect[0].stop - 6), slice(rect[1].start + 6, rect[1].stop - 6))
        mid = (support[0][core] > .05) & (support[0][core] < .95)
        assert int(mid.sum()) == 0, f'полупрозрачные пиксели внутри фигуры: {int(mid.sum())}'
        assert torch.allclose(final[0][core], h3[0][core], atol=1e-5), 'внутри фигуры итог должен быть кадром H3'


def test_compose_h3_person_mask_union_keeps_hair():
    """SAM3-маска с мягким краем (0.3) + волосы за силуэтом рендера: после фикса волосы остаются от H3."""
    h3, guides = scene()
    sam = torch.zeros(1, H, W)
    sam[0][BODY] = 1
    sam[0][HAIR] = .3                                   # мягкий край/полупрозрачные пряди
    guides['clean_shadows']['person_source'] = 'render'
    final, support, metrics = N.NativeComposite().compose_h3(h3.clone(), guides, person_mask=sam)
    info = json_metrics(metrics)['clean_shadows']
    assert info['applied']
    hair = (slice(HAIR[0].start + 2, HAIR[0].stop - 2), slice(HAIR[1].start + 2, HAIR[1].stop - 2))
    keep = sam[0] > 0
    hair_kept = keep[hair]
    assert hair_kept.all() and torch.allclose(final[0][hair], h3[0][hair], atol=1e-5), \
        'волосы за силуэтом (SAM3 > 0) не должны подмешивать комнату'


def test_compose_h3_render_silhouette_grow_keeps_hair_band():
    """Без SAM3: полоса вокруг силуэта рендера (2% высоты) сохраняет кадр H3 у самого контура тела;
    дальше контура у системы нет данных о фактической фигуре — там работает чистая тень (допускается)."""
    h3, guides = scene()
    final, support, metrics = N.NativeComposite().compose_h3(h3.clone(), guides)
    info = json_metrics(metrics)['clean_shadows']
    grow = max(2, int(round(H * .02)))
    near = (slice(BODY[0].start + grow, HAIR[0].stop - 2), slice(BODY[1].stop, BODY[1].stop + grow - 1))
    assert torch.allclose(final[0][near], h3[0][near], atol=1e-5), 'полоса запаса у силуэта должна хранить кадр H3'
    far = (slice(BODY[0].start + grow, HAIR[0].stop - 2), slice(BODY[1].stop + grow + 2, HAIR[1].stop - 2))
    changed = (final[0][far] - h3[0][far]).abs().max() > 1e-5
    assert bool(changed), 'чистая тень вне полосы запаса должна по-прежнему применяться'


def test_compose_h3_shadow_still_works():
    """Тень вне человека (в области H3-силы, вне силуэта) по-прежнему затемняется: комната × отношение
    яркостей H3 — текстура пола родная, форма тени от H3."""
    h3, guides = scene()
    zone = torch.zeros(1, H, W)
    zone[0][105:120, 65:105] = 1                            # «тень» на полу под телом, вне силуэта
    h3[0][105:120, 65:105] = ROOM[0][105:120, 65:105] * .5  # H3 затемнил пол
    guides['clean_shadows']['zone'] = zone
    final, _, metrics = N.NativeComposite().compose_h3(h3.clone(), guides)
    info = json_metrics(metrics)['clean_shadows']
    assert info['applied'] and info['darkened_px'] > 0
    patch = (slice(108, 118), slice(70, 100))
    ratio = (final[0][patch] / torch.clamp(ROOM[0][patch], min=1e-6)).mean()
    assert .4 < float(ratio) < .95, 'тень должна остаться затемнением комнаты'


def json_metrics(metrics):
    import json
    return json.loads(metrics)
