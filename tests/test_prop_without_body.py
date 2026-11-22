"""E17: предмет рисуется, даже если манекен целиком за кадром.

Было: пустой силуэт тела -> prepare_images бросал «renderer produced empty silhouette» -> 3DAPlacePose отдавал
fallback -> 3DANativeGuides переводил граф на обычное фото, предмет (голова, таз) пропадал. Теперь с предметом
пустое тело допустимо, запас вокруг предмета тот же (body_margin), что вокруг тела.
CPU, синтетика. Запуск: python -m pytest tests/test_prop_without_body.py -q"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools/scene3d'), str(ROOT / 'tools/scene3d/plate')]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pack_cli = load('threeda_pack_cli_e17', ROOT / 'tools/scene3d/pack_cli.py')
N = load('threeda_nodes_e17', ROOT / 'nodes.py')
N.config = lambda: {'repo': str(ROOT)}
import pose_proxy_v26 as proxy  # noqa: E402

W, H, CELL, BAND = 256, 160, 16, 16
PROP_BOX = (slice(70, 100), slice(40, 80))          # «предмет» (строки, столбцы)
BODY_BOX = (slice(50, 130), slice(170, 200))        # «манекен» (для сравнения с запасом тела)
ROOM = np.linspace(40, 200, W, dtype=np.uint8)[None, :, None].repeat(H, 0).repeat(3, 2)


def png(path, array, mode=None):
    Image.fromarray(array, mode).save(path)


def mask_of(box):
    m = np.zeros((H, W), np.uint8)
    m[box] = 255
    return m


def render_dir(tmp_path, body=None):
    """Каталог рендера 3DAPlacePose до prepare_images: playblast/silhouette (тело или пусто) и фото комнаты."""
    d = tmp_path / 'pose'
    d.mkdir()
    play = ROOM.copy()
    sil = np.zeros((H, W), np.uint8)
    if body is not None:
        play[body] = (170, 170, 170)
        sil = mask_of(body)
    png(d / 'playblast.png', play)
    png(d / 'silhouette.png', sil)
    plate = tmp_path / 'plate_photo.png'
    png(plate, ROOM)
    return d, plate


def prop_overlay(tmp_path, strength=.6, box=PROP_BOX):
    """Результат prop_render.py: цветной слой предмета и его силуэт."""
    o = tmp_path / 'props_render'
    o.mkdir()
    rgba = np.zeros((H, W, 4), np.uint8)
    rgba[box] = (200, 40, 40, 255)
    png(o / 'textured.png', rgba)
    png(o / 'sil.png', mask_of(box))
    (o / 'props_render.json').write_text(json.dumps({
        'shadow': None, 'textured_rgba': str(o / 'textured.png'),
        'props': [{'silhouette': str(o / 'sil.png'), 'prop_strength': strength}]}), encoding='utf-8')
    return o


def prepare(d, plate, allow):
    pack_cli.prepare_images(d, plate, 'grey', None, 1.0, CELL, (W, H), 1.0, allow_empty_body=allow)


def read(path):
    return np.asarray(Image.open(path).convert('L'), np.float32) / 255


def test_empty_body_without_props_still_raises(tmp_path):
    """Без предмета поведение прежнее: пустой силуэт -> ошибка -> обычное фото."""
    d, plate = render_dir(tmp_path)
    with pytest.raises(ValueError, match='empty silhouette'):
        prepare(d, plate, False)


def test_empty_body_with_props_gives_zero_body_strength(tmp_path):
    d, plate = render_dir(tmp_path)
    prepare(d, plate, True)
    assert read(d / 'strength.png').max() == 0
    assert (np.asarray(Image.open(d / 'init.png').convert('RGB')) == ROOM).all()


def test_color_proxy_with_empty_body_does_not_crash(tmp_path):
    """Режим color (V3.0) читает labels.png и body.json с цветами одежды — при пустом теле это пропускается."""
    d, plate = render_dir(tmp_path)
    pack_cli.prepare_images(d, plate, 'color', None, 1.0, CELL, (W, H), 1.0, allow_empty_body=True)
    assert not (d / 'proxy.png').exists()
    assert read(d / 'strength.png').max() == 0


@pytest.mark.parametrize('prop_strength', [.35, .6, .7, 1.0])   # голова, таз, stick, ... — любая сила предмета
def test_prop_keeps_strength_and_same_margin_without_body(tmp_path, prop_strength):
    d, plate = render_dir(tmp_path)
    prepare(d, plate, True)
    o = prop_overlay(tmp_path, prop_strength)
    pack_cli.compose_props(d, o, (W, H), BAND)
    s = read(d / 'strength.png')
    inside = s[PROP_BOX]
    assert np.allclose(inside, prop_strength, atol=.01), 'внутри предмета — prop_strength'
    # запас вокруг предмета тот же, что вокруг тела: полная сила (strength_map тела при тех же band/силе 1.0)
    expect = proxy.strength_map(mask_of(PROP_BOX) > 0, 1.0, BAND)
    ring = (expect > 0) & (mask_of(PROP_BOX) == 0)
    assert np.allclose(s[ring], expect[ring], atol=.01), 'запас вокруг предмета = запас тела (body_margin)'
    assert (s > 0).sum() > (mask_of(PROP_BOX) > 0).sum() * 2


def test_margin_around_prop_equals_margin_around_body(tmp_path):
    """Один и тот же профиль запаса: расстояние до нуля у предмета (без тела) = у тела (без предмета)."""
    d, plate = render_dir(tmp_path)
    prepare(d, plate, True)
    pack_cli.compose_props(d, prop_overlay(tmp_path, .35), (W, H), BAND)
    prop_s = read(d / 'strength.png')
    tmp2 = tmp_path / 'b'
    tmp2.mkdir()
    d2, plate2 = render_dir(tmp2, body=PROP_BOX)         # тот же прямоугольник, но как тело
    prepare(d2, plate2, False)
    body_s = read(d2 / 'strength.png')
    ring = mask_of(PROP_BOX) == 0
    assert np.allclose(prop_s[ring], body_s[ring], atol=.01)


def test_repair_mask_covers_prop_without_body(tmp_path):
    d, plate = render_dir(tmp_path)
    prepare(d, plate, True)
    pack_cli.compose_props(d, prop_overlay(tmp_path), (W, H), BAND)
    repair = read(d / 'prop_repair_mask.png') > 0
    assert repair[PROP_BOX].all(), 'предмет целиком в маске восстановления'


def test_nothing_in_frame_falls_back_as_before(tmp_path):
    """Ни тела, ни предмета в кадре — рисовать нечего: ошибка, как раньше (обычное фото)."""
    d, plate = render_dir(tmp_path)
    prepare(d, plate, True)
    o = prop_overlay(tmp_path, box=(slice(0, 0), slice(0, 0)))
    with pytest.raises(ValueError, match='no body and no props'):
        pack_cli.compose_props(d, o, (W, H), BAND)


def test_prop_survives_final_composite_without_body(tmp_path):
    """Сквозная проверка вклейки: карта силы без тела -> NativeComposite.compose_h3 -> предмет в итоге и в support."""
    d, plate = render_dir(tmp_path)
    prepare(d, plate, True)
    pack_cli.compose_props(d, prop_overlay(tmp_path, .35), (W, H), BAND)
    strength = torch.from_numpy(read(d / 'strength.png'))[None]
    room = torch.from_numpy(ROOM.astype(np.float32) / 255)[None]
    h3 = room.clone()
    h3[0][PROP_BOX] = torch.tensor([.2, .5, .9])            # H3 нарисовал предмет
    prop_alpha = torch.zeros(1, H, W)
    prop_alpha[0][PROP_BOX] = 1
    guides = {'validation': {'reasons': [], 'warnings': []}, 'plate': room.clone(), 'playblast': h3.clone(),
              'init': h3.clone(), 'strength': strength, 'prompt': 'x', 'prop_mode': 'h3',
              'prop_rgb': room.clone(), 'prop_alpha': prop_alpha,
              'clean_shadows': dict(person=torch.zeros(1, H, W), person_source='render', zone=None)}
    final, support, metrics = N.NativeComposite().compose_h3(h3.clone(), guides)
    core = (slice(PROP_BOX[0].start + 2, PROP_BOX[0].stop - 2), slice(PROP_BOX[1].start + 2, PROP_BOX[1].stop - 2))
    assert (support[0][core] > .95).all(), 'предмет целиком в маске вклейки'
    assert torch.allclose(final[0][core], h3[0][core], atol=1e-5), 'в итоге предмет от H3, а не комната'
    assert json.loads(metrics)['clean_shadows']['keep_px'] >= (mask_of(PROP_BOX) > 0).sum()
