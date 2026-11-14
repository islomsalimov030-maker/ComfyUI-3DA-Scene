"""0.3.26: режим маски (маска на «Фото 1», 3D-контроль выключен). Запуск: python -m pytest tests/test_mask_mode.py -q"""
import importlib.util
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
# standalone pack: nodes.py в корне; копия в ферме: comfyui/ComfyUI-3DA/nodes.py
NODES = ROOT / 'nodes.py' if (ROOT / 'nodes.py').is_file() else ROOT / 'comfyui/ComfyUI-3DA/nodes.py'
sys.path[:0] = [str(ROOT), str(ROOT / 'tools/scene3d')]
spec = importlib.util.spec_from_file_location('threeda_nodes_mask_test', NODES)
N = importlib.util.module_from_spec(spec)
spec.loader.exec_module(N)
N.config = lambda: {'repo': str(ROOT)}
import prompt_h3  # noqa: E402

CONNECTED = {'<Picture 1>', '<Picture 2>', '<Video 1>'}
SRC = torch.rand(1, 1072, 1872, 3)
MASK = torch.zeros(1, 1072, 1872)
MASK[0, 524:672, 1001:1131] = 1  # прямоугольник из теста 14.11


def run_mask(prompt='She wears a red knit sweater.'):
    return N.NativeGuides().execute(frame_prompt=prompt, user_refs=1, mode=1, image=SRC, mask=MASK, megapixels=2.0)


def test_mask_present_and_raster():
    assert not N.mask_present(None)
    assert not N.mask_present(torch.zeros(1, 64, 64))
    assert N.mask_present(MASK)
    assert N.raster_size(1872, 1072, 2.0) == (1920, 1088)


def test_guides_mask_mode():
    plate, playblast, init, strength, prompt, w, h, accepted, mode, status, guides = run_mask()
    assert (mode, accepted, w, h) == (2, True, 1920, 1088)
    assert plate.shape == playblast.shape == init.shape == (1, 1088, 1920, 3)
    assert strength.shape == (1, 1088, 1920)
    assert float(strength[0, 606, 1093]) == 1 and float(strength[0, 0, 0]) == 0
    assert status.startswith('Режим маски') and 'Предупреждение' not in status
    assert guides['mask_mode'] and guides['grow_px'] == 16
    assert 'red knit sweater' in prompt and 'mannequin' not in prompt
    assert not [i for i in prompt_h3.lint(prompt, True, CONNECTED) if i.startswith('error')]


def test_guides_other_cases():
    g = N.NativeGuides()
    out = g.execute(mode=1, image=SRC, mask=torch.zeros(1, 64, 64))
    assert out[8] == 1 and out[9].startswith('Режим 1') and out[10] == {}
    out = g.execute(mode=2, image=SRC, mask=MASK)
    assert out[8] == 1 and '3DA не сработал' in out[9]
    assert 'промпт кадра пустой' in run_mask('')[9]
    out = g.execute(mode=1, image=None, mask=MASK)
    assert out[8] == 1 and out[9].startswith('Режим маски не сработал')


def test_guides_3d_mode_ignores_mask():
    img = lambda: torch.rand(1, 128, 192, 3)
    guides = {'validation': {'reasons': [], 'warnings': []}, 'plate': img(), 'playblast': img(), 'init': img(),
              'strength': torch.zeros(1, 128, 192), 'prompt': 'x', 'prop_rgb': img(), 'prop_alpha': torch.zeros(1, 128, 192)}
    guides['strength'][0, 40:80, 60:100] = 1
    out = N.NativeGuides().execute(body_guides=guides, mode=2, mask=MASK)
    assert len(out) == 11 and out[8] == 2 and out[10] is guides
    assert 'маска на «Фото 1» не используется' in out[9]
    assert 'маска' not in N.NativeGuides().execute(body_guides=guides, mode=2)[9]


def test_composite_mask_mode():
    guides = run_mask()[10]
    node = N.NativeComposite()
    assert node.check_lazy_status(image=None, body_guides=guides) == []
    assert node.check_lazy_status(image=None, body_guides={'clean_shadows': {}}) == ['person_mask']
    generated = torch.zeros(1, 1088, 1920, 3)
    generated[..., 0] = 1.
    final, support, metrics = node.execute(generated, body_guides=guides)
    metrics = json.loads(metrics)
    assert final.shape == SRC.shape and metrics['mode'] == 'mask'
    assert metrics['outside_changed_pixels'] == 0 and metrics['outside_passed']
    outside = ~(support[0] > 0)
    assert torch.equal(final[0][outside], SRC[0][outside])
    assert abs(float(final[0, 598, 1066, 0]) - 1) < 1e-3 and float(final[0, 598, 1066, 1]) < 1e-3
    assert float(final[0, 598, 1131 + 14, 0]) > .99  # запас 16 px — новый кадр
    assert torch.equal(final[0, 598, 1131 + 40], SRC[0, 598, 1131 + 40])  # дальше — исходник


def test_mask_prompt_labels():
    text = prompt_h3.mask_prompt('Use the sweater from <Picture 3>, room as in <Picture 1>.', 1)
    assert '<Picture 3> is a reference image' in text and 'room as in <Picture 1>' in text
    assert not [i for i in prompt_h3.lint(text, True, CONNECTED | {'<Picture 3>'}) if i.startswith('error')]
    try:
        prompt_h3.mask_prompt('see <Picture 6>', 1)
    except ValueError:
        pass
    else:
        raise AssertionError('лишняя метка должна давать ошибку')
    assert prompt_h3.mask_prompt('', 1) == prompt_h3.MASK_SCENE


def test_aio_v331_labels():
    P = prompt_h3
    assert [P.user_label(n) for n in (1, 2, 3, 4, 5)] == ['<Picture 1>', '<Video 1>', '<Picture 3>', '<Picture 4>', '<Picture 5>']
    assert [P.mask_label(n) for n in (1, 2, 3, 5)] == ['<Picture 1>', '<Picture 1>', '<Picture 3>', '<Picture 5>']
