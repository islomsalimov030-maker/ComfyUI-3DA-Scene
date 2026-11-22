"""E14: ручной сдвиг/поворот манекена в 3D-окне — рендер совпадает с окном по высоте.

Манекен Стейси (MPFB) после позы на MHR получает вертикальный «контактный» сдвиг: у зоны касания комнаты
(стопы, таз) такой же зазор, как у MHR (pose_tools.contact_shift). У базовой постановки окно показывает
именно этот манекен (body_mpfb.obj). После ручного сдвига рендер раньше пересчитывал контактный сдвиг по сдвинутому MHR:
на неровной геометрии комнаты (сиденье, ковёр, наклон пола, ошибка глубины) он давал другое число — фигура в рендере
оказывалась выше или ниже, чем стояла в окне (до 4 см на сидячих, до 0,7 см на стоячих сценах; при повороте ещё и боком).
Здесь: базовый сдвиг запоминается рядом с кэшем постановки, а сдвинутый рендер получает тот же сдвиг, повёрнутый
вместе с телом (как это делает гизмо окна). Нет запомненного сдвига (старая постановка) или правятся ручки позы —
поведение прежнее."""
import json
from pathlib import Path

import numpy as np

NAME = 'body_contact.json'


def save_base_contact(cache_dir, model):
    """Контактный сдвиг и масштаб манекена базовой (не сдвинутой) постановки из body_model.json рендера. -> записан ли."""
    dz = (model or {}).get('contact_dz_exact_m')
    if dz is None:
        return False
    data = {'contact_dz_m': float(dz)}
    if model.get('scale') is not None:
        data['scale'] = float(model['scale'])
    Path(cache_dir, NAME).write_text(json.dumps(data), encoding='utf-8')
    return True


def carry_vector(cache_dir, transform, base_kp):
    """Вектор (x, y, z), м: базовый контактный сдвиг (0, 0, dz), перенесённый жёстким преобразованием гизмо.
    transform(X, KP) -> X' — то же преобразование, что применяется к телу (manual_pose.transform). None — нет базового сдвига."""
    path = Path(cache_dir, NAME)
    if not path.is_file():
        return None
    dz = float(json.loads(path.read_text(encoding='utf-8'))['contact_dz_m'])
    probe = np.array([[0., 0., 0.], [0., 0., 1.]])
    moved = np.asarray(transform(probe, base_kp), float)
    return (moved[1] - moved[0]) * dz


def carry_scale(cache_dir, transform, base_kp):
    """E15: масштаб манекена после гизмо = базовый масштаб (у базовой постановки он уже мог упереться в предел 0,85–1,15)
    × масштаб гизмо; окно масштабирует тот же базовый манекен. None — базовый масштаб не запомнен (старая постановка)."""
    path = Path(cache_dir, NAME)
    if not path.is_file():
        return None
    base = json.loads(path.read_text(encoding='utf-8')).get('scale')
    if base is None:
        return None
    probe = np.array([[0., 0., 0.], [0., 0., 1.]])
    moved = np.asarray(transform(probe, base_kp), float)
    return float(base) * float(np.linalg.norm(moved[1] - moved[0]))


def render_args(cache_dir, transform, base_kp):
    """Ключи командной строки Blender-рендера для сдвинутого/правленого тела (--contact-carry, --body-scale) и их значения для timings.
    Нет запомненной базовой посадки (постановка до E14) — пусто, рендер как раньше."""
    vector = carry_vector(cache_dir, transform, base_kp)
    if vector is None:
        return [], {}
    args = ['--contact-carry=' + ','.join(repr(float(x)) for x in vector)]      # «=»: вектор может начинаться с «-»
    info = {'contact_carry_m': [round(float(x), 5) for x in vector]}
    scale = carry_scale(cache_dir, transform, base_kp)
    if scale is not None:
        args.append('--body-scale=' + repr(scale))
        info['body_scale'] = round(scale, 5)
    return args, info


def contact_translation(carry, compute):
    """Контактный сдвиг манекена: перенесённый из базовой постановки (carry — вектор) или посчитанный по сцене
    (compute() -> dz по оси Z, 0 = нет данных). -> (вектор (x, y, z) | None, dz для отчёта, источник)."""
    if carry is not None:
        v = tuple(float(x) for x in carry)
        return v, v[2], 'base'
    dz = compute()
    return ((0., 0., dz) if dz else None), dz, 'scene'
