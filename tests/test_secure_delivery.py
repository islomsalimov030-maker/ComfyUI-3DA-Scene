"""CPU-проверки изолированной выдачи. Только синтетические RGB-кадры."""
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from auto import secure_delivery as D, store, state, nodes  # noqa: E402


def answer(**updates):
    data = dict(one_person=True, pose_matches=True, on_expected_support=True,
                anatomy_ok=True, prop_ok=None, notes='')
    data.update(updates)
    return json.dumps(data)


@pytest.fixture
def pack(tmp_path, monkeypatch):
    store.set_roots(tmp_path / 'input' / 'auto', tmp_path / 'output' / 'auto')
    parsed = {'rows': [{'row_id': '001', 'errors': []}]}
    st = {'rows': {'001': {'status': 'parsed'}}}
    monkeypatch.setattr(nodes, 'batch_data', lambda batch: (parsed, st))
    yield tmp_path
    store._roots.clear()


def image(value=0.5):
    return np.full((1, 8, 9, 3), value, dtype=np.float32)


def test_checked_image_exact_and_mutation_isolated(pack):
    assert D.plan('pack', 0, 123) == (0, 123, True)
    source = image()
    frozen, ticket = D.stage('pack', 0, source, 0, 123)
    source[:] = 0
    assert frozen.numpy().mean() > .49
    D.adjudicate(ticket, answer())
    with zipfile.ZipFile(D.archive('pack')) as z:
        assert set(z.namelist()) == {'готово/001.png', 'manifest.json'}
        payload = z.read('готово/001.png')
        assert hashlib.sha256(payload).hexdigest() == ticket[3]
        pixels = np.asarray(Image.open(io.BytesIO(payload)))
        assert np.array_equal(pixels, (frozen.numpy()[0] * 255).astype(np.uint8))
        manifest = json.loads(z.read('manifest.json'))
        assert manifest[0]['status'] == 'done'
    assert D.plan('pack', 0, 123) == (0, 123, False)


@pytest.mark.parametrize('response', ['bad JSON', answer(notes='uncertain'), answer(one_person=False, anatomy_ok=False)])
def test_ambiguous_frame_quarantined_not_delivered(pack, response):
    D.plan('pack', 0, 0)
    _, ticket = D.stage('pack', 0, image(), 0, 0)
    assert D.adjudicate(ticket, response)[0] == 'artist'
    root = D._root('pack')
    (root / 'untrusted.png').write_bytes(b'not an image')
    with zipfile.ZipFile(D.archive('pack')) as z:
        assert z.namelist() == ['manifest.json']
        assert json.loads(z.read('manifest.json'))[0]['status'] == 'artist'
    assert list((root / 'quarantine').rglob('*.png'))


def test_retry_exactly_once_different_seed_and_resume_idempotent(pack):
    calls, seen = [], []
    def generate(seed, attempt):
        calls.append((seed, attempt))
        return image(.25 + attempt * .25)
    def inspect(frame, attempt):
        seen.append(frame.numpy().mean())
        return answer(pose_matches=False)
    assert D.run_row('pack', 0, D.MAX_SEED, generate, inspect)[0] == 'artist'
    assert calls == [(D.MAX_SEED, 0), (0, 1)]
    assert seen[1] > seen[0]
    assert D.run_row('pack', 0, D.MAX_SEED, generate, inspect)[0] == 'artist'
    assert len(calls) == 2
    assert len(list((D._root('pack') / 'quarantine').rglob('*.png'))) == 2


def test_retry_success_delivers_second_frame_only(pack):
    calls = []
    def generate(seed, attempt):
        calls.append(seed)
        return image(.2 + .5 * attempt)
    result = D.run_row('pack', 0, 13, generate, lambda frame, attempt: answer(pose_matches=bool(attempt)))
    assert result[0] == 'done' and calls == [13, 14]
    row = D._read(D._root('pack'))['rows']['001']
    with zipfile.ZipFile(D.archive('pack')) as z:
        assert hashlib.sha256(z.read('готово/001.png')).hexdigest() == row['frames']['1']
        assert row['frames']['0'] != row['approved']


def test_old_ticket_and_wrong_seed_cannot_release_frame(pack):
    D.plan('pack', 0, 4)
    _, old = D.stage('pack', 0, image(.1), 0, 4)
    assert D.adjudicate(old, answer(anatomy_ok=False))[0] == 'pending'
    with pytest.raises(ValueError):
        D.adjudicate(old, answer())
    with pytest.raises(ValueError):
        D.stage('pack', 0, image(.8), 0, 4)
    with pytest.raises(ValueError):
        D.stage('pack', 0, image(.8), 1, 4)
    with pytest.raises(ValueError):
        D.plan('pack', 0, 5)
    _, current = D.stage('pack', 0, image(.8), 1, 5)
    assert D.adjudicate(current, answer())[0] == 'done'


def test_no_replacement_and_no_batch_images(pack):
    D.plan('pack', 0, 1)
    D.stage('pack', 0, image(), 0, 1)
    with pytest.raises(ValueError):
        D.stage('pack', 0, image(.3), 0, 1)
    with pytest.raises(ValueError):
        D.stage('pack', 0, np.zeros((2, 8, 8, 3)), 0, 1)
    with pytest.raises(ValueError):
        D.stage('pack', 0, image(float('nan')), 0, 1)
    with pytest.raises(ValueError):
        D.archive('pack')


def test_tampering_quarantine_prevents_release(pack):
    D.plan('pack', 0, 1)
    _, ticket = D.stage('pack', 0, image(), 0, 1)
    path = next((D._root('pack') / 'quarantine').rglob('*.png'))
    path.write_bytes(b'changed')
    with pytest.raises(ValueError):
        D.adjudicate(ticket, answer())
    with pytest.raises(ValueError):
        D.archive('pack')


def test_exception_and_restart_never_regenerate_started_attempt(pack):
    calls = []
    def fail(seed, attempt):
        calls.append(seed)
        raise RuntimeError('local generation failure')
    assert D.run_row('pack', 0, 1, fail, lambda *_: answer())[0] == 'artist'
    D.run_row('pack', 0, 1, fail, lambda *_: answer())
    assert calls == [1]
    root = D._root('pack')
    data = D._read(root)
    data['rows']['001']['status'] = 'pending'  # имитация обрыва после записи started
    D._write(root, data)
    assert D.run_row('pack', 0, 1, fail, lambda *_: answer())[0] == 'artist'
    assert calls == [1]


def test_vlm_failure_holds_image(pack):
    def fail(*_):
        raise RuntimeError('vision failed')
    assert D.run_row('pack', 0, 2, lambda *_: image(), fail)[0] == 'artist'
    with zipfile.ZipFile(D.archive('pack')) as z:
        assert z.namelist() == ['manifest.json']


def test_legacy_zip_cannot_find_quarantine(pack):
    D.plan('pack', 0, 1)
    _, ticket = D.stage('pack', 0, image(), 0, 1)
    D.adjudicate(ticket, 'bad')
    from auto.report import make_zip
    legacy = store.root('output') / 'pack'
    legacy.mkdir(parents=True)
    with zipfile.ZipFile(make_zip(legacy, 'pack')) as z:
        assert z.namelist() == []
    assert not D._root('pack').is_relative_to(legacy)


def test_all_source_rows_required(pack, monkeypatch):
    D.run_row('pack', 0, 0, lambda *_: image(), lambda *_: answer())
    monkeypatch.setattr(nodes, 'batch_data', lambda batch: ({'rows': [{'row_id': '001'}, {'row_id': '002'}]}, {'rows': {}}))
    with pytest.raises(ValueError, match='все строки'):
        D.archive('pack')


@pytest.mark.parametrize('change', ['prompt', 'mods', 'photo'])
def test_reused_batch_changed_row_requires_new_generation(pack, monkeypatch, change):
    photo = pack / 'reference.png'
    photo.write_bytes(b'first synthetic reference')
    source = {'row_id': '001', 'errors': [], 'prompt': 'synthetic input A',
              'mods': {'mod_face': 'v1'}, 'photos': [str(photo)]}
    monkeypatch.setattr(nodes, 'batch_data', lambda batch: (
        {'rows': [source]}, {'rows': {'001': {'status': 'parsed'}}}))
    calls = []
    def generate(seed, attempt):
        calls.append((seed, attempt))
        return image()
    assert D.run_row('pack', 0, 7, generate, lambda *_: answer())[0] == 'done'
    archive = Path(D.archive('pack'))
    old_identity = D._read(D._root('pack'))['rows']['001']['identity']
    if change == 'photo':
        # Те же путь, размер и mtime: проверка должна зависеть от содержимого.
        stat = photo.stat()
        photo.write_bytes(b'other synthetic reference')
        import os
        os.utime(photo, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    elif change == 'mods':
        source['mods']['mod_face'] = 'v2'
    else:
        source['prompt'] = 'synthetic input B'
    with pytest.raises(ValueError, match='строки изменилось'):
        D.archive('pack')
    assert not archive.exists()
    assert D.plan('pack', 0, 7) == (0, 7, True)
    row = D._read(D._root('pack'))['rows']['001']
    assert row['identity'] != old_identity
    assert row['frames'] == {} and 'approved' not in row and 'started' not in row
    assert D.run_row('pack', 0, 7, generate, lambda *_: answer())[0] == 'done'
    assert calls == [(7, 0), (7, 0)]


def test_changed_row_rejects_old_ticket_even_for_identical_image(pack, monkeypatch):
    source = {'row_id': '001', 'errors': [], 'prompt': 'synthetic input A'}
    monkeypatch.setattr(nodes, 'batch_data', lambda batch: (
        {'rows': [source]}, {'rows': {'001': {'status': 'parsed'}}}))
    D.plan('pack', 0, 2)
    _, old = D.stage('pack', 0, image(), 0, 2)
    source['prompt'] = 'synthetic input B'
    with pytest.raises(ValueError):
        D.adjudicate(old, answer())
    with pytest.raises(ValueError):
        D.stage('pack', 0, image(), 0, 2)
    assert D.plan('pack', 0, 2) == (0, 2, True)
    _, current = D.stage('pack', 0, image(), 0, 2)
    assert current[3] == old[3]
    with pytest.raises(ValueError, match='устаревшая входная строка'):
        D.adjudicate(old, answer())
    assert D.adjudicate(current, answer())[0] == 'done'


def test_changed_photo_after_stage_rejects_ticket_and_archive(pack, monkeypatch):
    photo = pack / 'reference.png'
    photo.write_bytes(b'first')
    source = {'row_id': '001', 'errors': [], 'photos': [str(photo)]}
    monkeypatch.setattr(nodes, 'batch_data', lambda batch: (
        {'rows': [source]}, {'rows': {'001': {'status': 'parsed'}}}))
    D.plan('pack', 0, 1)
    _, ticket = D.stage('pack', 0, image(), 0, 1)
    photo.write_bytes(b'other')
    with pytest.raises(ValueError, match='строки изменилось'):
        D.adjudicate(ticket, answer())
    with pytest.raises(ValueError):
        D.archive('pack')
    assert not (D._root('pack') / 'delivery.zip').exists()


def test_source_change_during_inspect_never_releases_old_frame(pack, monkeypatch):
    source = {'row_id': '001', 'errors': [], 'prompt': 'synthetic A'}
    monkeypatch.setattr(nodes, 'batch_data', lambda batch: (
        {'rows': [source]}, {'rows': {'001': {'status': 'parsed'}}}))
    def inspect(*_):
        source['prompt'] = 'synthetic B'
        return answer()
    with pytest.raises(ValueError, match='строки изменилось'):
        D.run_row('pack', 0, 1, lambda *_: image(), inspect)
    with pytest.raises(ValueError):
        D.archive('pack')
    assert not (D._root('pack') / 'delivery.zip').exists()
    assert D.plan('pack', 0, 1) == (0, 1, True)


@pytest.mark.parametrize('replan', [False, True])
def test_source_change_during_generate_fails_closed(pack, monkeypatch, replan):
    source = {'row_id': '001', 'errors': [], 'prompt': 'synthetic A'}
    monkeypatch.setattr(nodes, 'batch_data', lambda batch: (
        {'rows': [source]}, {'rows': {'001': {'status': 'parsed'}}}))
    def generate(seed, attempt):
        source['prompt'] = 'synthetic B'
        if replan:  # имитация параллельного plan, перезаписавшего строку под новый источник
            assert D.plan('pack', 0, 5) == (0, 5, True)
        return image(.3)
    with pytest.raises(ValueError, match='строки изменилось'):
        D.run_row('pack', 0, 5, generate, lambda *_: answer())
    row = D._read(D._root('pack'))['rows']['001']
    if replan:
        assert row['identity'] == D._identity(source)
    assert row['frames'] == {} and row['status'] == 'pending' and 'approved' not in row
    assert not list((D._root('pack') / 'quarantine').rglob('*.png'))
    with pytest.raises(ValueError):
        D.archive('pack')
    assert not (D._root('pack') / 'delivery.zip').exists()


def test_plan_stage_nodes_reject_old_image_after_replan(pack, monkeypatch):
    source = {'row_id': '001', 'errors': [], 'prompt': 'synthetic A'}
    monkeypatch.setattr(nodes, 'batch_data', lambda batch: (
        {'rows': [source]}, {'rows': {'001': {'status': 'parsed'}}}))
    planner, staging = D.SecurePlan(), D.SecureStage()
    attempt, seed, active, old_token = planner.execute('pack', 0, 5)
    assert active
    old_image = image(.2)
    source['prompt'] = 'synthetic B'
    new_attempt, new_seed, active, new_token = planner.execute('pack', 0, 5)
    assert active and (attempt, seed) == (new_attempt, new_seed)
    assert old_token != new_token
    with pytest.raises(ValueError, match='токен'):
        staging.execute('pack', 0, old_image, attempt, seed, old_token)
    with pytest.raises(ValueError, match='токен'):
        staging.execute('pack', 0, old_image, attempt, seed, None)
    row = D._read(D._root('pack'))['rows']['001']
    assert row['frames'] == {} and row['status'] == 'pending'
    assert not list((D._root('pack') / 'quarantine').rglob('*.png'))
    # Возврат A после B тоже создаёт новую сессию: старый токен A не оживает.
    source['prompt'] = 'synthetic A'
    _, _, _, current_token = planner.execute('pack', 0, 5)
    with pytest.raises(ValueError, match='токен'):
        staging.execute('pack', 0, old_image, attempt, seed, old_token)
    _, ticket = staging.execute('pack', 0, image(.8), attempt, seed, current_token)
    assert D.adjudicate(ticket, answer())[0] == 'done'
    assert D.SecureStage.INPUT_TYPES()['required']['plan_token'][0] == '3DA_SECURE_PLAN'
    assert D.SecurePlan.RETURN_NAMES[-1] == 'plan_token'


def test_real_batch_json_snapshot_is_not_used_after_row_changes(tmp_path):
    import csv
    input_root, output_root = tmp_path / 'input' / 'auto', tmp_path / 'output' / 'auto'
    store.set_roots(input_root, output_root)
    try:
        folder = input_root / 'pack'
        folder.mkdir(parents=True)
        for name in ('one.png', 'two.png'):
            Image.new('RGB', (2, 2)).save(folder / name)
        table = folder / 'batch.csv'
        def save(prompt):
            with table.open('w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['n', 'photo1', 'photo2', 'prompt'])
                writer.writerow(['001', 'one.png', 'two.png', prompt])
        save('synthetic A')
        snapshot, _ = nodes.batch_data('pack')
        snapshot = json.dumps(snapshot)
        assert D.run_row(snapshot, 0, 2, lambda *_: image(), lambda *_: answer())[0] == 'done'
        save('synthetic B')
        with pytest.raises(ValueError, match='строки изменилось'):
            D.archive(snapshot)
        assert D.plan(snapshot, 0, 2) == (0, 2, True)
    finally:
        store._roots.clear()


def test_legacy_state_without_identity_is_not_resumed(pack):
    D.run_row('pack', 0, 3, lambda *_: image(), lambda *_: answer())
    root = D._root('pack')
    data = D._read(root)
    del data['rows']['001']['identity']
    D._write(root, data)
    assert D.plan('pack', 0, 3) == (0, 3, True)
    assert D._read(root)['rows']['001']['frames'] == {}


def test_restore_current_frozen_frame_and_reject_stale_ticket(pack):
    D.plan('pack', 0, 12)
    frozen, ticket = D.stage('pack', 0, image(.3), 0, 12)
    restored, restored_ticket = D.SecureRestore().execute('pack', 0)
    assert np.array_equal(frozen.numpy(), restored.numpy())
    assert restored_ticket == ticket
    assert D.adjudicate(restored_ticket, answer(pose_matches=False))[0] == 'pending'
    with pytest.raises(ValueError, match='не сохранён'):
        D.restore('pack', 0)
    _, second = D.stage('pack', 0, image(.8), 1, 13)
    with pytest.raises(ValueError):
        D.adjudicate(restored_ticket, answer())
    assert D.adjudicate(second, answer())[0] == 'done'
    with pytest.raises(ValueError, match='нет активного'):
        D.restore('pack', 0)


def test_restore_rejects_changed_input_and_corrupt_frame(pack, monkeypatch):
    source = {'row_id': '001', 'errors': [], 'prompt': 'synthetic A'}
    monkeypatch.setattr(nodes, 'batch_data', lambda batch: (
        {'rows': [source]}, {'rows': {'001': {'status': 'parsed'}}}))
    D.plan('pack', 0, 2)
    D.stage('pack', 0, image(), 0, 2)
    source['prompt'] = 'synthetic B'
    with pytest.raises(ValueError, match='строки изменилось'):
        D.restore('pack', 0)
    source['prompt'] = 'synthetic A'
    next((D._root('pack') / 'quarantine').rglob('*.png')).write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='карантине изменён'):
        D.restore('pack', 0)


def test_new_nodes_registered_without_old_state_changes(pack):
    before = state.new_row('1')
    assert nodes.NODE_CLASS_MAPPINGS['3DAAutoSecurePlan'] is D.SecurePlan
    assert nodes.NODE_CLASS_MAPPINGS['3DAAutoSecureRestore'] is D.SecureRestore
    D.run_row('pack', 0, 0, lambda *_: image(), lambda *_: answer())
    assert state.new_row('1') == before
    assert D.SecureStage.RETURN_TYPES == ('IMAGE', '3DA_SECURE_FRAME')
