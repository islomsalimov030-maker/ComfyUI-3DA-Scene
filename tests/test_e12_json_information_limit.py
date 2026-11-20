"""Контрпример: штатный JSON не содержит положения силуэта."""
import json

import torch

from test_candidates import load_nodes


def test_place_candidates_json_is_identical_for_different_silhouettes(monkeypatch):
    nodes = load_nodes()
    monkeypatch.setattr(nodes.time, 'monotonic', lambda: 10.0)
    entry = {'index': 0, 'status': 'ok', 'support_object': 'floor', 'variant': 'base',
             'score': 1.0, 'support_gap_cm': 0, 'max_penetration_cm': 0}
    stored = {'candidates': [entry], 'room_flags': [], 'limit_s': 10, 'pose_class': 'stand',
              'hint': {}, 'room_scale': 1, 'labels': {}}
    outs = [None] * 10
    outs[5] = 'ok\n{}'
    masks = []
    summaries = []
    for x in (5, 25):
        silhouette = torch.zeros((1, 40, 60))
        silhouette[:, 10:30, x:x + 10] = 1
        outs[3] = silhouette
        result = nodes.PlaceCandidates.finish(entry, outs, {}, stored, -1, 10.0, None, 'off')
        masks.append(result[3])
        summaries.append(json.loads(result[10]))
    assert not torch.equal(*masks)
    assert summaries[0] == summaries[1]
    assert 'silhouettes' not in summaries[0]
    assert 'bbox' not in summaries[0]['candidates'][0]
