"""Portable analysis kernel. See docs/extended_protocol.md for scope, provenance and I/O."""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
import representation_core as RC
SUMMARY_CAP = 262144
SUPPRESS_LOW, SUPPRESS_HIGH = (1, 4)
FORBIDDEN_KEY_PARTS = ('subject', 'participant', 'person_id', 'person_index', 'walk_id', 'walk_index', 'per_walk', 'embedding', 'state_dict', 'weights_array', 'logits_rows', 'prediction_rows', 'label_rows', 'password', 'secret', 'token')

def clean(obj):
    return RC.clean(obj)

def _forbidden(obj, found):
    if isinstance(obj, dict):
        for k, v in obj.items():
            low = str(k).lower()
            found.extend((p for p in FORBIDDEN_KEY_PARTS if p in low))
            _forbidden(v, found)
    elif isinstance(obj, list):
        for v in obj:
            _forbidden(v, found)
    return found

def render(obj):
    data = clean(obj)
    bad = _forbidden(data, [])
    if bad:
        raise C.SourceError('ABCD_RESULT_FORBIDDEN_KEY:' + bad[0])
    text = json.dumps(data, indent=1, allow_nan=False) + '\n'
    if len(text.encode('utf-8')) > SUMMARY_CAP:
        raise C.SourceError('ABCD_RESULT_TOO_LARGE')
    return text

def save_aggregate(path, obj):
    path = Path(path)
    if path.exists():
        raise C.SourceError('ABCD_RESULT_EXISTS')
    text = render(obj)
    tmp = path.with_name(path.name + '.tmp')
    with open(tmp, 'x', encoding='utf-8') as fh:
        fh.write(text)
    os.replace(tmp, path)
    return len(text.encode('utf-8'))

def pooled_recall_rows(conf_by_site):
    pooled = sum((np.asarray(c).reshape(-1, 3, 3).sum(axis=0) for c in conf_by_site.values()))
    rows = {}
    for k in range(3):
        row = pooled[k]
        if np.any((row >= SUPPRESS_LOW) & (row <= SUPPRESS_HIGH)) or row.sum() == 0:
            rows[f'grade_{k}'] = 'SUPPRESSED'
        else:
            rows[f'grade_{k}'] = {f'pred_{j}': float(row[j] / row.sum()) for j in range(3)}
    return rows
