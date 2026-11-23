"""Portable analysis kernel. See docs/provenance_protocol.md for scope, provenance and I/O."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import aggregate_core as IC
import encoder_inputs as F
COHORTS = F.COHORTS
FEAT = 263
MIN_WALK_COVERAGE = 0.9
MIN_PARTICIPANT_COVERAGE = 0.95
MIN_CLEAN_ARRAY_SHARE = 0.99
MIN_SPEED_RANK_AGREEMENT = 0.7
EXPECTED_STATE_TENSORS = 72

def parse_key(key):
    if not isinstance(key, str) or key.count('__') != 1:
        return None
    subject, walk = key.split('__')
    has_down = '_down' in walk
    walk = walk.split('_down')[0]
    if not subject or not walk:
        return None
    return (subject, walk, 'trimmed' in key.lower(), has_down)

def canonical_index(cohort_obj):
    out = {}
    for subject, walks in cohort_obj.items():
        for walk, rec in walks.items():
            trans = np.asarray(rec['trans'], dtype=np.float64)
            out[str(subject), walk] = (int(trans.shape[0]), float(rec['fps']), trans)
    return out

def first_view(arr):
    a = np.asarray(arr)
    if a.dtype.kind != 'f':
        return None
    if a.ndim == 3:
        a = a[0]
    if a.ndim != 2 or a.shape[1] != FEAT or a.shape[0] < 2:
        return None
    return a

def speed_from_humanml(raw, fps):
    v = np.asarray(raw[:, 1:3], dtype=np.float64)
    return float(np.linalg.norm(v, axis=1).mean() * fps)

def rank_corr(a, b):
    a, b = (np.asarray(a, float), np.asarray(b, float))
    if a.size < 5 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    ra, rb = (a.argsort().argsort().astype(float), b.argsort().argsort().astype(float))
    return float(np.corrcoef(ra, rb)[0, 1])

def join(npz_keys, index):
    chosen, n_trim, n_down, n_bad, n_no_match = ({}, 0, 0, 0, 0)
    for key in npz_keys:
        parsed = parse_key(key)
        if parsed is None:
            n_bad += 1
            continue
        subject, walk, trimmed, has_down = parsed
        if trimmed:
            n_trim += 1
            continue
        n_down += int(has_down)
        if (subject, walk) not in index:
            n_no_match += 1
            continue
        prev = chosen.get((subject, walk))
        if prev is None or (prev[1] and (not has_down)):
            chosen[subject, walk] = (key, has_down, prev[2] + 1 if prev else 1)
        else:
            chosen[subject, walk] = (prev[0], prev[1], prev[2] + 1)
    return (chosen, {'trimmed': n_trim, 'down': n_down, 'unparsed': n_bad, 'unmatched': n_no_match})

def cohort_report(npz, index, speed_fn):
    chosen, skipped = join(list(npz.files), index)
    participants_all = {s for s, _ in index}
    participants_hit = {s for s, _ in chosen}
    diffs, ratios, s_in, s_old, n_bad_array, n_nonfinite, aug = ([], [], [], [], 0, 0, [])
    for (subject, walk), (key, _down, _n) in chosen.items():
        arr = np.asarray(npz[key])
        aug.append(int(arr.shape[0]) if arr.ndim == 3 else 1)
        raw = first_view(arr)
        if raw is None:
            n_bad_array += 1
            continue
        if not np.isfinite(raw).all():
            n_nonfinite += 1
            continue
        frames, fps, trans = index[subject, walk]
        diffs.append(raw.shape[0] - frames)
        ratios.append(raw.shape[0] / frames)
        s_in.append(speed_from_humanml(raw, fps))
        s_old.append(speed_fn(trans, fps))
    n_ok = len(s_in)
    s_in_a, s_old_a = (np.asarray(s_in), np.asarray(s_old))
    pos = s_old_a > 1e-09
    return {'n_npz_keys': IC.count(len(npz.files)), 'n_keys_trimmed': IC.count(skipped['trimmed']), 'n_keys_down_suffix': IC.count(skipped['down']), 'n_keys_unparsed': IC.count(skipped['unparsed']), 'n_keys_without_v4_walk': IC.count(skipped['unmatched']), 'n_walks_v4': IC.count(len(index)), 'n_walks_matched': IC.count(len(chosen)), 'n_walks_with_several_keys': IC.count(sum((1 for v in chosen.values() if v[2] > 1))), 'n_walks_only_down_key': IC.count(sum((1 for v in chosen.values() if v[1]))), 'n_participants_v4': IC.count(len(participants_all)), 'n_participants_matched': IC.count(len(participants_hit)), 'n_arrays_bad_shape': IC.count(n_bad_array), 'n_arrays_nonfinite': IC.count(n_nonfinite), 'n_walks_usable': IC.count(n_ok), 'walk_coverage': round(len(chosen) / max(1, len(index)), 4), 'participant_coverage': round(len(participants_hit) / max(1, len(participants_all)), 4), 'clean_array_share': round(n_ok / max(1, len(chosen)), 4), 'aug_axis_min': int(min(aug)) if aug else 0, 'aug_axis_max': int(max(aug)) if aug else 0, 'frames_npz_minus_v4_min': int(min(diffs)) if diffs else 0, 'frames_npz_minus_v4_median': float(np.median(diffs)) if diffs else 0.0, 'frames_npz_minus_v4_max': int(max(diffs)) if diffs else 0, 'frames_ratio_median': round(float(np.median(ratios)), 4) if ratios else 0.0, 'speed_rank_agreement': (lambda r: None if r is None else round(r, 4))(rank_corr(s_in_a, s_old_a)), 'speed_ratio_median_input_over_frozen': round(float(np.median(s_in_a[pos] / s_old_a[pos])), 4) if pos.any() else None}

def cohort_passes(rep):
    return bool(rep['walk_coverage'] >= MIN_WALK_COVERAGE and rep['participant_coverage'] >= MIN_PARTICIPANT_COVERAGE and (rep['clean_array_share'] >= MIN_CLEAN_ARRAY_SHARE) and (rep['speed_rank_agreement'] is not None) and (rep['speed_rank_agreement'] >= MIN_SPEED_RANK_AGREEMENT))

def stats_report(mean, std):
    mean, std = (np.asarray(mean), np.asarray(std))
    good = mean.shape == (FEAT,) and std.shape == (FEAT,) and (mean.dtype.kind == 'f') and (std.dtype.kind == 'f') and bool(np.isfinite(mean).all()) and bool(np.isfinite(std).all()) and bool((std > 0).all())
    return bool(good)

def checkpoint_report(obj):
    import momask_encoder as M
    import torch
    if not isinstance(obj, dict) or 'net' not in obj or (not isinstance(obj['net'], dict)):
        return {'strict_load_ok': False, 'reason': 'no-net-state-dict'}
    net = obj['net']
    try:
        M.load_strict(net)
        loaded = True
    except Exception as exc:
        return {'strict_load_ok': False, 'reason': type(exc).__name__}
    counters = {k: int(v) for k, v in obj.items() if k != 'net' and isinstance(v, (int, float)) and (not isinstance(v, bool))}
    other = sorted((str(k)[:40] for k in obj if k != 'net' and k not in counters))[:12]
    return {'strict_load_ok': loaded, 'n_tensors': IC.count(len(net)), 'n_parameters': int(sum((int(v.numel()) for v in net.values() if torch.is_tensor(v)))), 'tensor_count_as_expected': len(net) == EXPECTED_STATE_TENSORS, 'scalar_counters': {str(k)[:40]: v for k, v in list(counters.items())[:8]}, 'other_top_level_entries': other}

def decide(reports, stats_ok, ckpt):
    if not ckpt.get('strict_load_ok'):
        return 'C2_PRECHECK_FAIL_CHECKPOINT'
    if not stats_ok:
        return 'C2_PRECHECK_FAIL_NORMALIZATION_FILES'
    if not all((cohort_passes(r) for r in reports.values())):
        return 'C2_PRECHECK_FAIL_OVERLAP'
    return 'C2_PRECHECK_PASS'
