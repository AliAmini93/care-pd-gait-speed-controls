"""Portable analysis kernel. See docs/provenance_protocol.md for scope, provenance and I/O."""
from __future__ import annotations
import hashlib
import json
import os
import platform
import sys
import warnings
from pathlib import Path
import numpy as np
from scipy import stats
import aggregate_core as IC
import root_core as G
import common as C
import p0_core as P
import p1_core as Q
STATUS_COMPLETE = 'H_EXPLORATORY_SENSITIVITY_COMPLETE'
STATUS_INPUT = 'H_INPUT_BLOCKED'
STATUS_PARITY = 'H_PARITY_BLOCKED'
STATUS_PRIVACY = 'H_PRIVACY_BLOCKED'
EXIT_CODE = {STATUS_COMPLETE: 0, STATUS_INPUT: 5, STATUS_PARITY: 6, STATUS_PRIVACY: 7}
MARGIN = Q.SCREEN_MIN
PARITY_TOL = 1e-05
POSE_TOL = 0.0001
UNIT_BAND = (0.8, 1.25)
H_SHARE_MIN = 0.9
H_SHARE_MAX_BELOW = 0.05
RANK_MIN = 0.5
CANONICAL_DIR = None
P1_RESULT = None
P1_RESULT_SHA256 = None
RECEIPT_FILES = ()
DOWNLOAD_PASS = 'G_ROOT_DOWNLOAD_VERIFIED_SAFE_LOAD_PASS'
ROOT_SHA256 = {'3DGait': 'b5349f34ace0c3bd50d3dcf2ceed243ce6b2b7987b4e5caa8a3ccad7d527e550', 'BMCLab': 'ab6484b69efa9797f025449c4cec2ac9f0057f1ba6ad0b2733adddbac5144eae', 'T-SDU-PD': '2cc1b56ea628f50d62e3b25d79313acc9b4cbc99d8acac2e338d9b7e982c2938', 'PD-GaM': '94101283255fc0083b1f8712d9f413d2f5fb0268af9569e1a283c4492247851e'}
ARMS = ('A', 'B', 'C', 'D')
ARM_DEFINITION = {'A': 'S_canonical: speed from canonical v4.0 trans (P1 arm S)', 'B': 'S_root: the same speed function on the publisher original root trans', 'C': 'P_canonical + S_canonical (P1 arm PS)', 'D': "P_canonical + S_root: C's canonical pose columns unchanged, last column replaced by S_root"}
CANONICAL_TWIN = {'A': 'S', 'C': 'PS'}
CHECKS = ('schema', 'label', 'medication', 'other', 'fps', 'frames', 'finite', 'body_pose', 'row_order', 'duration')
CONTRAST_METRICS = ('macro_f1', 'balanced_accuracy', 'ordinal_mae')
LIMITS = ['Exploratory: post hoc sensitivity on four reused cohorts whose P1 contrast was examined repeatedly; not a replication.', 'CIs resample participants within the four observed cohorts, predictions fixed; no unseen-site or population CI.', 'Arm D mixes canonical pose with root speed: a controlled substitution, not a homogeneous root representation.', 'Root and canonical motion are compared record by record only by the separate root-parity command; this command does not identify the canonicalisation step.', 'Labels are the canonical v4.0 labels; the input gate requires equality with the root labels for every joined record; PD-GaM v1.x labels are not tested.', "Speed units are those of each file's trans; equality of the unit convention is checked only by a ratio band.", 'MDS-UPDRS 3.10 rates stride speed among other behaviours, so speed-label agreement is partly criterion overlap.', 'Only the macro-F1 difference-in-differences and delta_root enter the frozen decision; all else is descriptive.', 'Not clinical. No diagnostic, prognostic, causal or deployment claim.']

def rounded(obj):
    if isinstance(obj, float):
        return round(obj, 6)
    if isinstance(obj, dict):
        return {k: rounded(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [rounded(v) for v in obj]
    return obj

def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()

def spearman(a, b):
    a, b = (np.asarray(a, float), np.asarray(b, float))
    if a.size < IC.SMALL_CELL or np.ptp(a) <= 0 or np.ptp(b) <= 0:
        return None
    r = float(stats.spearmanr(a, b).statistic)
    return r if np.isfinite(r) else None

def iter_rows(cohort_objs):
    for ci, name in enumerate(C.COHORTS):
        obj = cohort_objs[name]
        reference = C.walks_of(obj)
        for subject, walks in obj.items():
            for walk, rec in walks.items():
                ref = next(reference, None)
                if ref is None or ref[1] is not rec or ref[0] != subject:
                    raise C.SourceError('ROW_ORDER_ITERATION')
                pose, trans = C.read_arrays(rec)
                if not (np.isfinite(pose).all() and np.isfinite(trans).all()):
                    continue
                yield (ci, subject, walk, rec)
        if next(reference, None) is not None:
            raise C.SourceError('ROW_ORDER_ITERATION')

def key_index(obj):
    if type(obj) is not dict:
        return ({}, 0, 1)
    groups = {}
    for s in obj:
        groups.setdefault(str(s), []).append(s)
    out, coll, bad = ({}, 0, 0)
    for s, walks in obj.items():
        if type(s) not in (str, int) or type(walks) is not dict:
            bad += 1
            continue
        if len(groups[str(s)]) > 1:
            coll += len(walks)
            continue
        for w, rec in walks.items():
            if type(w) is str and type(rec) is dict:
                out[str(s), w] = rec
            else:
                bad += 1
    return (out, coll, bad)

def row_checks(rec_c, rec_r, table, i, ci, pose_tol=POSE_TOL):
    res = dict.fromkeys(CHECKS, False)
    pose_c, trans_c = C.read_arrays(rec_c)
    fps_c = C.read_fps(rec_c['fps'])
    lab_c = C.read_label(rec_c['UPDRS_GAIT'])
    s_c, d_c = P.speed_and_duration(trans_c, fps_c)
    res['row_order'] = bool(i < table['label'].size and int(table['cohort'][i]) == ci and (int(table['label'][i]) == (-1 if lab_c is None else lab_c)) and (table['speed'][i] == s_c) and (table['duration'][i] == d_c))
    try:
        if type(rec_r) is not dict or not C.WALK_KEYS.issubset(rec_r):
            raise C.SourceError('ROOT_SCHEMA')
        pose_r, trans_r = C.read_arrays(rec_r)
    except C.SourceError:
        return (res, None, None, None)
    res['schema'] = True
    try:
        res['label'] = C.read_label(rec_r['UPDRS_GAIT']) == lab_c
    except C.SourceError:
        res['label'] = False
    for f in ('medication', 'other'):
        res[f] = G.compare_field(G.field_value(rec_r, f), G.field_value(rec_c, f)) == 'EQUAL'
    try:
        res['fps'] = C.read_fps(rec_r['fps']) == fps_c
    except C.SourceError:
        res['fps'] = False
    res['frames'] = pose_r.shape[0] == pose_c.shape[0]
    res['finite'] = bool(np.isfinite(pose_r).all() and np.isfinite(trans_r).all())
    if res['frames'] and res['finite']:
        res['body_pose'] = bool(np.abs(pose_r[:, 3:] - pose_c[:, 3:]).max() <= pose_tol)
    s_r = h_r = None
    if res['finite']:
        s_r, d_r = P.speed_and_duration(trans_r, fps_c)
        res['duration'] = bool(res['frames'] and d_r == d_c)
        h_r = horizontal_share(trans_r, fps_c)
    return (res, s_r, h_r, horizontal_share(trans_c, fps_c))

def root_speed_stops(speed):
    if speed.size == 0 or not np.isfinite(speed).all() or (speed <= 0).any():
        return 'FAIL_NON_FINITE_OR_NON_POSITIVE'
    if (speed <= C.LOW_SPEED).mean() > C.MAX_LOW_SPEED_SHARE:
        return 'FAIL_LOW_SPEED_SHARE'
    if P.iqr(speed) <= 0:
        return 'FAIL_ZERO_IQR'
    return 'PASS'

def smoothed_lengths(trans, fps):
    w = max(1, round(fps / 2))
    xyz = np.asarray(trans, dtype=float)
    if w > 1:
        left = (w - 1) // 2
        padded = np.pad(xyz, ((left, w - 1 - left), (0, 0)), mode='edge')
        kernel = np.ones(w) / w
        xyz = np.column_stack([np.convolve(padded[:, j], kernel, mode='valid') for j in range(3)])
    d = np.diff(xyz, axis=0)
    return (float(np.linalg.norm(d[:, [0, 2]], axis=1).sum()), float(np.linalg.norm(d, axis=1).sum()))

def horizontal_share(trans, fps):
    xz, xyz = smoothed_lengths(trans, fps)
    return xz / xyz if xyz > 0 else None

def coordinate_verdict(shares, h_min=H_SHARE_MIN, max_below=H_SHARE_MAX_BELOW):
    v = np.asarray([x for x in shares if x is not None], float)
    if v.size < IC.SMALL_CELL or v.size != len(shares):
        return 'FAIL_UNDEFINED'
    if np.median(v) < h_min:
        return 'FAIL_MEDIAN_BELOW_MIN'
    if (v < h_min).mean() > max_below:
        return 'FAIL_TOO_MANY_WALKS_BELOW_MIN'
    return 'PASS'

def join_root(table, canon_objs, root_objs, pose_tol=POSE_TOL, unit_band=UNIT_BAND):
    n = int(table['label'].size)
    speed_root = np.full(n, np.nan)
    idx = {}
    raw = {}
    for name in C.COHORTS:
        R, coll_r, bad_r = key_index(root_objs.get(name))
        K, coll_c, bad_c = key_index(canon_objs[name])
        idx[name] = R
        raw[name] = {'records': {'joined': len(set(R) & set(K)), 'root_only': len(set(R) - set(K)), 'canonical_only': len(set(K) - set(R)), 'collision_or_malformed': coll_r + coll_c + bad_r + bad_c}, 'checks': {c: {'equal': 0, 'differs': 0} for c in CHECKS}, 'h_root': [], 'h_canonical': []}
    seen = 0
    for i, (ci, subject, walk, rec_c) in enumerate(iter_rows(canon_objs)):
        seen = i + 1
        name = C.COHORTS[ci]
        rec_r = idx[name].get((str(subject), walk))
        if rec_r is None:
            continue
        res, s_r, h_r, h_c = row_checks(rec_c, rec_r, table, i, ci, pose_tol)
        for c in CHECKS:
            raw[name]['checks'][c]['equal' if res[c] else 'differs'] += 1
        raw[name]['h_root'].append(h_r)
        raw[name]['h_canonical'].append(h_c)
        if s_r is not None and i < n:
            speed_root[i] = s_r
    failure = None
    if seen != n:
        failure = 'ROW_COUNT'
    for name in C.COHORTS:
        rc = raw[name]['records']
        if failure is None and (rc['root_only'] or rc['canonical_only'] or rc['collision_or_malformed']):
            failure = f'RECORD_SETS:{name}'
    for c in CHECKS:
        for name in C.COHORTS:
            if failure is None and raw[name]['checks'][c]['differs']:
                failure = f'{c.upper()}:{name}'
    for ci, name in enumerate(C.COHORTS):
        m = table['cohort'] == ci
        s_r, s_c = (speed_root[m], table['speed'][m])
        ok = np.isfinite(s_r)
        raw[name]['root_speed_stops'] = root_speed_stops(s_r)
        h_root, h_can = (raw[name].pop('h_root'), raw[name].pop('h_canonical'))
        below = sum((1 for x in h_root if x is None or x < H_SHARE_MIN))
        raw[name]['coordinate'] = {'verdict_root': coordinate_verdict(h_root), 'min_share': H_SHARE_MIN, 'max_share_of_walks_below': H_SHARE_MAX_BELOW, 'root_horizontal_share': G.quantiles(h_root), 'canonical_horizontal_share': G.quantiles(h_can), 'canonical_verdict_descriptive': coordinate_verdict(h_can), 'root_walks': G.render_partition({'at_least_min': len(h_root) - below, 'below_min_or_undefined': below}, G.has_small({'a': len(h_root) - below, 'b': below}))}
        keep = ok & (s_c > C.LOW_SPEED)
        ratio = s_r[keep] / s_c[keep]
        med = float(np.median(ratio)) if ratio.size >= IC.SMALL_CELL else None
        band = 'PASS' if med is not None and unit_band[0] <= med <= unit_band[1] else 'FAIL'
        raw[name]['unit'] = {'median_ratio_root_over_canonical': med, 'band': list(unit_band), 'verdict': band, 'ratio_quartiles': G.quantiles(ratio.tolist())}
        el = m & np.isin(table['label'], (0, 1, 2)) & np.isfinite(speed_root)
        raw[name]['speed_agreement'] = {'spearman_root_vs_canonical_all_walks': spearman(s_r[ok], s_c[ok]), 'median_speed_canonical': float(np.median(s_c)) if s_c.size >= IC.SMALL_CELL else None, 'median_speed_root': float(np.median(s_r[ok])) if ok.sum() >= IC.SMALL_CELL else None, 'spearman_grade_vs_canonical_speed_eligible': spearman(table['speed'][el], table['label'][el]), 'spearman_grade_vs_root_speed_eligible': spearman(speed_root[el], table['label'][el])}
        if failure is None and raw[name]['root_speed_stops'] != 'PASS':
            failure = f'ROOT_SPEED_STOPS:{name}'
        if failure is None and raw[name]['coordinate']['verdict_root'] != 'PASS':
            failure = f'COORDINATE_CONVENTION:{name}'
        if failure is None and band != 'PASS':
            failure = f'UNIT_BAND:{name}'
        rank = raw[name]['speed_agreement']['spearman_root_vs_canonical_all_walks']
        raw[name]['speed_agreement']['rank_gate'] = 'PASS' if rank is not None and rank >= RANK_MIN else 'FAIL'
        if failure is None and raw[name]['speed_agreement']['rank_gate'] != 'PASS':
            failure = f'RANK_AGREEMENT:{name}'
    if failure is None and (not np.isfinite(speed_root).all()):
        failure = 'ROOT_SPEED_INCOMPLETE'
    return (speed_root, raw, failure)

def render_join(raw):
    out = {}
    for name in C.COHORTS:
        r = raw.get(name)
        if r is None:
            continue
        parts = [r['records']] + [r['checks'][c] for c in CHECKS]
        coarse = G.has_small(*parts)
        out[name] = {'mode': 'coarse' if coarse else 'exact', 'records': G.render_partition(r['records'], coarse), 'checks': {c: G.render_partition(r['checks'][c], coarse) for c in CHECKS}}
        for k in ('root_speed_stops', 'coordinate', 'unit', 'speed_agreement'):
            if k in r:
                out[name][k] = r[k]
    return out

def arm_inputs(table, speed_root):
    s_root = np.asarray(speed_root, dtype=float)
    return {'A': table['speed'][:, None], 'B': s_root[:, None], 'C': np.column_stack([table['pose'], table['speed']]), 'D': np.column_stack([table['pose'], s_root])}

def substitution_problems(X, table, speed_root):
    feats = Q.arm_features(table)
    probs = []
    if not np.array_equal(X['A'], feats['S']):
        probs.append('A_NOT_P1_S')
    if not np.array_equal(X['C'], feats['PS']):
        probs.append('C_NOT_P1_PS')
    if X['B'].shape != X['A'].shape or X['D'].shape != X['C'].shape:
        probs.append('SHAPE')
        return probs
    if not np.array_equal(X['D'][:, :-1], X['C'][:, :-1]):
        probs.append('D_POSE_COLUMNS_DIFFER_FROM_C')
    if not (np.array_equal(X['B'][:, 0], speed_root) and np.array_equal(X['D'][:, -1], speed_root)):
        probs.append('ROOT_COLUMN_NOT_S_ROOT')
    if not (np.isfinite(X['B']).all() and np.isfinite(X['D']).all()):
        probs.append('NON_FINITE')
    return probs

def fold_problems(folds, table):
    eligible = np.isin(table['label'], (0, 1, 2))
    probs = []
    for arm, f in folds.items():
        if len(f) != len(C.COHORTS):
            probs.append(f'FOLD_COUNT:{arm}')
            continue
        for ci in range(len(C.COHORTS)):
            if not np.array_equal(f[ci]['test_mask'], eligible & (table['cohort'] == ci)):
                probs.append(f'TEST_MASK:{arm}:{ci}')
    return probs

def predictions(folds, index_from=None):
    out = []
    for c, f in enumerate(folds):
        k = f['selected']['all'] if index_from is None else index_from[c]['selected']['all']
        out.append(f['proba'][k].argmax(axis=1))
    return out

def replay(table, folds, scorer, configs):
    names = scorer.names
    members = Q.policy_members(configs)
    conf = scorer.conf(predictions(folds))
    fixed, stability = ({}, {})
    for i, cfg in enumerate(configs):
        scores = [Q.walk_macro_f1(scorer.y[nm], Q.proba_for(folds, c, config_index=i).argmax(axis=1)) for c, nm in enumerate(names)]
        fixed[cfg[0]] = {'per_cohort': dict(zip(names, scores)), 'M': float(np.mean(scores))}
    for c, nm in enumerate(names):
        f = folds[c]
        inner = f['inner'].mean(axis=1)
        outer = np.array([Q.walk_macro_f1(scorer.y[nm], p.argmax(axis=1)) for p in f['proba']])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', stats.ConstantInputWarning)
            rho = stats.spearmanr(inner, outer).statistic
        k = f['selected']['all']
        stability[nm] = {'selected': configs[k][0], 'selected_inner_mean': float(inner[k]), 'selected_outer': float(outer[k]), 'inner_mean_by_config': {cfg[0]: float(v) for cfg, v in zip(configs, inner)}, 'outer_by_config': {cfg[0]: float(v) for cfg, v in zip(configs, outer)}, 'spearman_inner_outer': None if not np.isfinite(rho) else float(rho)}
    return {'metrics': scorer.all_metrics(conf), 'fixed': fixed, 'stability': stability, 'conf': conf, 'selected': {nm: {p: configs[folds[c]['selected'][p]][0] for p in members} for c, nm in enumerate(names)}, 'confusions': {nm: conf[nm].sum(axis=0).astype(int).tolist() for nm in names}}

def compare_tree(frozen, got, path, tol, leaves, failures):
    if isinstance(frozen, dict):
        if not isinstance(got, dict) or set(frozen) != set(got):
            failures.append((path, None))
            return
        for k in sorted(frozen):
            compare_tree(frozen[k], got[k], f'{path}/{k}', tol, leaves, failures)
    elif isinstance(frozen, list):
        if not isinstance(got, (list, tuple)) or len(frozen) != len(got):
            failures.append((path, None))
            return
        for j, (a, b) in enumerate(zip(frozen, got)):
            compare_tree(a, b, f'{path}[{j}]', tol, leaves, failures)
    elif isinstance(frozen, float):
        ok_type = isinstance(got, (float, int, np.floating, np.integer)) and (not isinstance(got, (bool, np.bool_)))
        dev = abs(round(float(got), 6) - frozen) if ok_type and np.isfinite(float(got)) else None
        leaves.append(dev)
        if dev is None or dev > tol:
            failures.append((path, dev))
    else:
        leaves.append(0.0)
        same = (type(frozen) is type(got) or (type(frozen) is int and isinstance(got, (int, np.integer)) and (not isinstance(got, (bool, np.bool_))))) and frozen == got
        if not same:
            failures.append((path, None))

def baseline_parity(table, folds, scorer, configs, frozen, tol=PARITY_TOL):
    rep = {a: replay(table, folds[a], scorer, configs) for a in ('A', 'C')}
    eligible = np.isin(table['label'], (0, 1, 2))
    pairs = [('totals', frozen.get('totals'), {'walks_all_labels': int((table['label'] >= 0).sum()), 'walks_eligible_012': int(eligible.sum()), 'subjects_eligible': int(len(np.unique(table['subject'][eligible])))}), ('candidate_configs', frozen.get('candidate_configs'), [{'name': c[0], 'family': c[1], 'params': c[2]} for c in configs]), ('primary/contrast', (frozen.get('primary') or {}).get('contrast'), scorer.contrast(rep['C']['conf'], rep['A']['conf']))]
    for arm, twin in CANONICAL_TWIN.items():
        get = lambda *keys: _dig(frozen, keys)
        pairs += [(f'arms_inductive_strong/{twin}', get('arms_inductive_strong', twin), rep[arm]['metrics']), (f'per_config_macro_f1_exploratory/{twin}', get('per_config_macro_f1_exploratory', twin), rep[arm]['fixed']), (f'selected_configs/{twin}', get('selected_configs', twin), rep[arm]['selected']), (f'selection_stability/{twin}', get('selection_stability', twin), rep[arm]['stability']), (f'confusions_by_cohort/{twin}_strong', get('confusions_by_cohort', f'{twin}_strong'), rep[arm]['confusions'])]
    leaves, failures = ([], [])
    for path, want, got in pairs:
        if want is None:
            failures.append((path + ' (absent in frozen summary)', None))
            continue
        compare_tree(want, rounded(_plain(got)), path, tol, leaves, failures)
    devs = [d for d in leaves if d is not None]
    screen = Q.decide(*_screen_args(scorer, rep['C']['conf'], rep['A']['conf']))
    report = {'verdict': 'PASS' if not failures else 'FAIL', 'tolerance': tol, 'compared_values': len(leaves), 'failed_values': len(failures), 'max_abs_deviation': max(devs) if devs else None, 'failing_paths': [p[:200] for p, _ in failures[:12]], 'sections': [p for p, _, _ in pairs], 'replayed_p1_screen_rule_without_null': {k: screen[k] for k in ('verdict', 'qualifier', 'sites_nonnegative')}}
    return (report, not failures)

def _dig(obj, keys):
    for k in keys:
        if not isinstance(obj, dict) or k not in obj:
            return None
        obj = obj[k]
    return obj

def _plain(obj):
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj

def _screen_args(scorer, conf_pose, conf_speed):
    block = scorer.contrast(conf_pose, conf_speed)
    m = block['M']
    per_site = [block['per_cohort'][n]['point'] for n in scorer.names]
    return (m['point'], m['ci95'][0], m['ci95'][1], per_site, 1.0, 0, 0.0, 0.0)

def classify(did, did_lo, did_hi, d_root, d_root_lo, d_root_hi, margin=MARGIN):
    if (did_lo > 0 or did_hi < 0) and abs(did) >= margin:
        return ('CLAIM_SENSITIVE_TO_ROOT_SPEED', 'DID_CI_EXCLUDES_ZERO_AND_ABS_POINT_AT_LEAST_MARGIN')
    if d_root_lo > 0:
        return ('CLAIM_SENSITIVE_TO_ROOT_SPEED', 'DELTA_ROOT_CI_ABOVE_ZERO_POSE_ADDS_BEYOND_ROOT_SPEED')
    if did_lo > -margin and did_hi < margin and (d_root_hi < 0):
        return ('CLAIM_ROBUST_TO_ROOT_SPEED', 'DID_CI_INSIDE_MARGIN_AND_DELTA_ROOT_CI_BELOW_ZERO')
    return ('INDETERMINATE', 'NEITHER_RULE_MET')

def contrasts(scorer, conf, metric):
    st = {a: scorer.stats(conf[a], metric) for a in ARMS}
    p, d = ({a: st[a][0] for a in ARMS}, {a: st[a][1] for a in ARMS})
    return {'delta_canonical_C_minus_A': scorer.block(p['C'] - p['A'], d['C'] - d['A']), 'delta_root_D_minus_B': scorer.block(p['D'] - p['B'], d['D'] - d['B']), 'difference_in_differences': scorer.block(p['D'] - p['B'] - (p['C'] - p['A']), d['D'] - d['B'] - (d['C'] - d['A'])), 'B_minus_A': scorer.block(p['B'] - p['A'], d['B'] - d['A']), 'D_minus_C': scorer.block(p['D'] - p['C'], d['D'] - d['C'])}

def policy_result(scorer, conf, arms_to_report):
    cons = {m: contrasts(scorer, conf, m) for m in CONTRAST_METRICS}
    mf = cons['macro_f1']
    did, dr = (mf['difference_in_differences']['M'], mf['delta_root_D_minus_B']['M'])
    verdict, rule = classify(did['point'], did['ci95'][0], did['ci95'][1], dr['point'], dr['ci95'][0], dr['ci95'][1])
    screen_c = Q.decide(*_screen_args(scorer, conf['C'], conf['A']))
    screen_r = Q.decide(*_screen_args(scorer, conf['D'], conf['B']))
    per = {n: mf['difference_in_differences']['per_cohort'][n]['point'] for n in scorer.names}
    return {'arms': {a: scorer.all_metrics(conf[a]) for a in arms_to_report}, 'contrasts': cons, 'decision': {'verdict': verdict, 'rule_met': rule, 'margin': MARGIN, 'difference_in_differences_macro_f1_M': did, 'delta_root_macro_f1_M': dr, 'delta_canonical_macro_f1_M': mf['delta_canonical_C_minus_A']['M'], 'cohorts_with_abs_did_point_at_least_margin': int(sum((abs(v) >= MARGIN for v in per.values()))), 'cohorts_with_did_point_positive': int(sum((v > 0 for v in per.values())))}, 'p1_screen_rule_without_null': {'canonical_C_vs_A': {k: screen_c[k] for k in ('verdict', 'qualifier', 'sites_nonnegative')}, 'root_D_vs_B': {k: screen_r[k] for k in ('verdict', 'qualifier', 'sites_nonnegative')}}}

def selection_summary(folds, configs, names):
    out = {}
    for a in ARMS:
        by, counts = ({}, {})
        for c, nm in enumerate(names):
            f = folds[a][c]
            k = f['selected']['all']
            inner = f['inner'].mean(axis=1)
            by[nm] = {'selected': configs[k][0], 'selected_inner_source_mean': float(inner[k]), 'inner_source_mean_by_config': {cfg[0]: float(v) for cfg, v in zip(configs, inner)}, 'selected_by_family_policy': {p: configs[f['selected'][p]][0] for p in ('LR', 'RF', 'HGB')}}
            counts[configs[k][0]] = counts.get(configs[k][0], 0) + 1
        out[a] = {'by_held_out_cohort': by, 'selection_counts_over_four_folds': counts}
    for root, canon in (('B', 'A'), ('D', 'C')):
        same = sum((folds[root][c]['selected']['all'] == folds[canon][c]['selected']['all'] for c in range(len(names))))
        out[root]['folds_selecting_same_config_as_' + canon] = int(same)
    return out

def per_config_table(folds, scorer, configs):
    out = {}
    for a in ARMS:
        out[a] = {}
        for i, cfg in enumerate(configs):
            scores = [Q.walk_macro_f1(scorer.y[nm], Q.proba_for(folds[a], c, config_index=i).argmax(axis=1)) for c, nm in enumerate(scorer.names)]
            out[a][cfg[0]] = {'per_cohort': dict(zip(scorer.names, scores)), 'M': float(np.mean(scores))}
    return out

def refit_sensitivity(table, X, folds, scorer, configs, seeds=Q.REFIT_SEEDS):
    eligible = np.isin(table['label'], (0, 1, 2))
    warned = [0]
    by = {a: [] for a in ARMS}
    for seed in seeds:
        for a in ARMS:
            scores = []
            for c, nm in enumerate(scorer.names):
                cfg = configs[folds[a][c]['selected']['all']]
                if cfg[1] == 'RF':
                    tr = eligible & (table['cohort'] != c)
                    w = Q.train_weights(table['label'][tr], table['subject'][tr], 'walk')
                    proba = Q.fit_predict_proba(cfg, X[a][tr], table['label'][tr], w, X[a][scorer.masks[nm]], warned, seed)
                else:
                    proba = Q.proba_for(folds[a], c, policy='all')
                scores.append(Q.walk_macro_f1(scorer.y[nm], proba.argmax(axis=1)))
            by[a].append(float(np.mean(scores)))
    did = [by['D'][k] - by['B'][k] - (by['C'][k] - by['A'][k]) for k in range(len(seeds))]
    d_root = [by['D'][k] - by['B'][k] for k in range(len(seeds))]
    return {'seeds': list(seeds), 'M_macro_f1_by_seed': by, 'did_by_seed': did, 'did_range': [min(did), max(did)], 'delta_root_by_seed': d_root, 'delta_root_range': [min(d_root), max(d_root)], 'convergence_warning_count': warned[0]}

def frozen_p1(path=None, want=None):
    """Portable replacement: the user's own completed P1 aggregate, pinned to the SHA-256 the CLI recorded."""
    path = P1_RESULT if path is None else path
    want = P1_RESULT_SHA256 if want is None else want
    if path is None or want is None:
        raise C.SourceError('P1_RESULT_NOT_CONFIGURED')
    raw = Path(path).read_bytes()
    if sha256_bytes(raw) != want:
        raise C.SourceError('FROZEN_P1_SUMMARY_PIN')
    return json.loads(raw.decode('utf-8'))

def receipt_rows():
    """Portable replacement: verified rows of the user's own passed fetch-originals aggregates, checked against pins."""
    rows = {}
    for path in RECEIPT_FILES:
        path = Path(path)
        if not path.is_file():
            raise C.SourceError('DOWNLOAD_RECEIPT_MISSING')
        try:
            agg = json.loads(path.read_bytes().decode('utf-8'))
        except (ValueError, UnicodeError):
            raise C.SourceError('DOWNLOAD_RECEIPT_INVALID') from None
        if not isinstance(agg, dict) or agg.get('status') != DOWNLOAD_PASS:
            raise C.SourceError('DOWNLOAD_RECEIPT_NOT_PASS')
        for r in agg.get('files', []):
            if isinstance(r, dict) and r.get('integrity_ok') and (r.get('cohort') in C.COHORTS):
                rows[r['cohort']] = r
    for name in C.COHORTS:
        r = rows.get(name)
        if r is None or r.get('publisher_datafile') != G.ROOT_PINS[name][1] or r.get('md5_pinned') != G.ROOT_PINS[name][3] or (r.get('sha256') != ROOT_SHA256[name]):
            raise C.SourceError(f'DOWNLOAD_RECEIPT_ROW:{name}')
    return rows

def load_roots():
    out = {}
    for name in C.COHORTS:
        path = G.root_path(name)
        if not path.is_file():
            raise C.SourceError(f'ROOT_FILE_MISSING:{name}')
        data = path.read_bytes()
        _, _, size, md5 = G.ROOT_PINS[name][:4]
        if len(data) != size or hashlib.md5(data).hexdigest() != md5 or sha256_bytes(data) != ROOT_SHA256[name]:
            raise C.SourceError(f'ROOT_FILE_PIN:{name}')
        out[name] = G.strict_loads(data)
        del data
    return out

def load_canonical():
    """Portable replacement: the four pinned canonical files from the folder the CLI set (--data-dir)."""
    if CANONICAL_DIR is None:
        raise C.SourceError('CANONICAL_DIR_NOT_CONFIGURED')
    return {c: C.load_cohort(c, CANONICAL_DIR) for c in C.COHORTS}

def input_provenance():
    """Portable replacement: input pins plus SHA-256 digests of the user's own P1 and download aggregates."""
    return {'canonical_v4': {c: {'file': f[0], 'bytes': f[1], 'sha256': f[2], 'md5': f[3]} for c, f in C.FILES.items()}, 'canonical_hf_commit': C.HF_COMMIT, 'root_original': {c: {'file': G.ROOT_PINS[c][0], 'publisher_datafile': G.ROOT_PINS[c][1], 'bytes': G.ROOT_PINS[c][2], 'md5': G.ROOT_PINS[c][3], 'sha256': ROOT_SHA256[c], 'versions': G.ROOT_PINS[c][4]} for c in C.COHORTS}, 'root_doi': G.DOI, 'frozen_p1_summary': {'path': 'local P1 aggregate ' + (Path(P1_RESULT).name if P1_RESULT else 'UNSET'), 'sha256': P1_RESULT_SHA256 or 'UNSET'}, 'download_receipts_sha256': {Path(p).name: sha256_bytes(Path(p).read_bytes()) if Path(p).is_file() else 'ABSENT' for p in RECEIPT_FILES}}

def base(meta, configs=Q.CONFIGS, n_boot=Q.N_BOOT):
    import scipy
    import sklearn
    out = dict(meta)
    out.update({'status': None, 'failed_gate': None, 'stop_reason': None, 'scored_root_arms': False, 'exploratory_due_to_four_reused_sites': True, 'inference': 'exploratory sensitivity; CIs conditional on the four reused cohorts and their observed participants; no site-level population CI and no causal claim', 'design': {'arms': ARM_DEFINITION, 'primary_estimand': 'difference_in_differences = (M(D)-M(B)) - (M(C)-M(A)), macro-F1, equal-cohort mean, shared participant-cluster bootstrap draws', 'policies': {'primary_source_only': 'each arm selects its own config (P1 policy all) on source cohorts only', 'secondary_locked': 'B and D refit the config index A and C selected in the same outer fold'}, 'decision_margin': MARGIN, 'parity_tolerance': PARITY_TOL, 'pose_tolerance': POSE_TOL, 'unit_band': list(UNIT_BAND), 'horizontal_share_min': H_SHARE_MIN, 'horizontal_share_max_below': H_SHARE_MAX_BELOW, 'rank_agreement_min': RANK_MIN, 'n_boot': int(n_boot), 'bootstrap_seed': Q.SEED, 'candidates': [c[0] for c in configs], 'training_weights': 'walk (class-balanced), as P1 primary', 'ordinal_mae_note': 'lower is better, so its contrasts have the opposite sign convention'}, 'provenance': dict(input_provenance(), python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__, omp_num_threads=os.environ.get('OMP_NUM_THREADS', 'UNSET'), cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES', 'UNSET')), 'limits': LIMITS})
    return out

def blocked(out, status, gate, reason):
    out.update({'status': status, 'failed_gate': gate, 'stop_reason': G.sanitize_reason(reason, 120), 'scored_root_arms': False})
    return out

def run(meta, canonical_loader=load_canonical, root_loader=load_roots, frozen_loader=frozen_p1, receipts=receipt_rows, configs=Q.CONFIGS, n_boot=Q.N_BOOT, check_stops=Q.check_stops, fit=Q.outer_run, refit_seeds=Q.REFIT_SEEDS, progress=print):
    import scipy
    import sklearn
    out = base(meta, configs, n_boot)
    try:
        Q.check_versions(np.__version__, scipy.__version__, sklearn.__version__)
        frozen = frozen_loader()
    except C.SourceError as e:
        return blocked(out, STATUS_PARITY, 'BASELINE_INPUTS', e)
    try:
        receipts()
        canon = canonical_loader()
        table = Q.build_table(canon)
    except C.SourceError as e:
        return blocked(out, STATUS_INPUT, 'CANONICAL_INPUTS', e)
    try:
        check_stops(table)
    except C.SourceError as e:
        return blocked(out, STATUS_PARITY, 'P1_PRE_SCORE_CHECKS', e)
    progress('PROGRESS canonical table built and P1 pre-score checks passed')
    try:
        roots = root_loader()
        speed_root, raw, failure = join_root(table, canon, roots)
    except C.SourceError as e:
        out['input_gate'] = {'verdict': 'FAIL'}
        return blocked(out, STATUS_INPUT, 'ROOT_INPUTS', e)
    out['input_gate'] = {'verdict': 'PASS' if failure is None else 'FAIL', 'cohorts': render_join(raw)}
    del canon, roots
    if failure is not None:
        return blocked(out, STATUS_INPUT, 'INPUT_JOIN', failure)
    X = arm_inputs(table, speed_root)
    probs = substitution_problems(X, table, speed_root)
    out['substitution_gate'] = {'verdict': 'PASS' if not probs else 'FAIL', 'problems': probs, 'pose_columns_held': int(X['C'].shape[1] - 1), 'changed_columns': 1}
    if probs:
        return blocked(out, STATUS_INPUT, 'SUBSTITUTION', probs[0])
    progress(f"PROGRESS input and substitution gates {out['input_gate']['verdict']}")
    warned = [0]
    folds = {a: fit(X[a], table['label'], table['cohort'], table['subject'], 'walk', configs, warned) for a in ('A', 'C')}
    scorer = Q.Scorer(table, n_boot)
    report, ok = baseline_parity(table, folds, scorer, configs, frozen)
    out['baseline_parity'] = report
    progress(f"PROGRESS baseline parity {report['verdict']}")
    if not ok:
        return blocked(out, STATUS_PARITY, 'BASELINE_PARITY', 'BASELINE_PARITY_FAIL')
    for a in ('B', 'D'):
        folds[a] = fit(X[a], table['label'], table['cohort'], table['subject'], 'walk', configs, warned)
    fprobs = fold_problems(folds, table)
    if fprobs:
        return blocked(out, STATUS_INPUT, 'FOLD_MEMBERSHIP', fprobs[0])
    out['scored_root_arms'] = True
    progress('PROGRESS root arms fitted')
    conf_primary = {a: scorer.conf(predictions(folds[a])) for a in ARMS}
    conf_locked = {'A': conf_primary['A'], 'C': conf_primary['C'], 'B': scorer.conf(predictions(folds['B'], index_from=folds['A'])), 'D': scorer.conf(predictions(folds['D'], index_from=folds['C']))}
    eligible = np.isin(table['label'], (0, 1, 2))
    out['totals'] = {'walks_all_labels': IC.count((table['label'] >= 0).sum()), 'walks_eligible_012': IC.count(eligible.sum()), 'subjects_eligible': IC.count(len(np.unique(table['subject'][eligible]))), 'by_cohort': {n: {'walks_eligible': IC.count((eligible & (table['cohort'] == c)).sum()), 'subjects_eligible': IC.count(len(np.unique(table['subject'][eligible & (table['cohort'] == c)])))} for c, n in enumerate(scorer.names)}}
    out['primary_source_only'] = dict(policy_result(scorer, conf_primary, ARMS), selection=selection_summary(folds, configs, scorer.names))
    out['secondary_locked_policy'] = policy_result(scorer, conf_locked, ('B', 'D'))
    out['secondary_locked_policy']['locked_configs'] = {'B': {n: configs[folds['A'][c]['selected']['all']][0] for c, n in enumerate(scorer.names)}, 'D': {n: configs[folds['C'][c]['selected']['all']][0] for c, n in enumerate(scorer.names)}}
    out['per_config_outer_macro_f1_exploratory'] = per_config_table(folds, scorer, configs)
    out['refit_seed_sensitivity'] = refit_sensitivity(table, X, folds, scorer, configs, refit_seeds)
    out['convergence_warning_count'] = warned[0]
    out['status'] = STATUS_COMPLETE
    return out

def finalize(out):
    out = rounded(_plain(out))
    problems = IC.problems_of(out) + G.disclosure_problems(out)
    if problems:
        keep = {k: out[k] for k in ('card', 'card_sha256', 'run_commit', 'protocol_class', 'seed', 'compute') if k in out}
        keep.update({'status': STATUS_PRIVACY, 'failed_gate': 'AGGREGATE_DISCLOSURE', 'scored_root_arms': out.get('scored_root_arms'), 'exploratory_due_to_four_reused_sites': True, 'problem_paths': sorted({p[:200] for p in problems})[:20]})
        return keep
    return out
