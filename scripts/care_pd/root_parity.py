"""Portable analysis kernel. See docs/provenance_protocol.md for scope, provenance and I/O."""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path
import numpy as np
import aggregate_core as IC
import root_core as G
import common as C
import encoder_inputs as F
import encoder_precheck as K
import humanml_root as E2
DOWNLOAD_PASS = 'G_ROOT_DOWNLOAD_VERIFIED_SAFE_LOAD_PASS'
OFFICIAL_FPS = {'3DGait': 30, 'BMCLab': 150, 'T-SDU-PD': 30, 'PD-GaM': 25}
OFFICIAL_RATE = {c: max(1, int(f / 30)) for c, f in OFFICIAL_FPS.items()}
VERTICAL_DISCARDED = ('T-SDU-PD',)
CROSS_ERA_HUMANML = ('PD-GaM',)
PROFILE_R = 0.99
RATIO_DEV = 0.01
RESID = 0.01
MATCH_SHARE = 0.95
NO_MATCH_SHARE = 0.5
MIN_STEPS = 12
MAX_LAG = 5
POSE_TOL = 0.0001
MAX_OFFSET = 30
STRIDES = (2, 3, 4, 5, 6)
ORIENT_TOL_DEG = 1.0
SCALE_UNITY_TOL = 0.01
C3X_MIN_FRAMES = 8
E2_PARITY_TOL = 0.0001
MECHANISMS = ('TIME_SHIFT', 'ORIENTATION_COUPLED_TRANSLATION', 'ORIENTATION_COUPLED_PLANAR_ONLY', 'UNIDENTIFIED')
IDENTIFIED = MECHANISMS[:3]
LABEL_CELLS = ('EQUAL_GRADE', 'EQUAL_ABSENT', 'ABSENT_ENCODING_DIFFERS', 'CHANGED_GRADE', 'GRADED_ROOT_ONLY', 'GRADED_CANONICAL_ONLY', 'INVALID_VALUE', 'ROOT_RECORD_ONLY_GRADED', 'ROOT_RECORD_ONLY_UNGRADED', 'ROOT_RECORD_ONLY_INVALID', 'CANONICAL_RECORD_ONLY_GRADED', 'CANONICAL_RECORD_ONLY_UNGRADED', 'CANONICAL_RECORD_ONLY_INVALID', 'IDENTITY_COLLISION_RECORDS')
MISMATCH_CELLS = ('CHANGED_GRADE', 'GRADED_ROOT_ONLY', 'GRADED_CANONICAL_ONLY')
UNRESOLVED_CELLS = ('INVALID_VALUE', 'ROOT_RECORD_ONLY_GRADED', 'ROOT_RECORD_ONLY_INVALID', 'CANONICAL_RECORD_ONLY_GRADED', 'CANONICAL_RECORD_ONLY_INVALID', 'IDENTITY_COLLISION_RECORDS')
FIELD_CELLS = ('EQUAL', 'ABSENT_ENCODING_DIFFERS', 'ONE_SIDE_ABSENT', 'DIFFERS', 'UNCOMPARABLE')
KIND_CELLS = ('GRADE', 'ABSENT_NONE', 'ABSENT_NAN', 'ABSENT_KEY', 'INVALID')

def pearson(a, b):
    a, b = (np.asarray(a, float), np.asarray(b, float))
    if a.size < MIN_STEPS or a.size != b.size or np.ptp(a) <= 0 or (np.ptp(b) <= 0):
        return None
    return float(np.corrcoef(a, b)[0, 1])

def step_norms(x):
    return np.linalg.norm(np.diff(np.asarray(x, float), axis=0), axis=1)

def ratio_stats(num, den):
    num, den = (np.asarray(num, float), np.asarray(den, float))
    if den.size == 0 or den.size != num.size:
        return (None, None)
    m = np.isfinite(num) & np.isfinite(den) & (den > 1e-09) & (num > 1e-12) & (den >= 0.1 * np.median(den))
    if m.sum() < MIN_STEPS:
        return (None, None)
    r = num[m] / den[m]
    med = float(np.median(r))
    return (med, float(np.median(np.abs(np.log(r / med)))))

def step_match(ref, other):
    r = pearson(ref, other)
    med, dev = ratio_stats(other, ref)
    return {'r': r, 'scale': med, 'dev': dev, 'match': bool(r is not None and dev is not None and (r >= PROFILE_R) and (dev <= RATIO_DEV))}

def unity_step_match(pred_steps, actual_steps):
    m = step_match(pred_steps, actual_steps)
    return bool(m['match'] and m['scale'] is not None and (abs(np.log(m['scale'])) <= SCALE_UNITY_TOL))

def lagged(a, b, lag):
    n = min(len(a), len(b))
    a, b = (np.asarray(a)[:n], np.asarray(b)[:n])
    if lag > 0:
        return (a[:-lag], b[lag:])
    if lag < 0:
        return (a[-lag:], b[:lag])
    return (a, b)

def best_lag(a, b, max_lag=MAX_LAG):
    best_l, best_r = (0, None)
    for lag in range(-max_lag, max_lag + 1):
        r = pearson(*lagged(a, b, lag))
        if r is not None and (best_r is None or r > best_r + 1e-12 or (abs(r - best_r) <= 1e-12 and abs(lag) < abs(best_l))):
            best_l, best_r = (lag, r)
    return (best_l, best_r)

def procrustes(p, q, with_scale=False, allow_reflection=False):
    p, q = (np.asarray(p, float), np.asarray(q, float))
    mp, mq = (p.mean(0), q.mean(0))
    x, y = (p - mp, q - mq)
    spread = float(np.sqrt((y ** 2).sum(1).mean()))
    u, sv, vt = np.linalg.svd(x.T @ y)
    d = np.ones(p.shape[1])
    if not allow_reflection and np.linalg.det(vt.T @ u.T) < 0:
        d[-1] = -1.0
    rot = vt.T @ np.diag(d) @ u.T
    denom = float((x ** 2).sum())
    s = float((sv * d).sum() / denom) if with_scale and denom > 0 else 1.0
    resid = y - s * (x @ rot.T)
    return (rot, s, float(np.sqrt((resid ** 2).sum(1).mean()) / spread) if spread > 0 else None)

def rotmats(aa):
    aa = np.asarray(aa, float)
    th = np.linalg.norm(aa, axis=1)
    k = aa / np.where(th > 1e-12, th, 1.0)[:, None]
    kx = np.zeros((len(aa), 3, 3))
    kx[:, 0, 1], kx[:, 0, 2], kx[:, 1, 2] = (-k[:, 2], k[:, 1], -k[:, 0])
    kx[:, 1, 0], kx[:, 2, 0], kx[:, 2, 1] = (k[:, 2], -k[:, 1], k[:, 0])
    s, c = (np.sin(th)[:, None, None], np.cos(th)[:, None, None])
    return np.eye(3)[None] + s * kx + (1.0 - c) * (kx @ kx)

def rot_angle_deg(r):
    return np.degrees(np.arccos(np.clip((np.trace(r, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)))

def orientation_model(tr, tc, g):
    t = len(tr)
    a = np.zeros((3 * t, 6))
    a[:, :3] = (g - np.eye(3)[None]).reshape(3 * t, 3)
    a[:, 3:] = np.tile(np.eye(3), (t, 1))
    y = (tc - np.einsum('tij,tj->ti', g, tr)).reshape(3 * t)
    sol, *_ = np.linalg.lstsq(a, y, rcond=None)
    res = (y - a @ sol).reshape(t, 3)
    yc = tc - tc.mean(0)
    s3, sh = (float(np.sqrt((yc ** 2).sum(1).mean())), float(np.sqrt((yc[:, [0, 2]] ** 2).sum(1).mean())))
    r3 = float(np.sqrt((res ** 2).sum(1).mean())) / s3 if s3 > 0 else None
    rh = float(np.sqrt((res[:, [0, 2]] ** 2).sum(1).mean())) / sh if sh > 0 else None
    return (r3, rh, float(np.linalg.norm(sol[:3])), tc - res)

def motion_ok(rec):
    pose = rec.get('pose') if type(rec) is dict else None
    trans = rec.get('trans') if type(rec) is dict else None
    return bool(type(pose) is np.ndarray and type(trans) is np.ndarray and (pose.ndim == 2) and (trans.ndim == 2) and (pose.shape[1] == 72) and (trans.shape[1] == 3) and (pose.shape[0] == trans.shape[0]) and (pose.shape[0] >= 2) and (pose.dtype.kind == 'f') and (trans.dtype.kind == 'f'))

def align(pose_r, pose_c):
    tr_, tc_ = (len(pose_r), len(pose_c))
    br, bc = (np.asarray(pose_r, float)[:, 3:], np.asarray(pose_c, float)[:, 3:])
    cands = []
    if tr_ == tc_:
        cands.append(('equal', np.arange(tr_), np.arange(tc_)))
    if 0 < tr_ - tc_ <= MAX_OFFSET:
        cands += [('offset', np.arange(k, k + tc_), np.arange(tc_)) for k in range(tr_ - tc_ + 1)]
    if 0 < tc_ - tr_ <= MAX_OFFSET:
        cands += [('offset', np.arange(tr_), np.arange(k, k + tr_)) for k in range(tc_ - tr_ + 1)]
    for m in STRIDES:
        for d in range(m):
            if len(range(d, tr_, m)) == tc_:
                cands.append(('stride', np.arange(d, tr_, m), np.arange(tc_)))
            if len(range(d, tc_, m)) == tr_:
                cands.append(('stride', np.arange(tr_), np.arange(d, tc_, m)))
    best = None
    for kind, ir, ic in cands:
        if len(ir) < 2:
            continue
        mx = float(np.abs(br[ir] - bc[ic]).max())
        if best is None or mx < best[0]:
            best = (mx, kind, ir, ic)
    if best is not None and best[0] <= POSE_TOL:
        return (best[2], best[3], best[1] + '_pose_same')
    if tr_ == tc_:
        return (np.arange(tr_), np.arange(tc_), 'equal_pose_differs')
    return None

def root_vs_canonical(rec_r, rec_c):
    al = align(rec_r['pose'], rec_c['pose'])
    if al is None:
        return {'alignment': 'unaligned', 'scored': False}
    ir, ic, kind = al
    tr = np.asarray(rec_r['trans'], float)[ir]
    tc = np.asarray(rec_c['trans'], float)[ic]
    out = {'alignment': kind, 'scored': False}
    if not (np.isfinite(tr).all() and np.isfinite(tc).all()):
        out['alignment'] = 'nonfinite'
        return out
    sr, sc = (step_norms(tr), step_norms(tc))
    sm = step_match(sr, sc)
    if sm['r'] is None or sm['dev'] is None:
        out['alignment'] = 'too_short_or_static'
        return out
    lag, r_lag = best_lag(sr, sc)
    lag_m = step_match(*lagged(sr, sc, lag)) if lag != 0 else {'match': False}
    _, _, rigid = procrustes(tr, tc)
    _, sim_s, sim = procrustes(tr, tc, with_scale=True)
    g = rotmats(np.asarray(rec_c['pose'], float)[ic, :3]) @ np.transpose(rotmats(np.asarray(rec_r['pose'], float)[ir, :3]), (0, 2, 1))
    orient_spread = float(np.percentile(rot_angle_deg(g @ np.transpose(g[:1], (0, 2, 1))), 95))
    om3, omh, jn, pred = orientation_model(tr, tc, g)
    om_steps = unity_step_match(step_norms(pred), sc)
    om_steps_h = unity_step_match(step_norms(pred[:, [0, 2]]), step_norms(tc[:, [0, 2]]))
    rad = pearson(np.linalg.norm(tr - tr[0], axis=1), np.linalg.norm(tc - tc[0], axis=1))
    if sm['match']:
        mech = 'STEP_MATCH'
    elif lag_m['match']:
        mech = 'TIME_SHIFT'
    elif om3 is not None and om3 <= RESID and om_steps:
        mech = 'ORIENTATION_COUPLED_TRANSLATION'
    elif omh is not None and omh <= RESID and om_steps_h:
        mech = 'ORIENTATION_COUPLED_PLANAR_ONLY'
    else:
        mech = 'UNIDENTIFIED'
    out.update({'scored': True, 'match': sm['match'], 'r': sm['r'], 'scale': sm['scale'], 'dev': sm['dev'], 'lag': lag, 'r_at_lag': r_lag, 'rigid_resid': rigid, 'similarity_resid': sim, 'similarity_scale': sim_s, 'orient_spread_p95_deg': orient_spread, 'orient_model_resid': om3, 'orient_model_resid_h': omh, 'orient_model_offset_norm': jn, 'radial_r': rad, 'mechanism': mech, 'fps_equal': C._as_number(rec_r.get('fps')) == C._as_number(rec_c.get('fps'))})
    return out
_DOWN_RX = re.compile('_down(\\d+)$')

def down_index(key, walk):
    tail = key.split('__', 1)[1][len(walk):] if '__' in key else ''
    m = _DOWN_RX.search(tail)
    return int(m.group(1)) if m else 0

def root_vs_humanml(raw, trans_r, down, rate, vertical_ok):
    raw = np.asarray(raw, float)
    sub = np.asarray(trans_r, float)[down::rate]
    if sub.shape[0] != raw.shape[0] + 1:
        return {'frame_relation_ok': False, 'scored': False}
    if not (np.isfinite(sub).all() and np.isfinite(raw).all()):
        return {'frame_relation_ok': True, 'scored': False, 'nonfinite': True}
    d = np.diff(sub, axis=0)
    r_xz, r_3d = (np.linalg.norm(d[:, [0, 2]], axis=1), np.linalg.norm(d, axis=1))
    h_xz = np.linalg.norm(raw[:, 1:3], axis=1)
    h_3d = np.sqrt(h_xz[:-1] ** 2 + np.diff(raw[:, 3]) ** 2)
    planar = step_match(r_xz, h_xz)
    primary = step_match(r_3d[:-1], h_3d) if vertical_ok else planar
    if primary['r'] is None or primary['dev'] is None:
        return {'frame_relation_ok': True, 'scored': False}
    ref, oth = (r_3d[:-1], h_3d) if vertical_ok else (r_xz, h_xz)
    lag, r_lag = best_lag(ref, oth)
    lag_m = step_match(*lagged(ref, oth, lag)) if lag != 0 else {'match': False}
    path = E2.integrate_root_xz(raw)
    _, s2, res2 = procrustes(sub[:, [0, 2]], path, with_scale=True)
    _, _, res2m = procrustes(sub[:, [0, 2]], path, with_scale=True, allow_reflection=True)
    return {'frame_relation_ok': True, 'scored': True, 'match': primary['match'], 'r': primary['r'], 'scale': primary['scale'], 'dev': primary['dev'], 'planar_match': planar['match'], 'planar_r': planar['r'], 'planar_scale': planar['scale'], 'lag': lag, 'r_at_lag': r_lag, 'mechanism': 'STEP_MATCH' if primary['match'] else 'TIME_SHIFT' if lag_m['match'] else 'UNIDENTIFIED', 'path2d_resid': res2, 'path2d_resid_reflection_allowed': res2m, 'path2d_scale': s2}

def pair_verdict(n_scored, n_match):
    if n_scored < IC.SMALL_CELL:
        return 'INSUFFICIENT'
    share = n_match / n_scored
    if share >= MATCH_SHARE:
        return 'MATCH'
    if share <= NO_MATCH_SHARE:
        return 'NO_MATCH'
    return 'AMBIGUOUS'

def trajectory_class(v_h, v_c, cross_era):
    if cross_era:
        return 'UNRESOLVED'
    if v_h == 'MATCH' and v_c == 'MATCH':
        return 'BOTH_MATCH_WITH_TRANSFORM'
    if v_h == 'MATCH' and v_c == 'NO_MATCH':
        return 'ROOT_MATCHES_HUMANML_ONLY'
    if v_c == 'MATCH' and v_h == 'NO_MATCH':
        return 'ROOT_MATCHES_CANONICAL_ONLY'
    if v_h == 'NO_MATCH' and v_c == 'NO_MATCH':
        return 'NEITHER_MATCH'
    return 'UNRESOLVED'

def mechanism_verdict(mechs):
    bad = [m for m in mechs if m != 'STEP_MATCH']
    if not bad:
        return 'NONE_ALL_MATCH'
    top = max(set(bad), key=bad.count)
    return top if bad.count(top) / len(bad) >= MATCH_SHARE else 'MIXED'

def walk_index(obj):
    groups = {}
    for s in obj:
        groups.setdefault(str(s), []).append(s)
    out, n_coll, n_bad = ({}, 0, 0)
    for s, walks in obj.items():
        if type(walks) is not dict:
            n_bad += 1
            continue
        if len(groups[str(s)]) > 1:
            n_coll += len(walks)
            continue
        for w, rec in walks.items():
            if type(w) is str and type(rec) is dict:
                out[str(s), w] = rec
            else:
                n_bad += 1
    return (out, n_coll, n_bad)

def clinical_raw(name, R, Cn, coll, npz_keys, npz_get):
    both = sorted(set(R) & set(Cn))
    r_only, c_only = (sorted(set(R) - set(Cn)), sorted(set(Cn) - set(R)))
    cells = dict.fromkeys(LABEL_CELLS, 0)
    kinds_r, kinds_c = (dict.fromkeys(KIND_CELLS, 0), dict.fromkeys(KIND_CELLS, 0))
    grades = dict.fromkeys(('grade_0', 'grade_1', 'grade_2', 'grade_3'), 0)
    g3 = {'root': 0, 'canonical': 0, 'both': 0}
    for k in both:
        a, b = (G.label_class(R[k]), G.label_class(Cn[k]))
        cat = G.compare_grades(a, b)
        cells[cat] += 1
        kinds_r[a[0]] += 1
        kinds_c[b[0]] += 1
        if cat == 'EQUAL_GRADE':
            grades[f'grade_{a[1]}'] += 1
        g3['root'] += int(a == ('GRADE', 3))
        g3['canonical'] += int(b == ('GRADE', 3))
        g3['both'] += int(a == ('GRADE', 3) and b == ('GRADE', 3))
    for side, keys, src in (('ROOT', r_only, R), ('CANONICAL', c_only, Cn)):
        for k in keys:
            kind = G.label_class(src[k])[0]
            cells[f'{side}_RECORD_ONLY_' + ('GRADED' if kind == 'GRADE' else 'INVALID' if kind == 'INVALID' else 'UNGRADED')] += 1
    cells['IDENTITY_COLLISION_RECORDS'] = coll
    fields = {}
    for f in ('medication', 'other'):
        fc = dict.fromkeys(FIELD_CELLS, 0)
        for k in both:
            fc[G.compare_field(G.field_value(R[k], f), G.field_value(Cn[k], f))] += 1
        fields[f] = fc
    index_c = {k: (0, 0.0, None) for k in Cn}
    chosen, _ = K.join(sorted(npz_keys), index_c)
    sub = {'EQUAL_GRADE': 0, 'CHANGED_GRADE': 0, 'ROOT_UNGRADED_OR_INVALID': 0, 'ROOT_RECORD_MISSING': 0}
    for k, (key, _d, _n) in chosen.items():
        kc, vc = G.label_class(Cn[k])
        if kc != 'GRADE' or vc == 3:
            continue
        raw = K.first_view(np.asarray(npz_get(key)))
        if raw is None or not np.isfinite(raw).all() or raw.shape[0] < C3X_MIN_FRAMES:
            continue
        if k not in R:
            sub['ROOT_RECORD_MISSING'] += 1
            continue
        kr, vr = G.label_class(R[k])
        sub['ROOT_UNGRADED_OR_INVALID' if kr != 'GRADE' else 'EQUAL_GRADE' if vr == vc else 'CHANGED_GRADE'] += 1
    records = {'both': len(both), 'root_only': len(r_only), 'canonical_only': len(c_only)}
    return {'records': records, 'labels': cells, 'kinds_root': kinds_r, 'kinds_canonical': kinds_c, 'grades_equal': grades, 'grade3': g3, 'fields': fields, 'c3x_subset': sub}

def label_verdict(cells):
    if sum((cells[k] for k in MISMATCH_CELLS)):
        return 'LABEL_PARITY_FAIL'
    if sum((cells[k] for k in UNRESOLVED_CELLS)) or cells['EQUAL_GRADE'] == 0:
        return 'LABEL_PARITY_PARTIAL'
    return 'LABEL_PARITY_PASS'

def subset_verdict(sub):
    if sub['CHANGED_GRADE']:
        return 'C3X_SUBSET_LABELS_DIFFER'
    if sub['ROOT_UNGRADED_OR_INVALID'] or sub['ROOT_RECORD_MISSING'] or sub['EQUAL_GRADE'] == 0:
        return 'C3X_SUBSET_PARTIAL'
    return 'C3X_SUBSET_LABELS_EQUAL'

def trajectory_raw(name, R, Cn, npz_keys, npz_get):
    both = sorted(set(R) & set(Cn))
    rc, malformed = ([], 0)
    subj_of = {}
    for k in both:
        if not (motion_ok(R[k]) and motion_ok(Cn[k])):
            malformed += 1
            continue
        res = root_vs_canonical(R[k], Cn[k])
        res['subject'] = subj_of.setdefault(k[0], len(subj_of))
        rc.append(res)
    root_index = {k: (int(rec['trans'].shape[0]), 0.0, None) for k, rec in R.items() if motion_ok(rec)}
    chosen, _ = K.join(sorted(npz_keys), root_index)
    rh, n_no_key = ([], len(root_index) - len(chosen))
    rate, vertical = (OFFICIAL_RATE[name], name not in VERTICAL_DISCARDED)
    for k, (key, _d, _n) in sorted(chosen.items()):
        raw = K.first_view(np.asarray(npz_get(key)))
        if raw is None:
            rh.append({'frame_relation_ok': False, 'scored': False, 'bad_array': True})
            continue
        res = root_vs_humanml(raw, R[k]['trans'], down_index(key, k[1]), rate, vertical)
        res['subject'] = subj_of.setdefault(k[0], len(subj_of))
        rh.append(res)
    fps_root = [C._as_number(R[k].get('fps')) for k in root_index]
    return {'rc': rc, 'rc_malformed': malformed, 'rh': rh, 'rh_root_without_key': n_no_key, 'rh_root_walks_motion_ok': len(root_index), 'fps_root_official': sum((1 for f in fps_root if f == OFFICIAL_FPS[name])), 'fps_root_other': sum((1 for f in fps_root if f != OFFICIAL_FPS[name]))}

def _q(rows, key, digits=4):
    return G.quantiles([r[key] for r in rows if r.get(key) is not None], digits=digits)

def trajectory_partitions(t):
    rc_scored = [r for r in t['rc'] if r['scored']]
    rh_scored = [r for r in t['rh'] if r['scored']]
    align_cells = {}
    for r in t['rc']:
        align_cells[r['alignment']] = align_cells.get(r['alignment'], 0) + 1
    for kind in ('equal_pose_same', 'offset_pose_same', 'stride_pose_same', 'equal_pose_differs', 'unaligned', 'nonfinite', 'too_short_or_static'):
        align_cells.setdefault(kind, 0)
    align_cells['malformed_motion'] = t['rc_malformed']
    mech_rc = {m: 0 for m in ('STEP_MATCH',) + MECHANISMS}
    for r in rc_scored:
        mech_rc[r['mechanism']] += 1
    mech_rh = {m: 0 for m in ('STEP_MATCH', 'TIME_SHIFT', 'UNIDENTIFIED')}
    for r in rh_scored:
        mech_rh[r['mechanism']] += 1
    parts = {'rc_alignment': align_cells, 'rc_match': {'match': sum((r['match'] for r in rc_scored)), 'no_match': sum((not r['match'] for r in rc_scored))}, 'rc_mechanism': mech_rc, 'rc_lag': {'lag_zero': sum((r['lag'] == 0 for r in rc_scored)), 'lag_nonzero': sum((r['lag'] != 0 for r in rc_scored))}, 'rc_scale_unity': {'unity': sum((abs(np.log(r['scale'])) <= SCALE_UNITY_TOL for r in rc_scored)), 'not_unity': sum((abs(np.log(r['scale'])) > SCALE_UNITY_TOL for r in rc_scored))}, 'rc_orientation_constant': {'constant': sum((r['orient_spread_p95_deg'] <= ORIENT_TOL_DEG for r in rc_scored)), 'varies': sum((r['orient_spread_p95_deg'] > ORIENT_TOL_DEG for r in rc_scored))}, 'rc_fps_equal': {'equal': sum((bool(r['fps_equal']) for r in rc_scored)), 'differs': sum((not r['fps_equal'] for r in rc_scored))}, 'rh_join': {'joined': len(t['rh']), 'root_walk_without_input_key': t['rh_root_without_key']}, 'rh_frames': {'official_relation': sum((r['frame_relation_ok'] for r in t['rh'])), 'unexpected_relation': sum((not r['frame_relation_ok'] for r in t['rh'])), 'scored': len(rh_scored)}, 'rh_match': {'match': sum((r['match'] for r in rh_scored)), 'no_match': sum((not r['match'] for r in rh_scored))}, 'rh_planar_same_frame': {'match': sum((r['planar_match'] for r in rh_scored)), 'no_match': sum((not r['planar_match'] for r in rh_scored))}, 'rh_mechanism': mech_rh, 'root_fps_vs_official': {'official': t['fps_root_official'], 'other': t['fps_root_other']}}
    return (parts, rc_scored, rh_scored)

def cohort_report(name, root, canon, npz_keys, npz_get, e2_point):
    R, coll_r, bad_r = walk_index(root)
    Cn, coll_c, bad_c = walk_index(canon)
    clin = clinical_raw(name, R, Cn, coll_r + coll_c, npz_keys, npz_get)
    traj = trajectory_raw(name, R, Cn, npz_keys, npz_get)
    t_parts, rc_s, rh_s = trajectory_partitions(traj)
    clin['records']['malformed_entries'] = bad_r + bad_c
    c_parts = [clin['records'], clin['labels'], clin['kinds_root'], clin['kinds_canonical'], clin['c3x_subset'], clin['fields']['medication'], clin['fields']['other'], clin['grade3']]
    coarse_c = G.has_small(*c_parts)
    coarse_t = coarse_c or G.has_small(*t_parts.values())
    verdict = label_verdict(clin['labels'])
    gp = G.render_partition(clin['grades_equal'], G.has_small(clin['grades_equal'])) if verdict == 'LABEL_PARITY_PASS' else 'WITHHELD_LABELS_NOT_PASS'
    clinical = {'mode': 'coarse' if coarse_c else 'exact', 'records': G.render_partition(clin['records'], coarse_c), 'label_partition': G.render_partition(clin['labels'], coarse_c), 'label_verdict': verdict, 'label_encodings_root_joined': G.render_partition(clin['kinds_root'], coarse_c), 'label_encodings_canonical_joined': G.render_partition(clin['kinds_canonical'], coarse_c), 'grade3_sets_same': clin['grade3']['root'] == clin['grade3']['both'] == clin['grade3']['canonical'], 'grade3_counts': G.render_partition(clin['grade3'], coarse_c), 'grade_counts_equal_walks': gp, 'c3x_subset': G.render_partition(clin['c3x_subset'], coarse_c), 'c3x_subset_verdict': subset_verdict(clin['c3x_subset']), 'medication': G.render_partition(clin['fields']['medication'], coarse_c), 'other_field': G.render_partition(clin['fields']['other'], coarse_c)}
    v_c = pair_verdict(len(rc_s), t_parts['rc_match']['match'])
    v_h = pair_verdict(len(rh_s), t_parts['rh_match']['match'])
    cross = name in CROSS_ERA_HUMANML
    cls = trajectory_class(v_h, v_c, cross)
    mech_c = mechanism_verdict([r['mechanism'] for r in rc_s])
    mech_h = mechanism_verdict([r['mechanism'] for r in rh_s])
    scale_h = [r['scale'] for r in rh_s if r.get('scale')]
    spread_h, n_multi = E2.within_participant_spread(np.array(scale_h, float), np.array([r['subject'] for r in rh_s if r.get('scale')]))
    qs = G.quantiles(scale_h, digits=5)
    trajectory = {'mode': 'coarse' if coarse_t else 'exact', 'partitions': {k: G.render_partition(v, coarse_t) for k, v in t_parts.items()}, 'root_vs_canonical': {'verdict': v_c, 'share_match': G.share_out(t_parts['rc_match']['match'], len(rc_s), coarse_t), 'mechanism_verdict': mech_c, 'r_step': _q(rc_s, 'r'), 'scale': _q(rc_s, 'scale', 5), 'ratio_dev': _q(rc_s, 'dev', 5), 'rigid_resid': _q(rc_s, 'rigid_resid'), 'similarity_resid': _q(rc_s, 'similarity_resid'), 'similarity_scale': _q(rc_s, 'similarity_scale'), 'orient_spread_p95_deg': _q(rc_s, 'orient_spread_p95_deg', 3), 'orient_model_resid': _q(rc_s, 'orient_model_resid'), 'orient_model_resid_h': _q(rc_s, 'orient_model_resid_h'), 'orient_model_offset_norm': _q(rc_s, 'orient_model_offset_norm'), 'radial_r': _q(rc_s, 'radial_r'), 'r_at_best_lag': _q(rc_s, 'r_at_lag')}, 'root_vs_humanml': {'verdict': 'CROSS_ERA_DESCRIPTIVE_ONLY' if cross else v_h, 'verdict_if_same_era': v_h, 'share_match': G.share_out(t_parts['rh_match']['match'], len(rh_s), coarse_t), 'comparison': 'planar (vertical discarded by the official conversion)' if name in VERTICAL_DISCARDED else 'three-dimensional', 'official_rate': OFFICIAL_RATE[name], 'mechanism_verdict': mech_h, 'r_step': _q(rh_s, 'r'), 'scale': qs, 'ratio_dev': _q(rh_s, 'dev', 5), 'scale_rel_iqr_across_walks': None if qs is None else round((qs['q75'] - qs['q25']) / qs['median'], 5), 'scale_within_participant_spread_median': None if spread_h is None else round(spread_h, 5), 'planar_r_same_frame': _q(rh_s, 'planar_r'), 'planar_scale': _q(rh_s, 'planar_scale', 5), 'path2d_resid': _q(rh_s, 'path2d_resid'), 'path2d_resid_reflection_allowed': _q(rh_s, 'path2d_resid_reflection_allowed'), 'r_at_best_lag': _q(rh_s, 'r_at_lag')}, 'humanml_vs_canonical_e2_parity': e2_point(), 'class': cls, 'derivation_step': mech_c if cls == 'ROOT_MATCHES_HUMANML_ONLY' and mech_c in IDENTIFIED else mech_h if cls == 'ROOT_MATCHES_CANONICAL_ONLY' and mech_h in IDENTIFIED else 'NOT_REQUIRED' if cls == 'BOTH_MATCH_WITH_TRANSFORM' else 'UNRESOLVED'}
    return {'clinical': clinical, 'trajectory': trajectory}

def load_root(cohort, receipt_rows):
    logical, fid, size, md5 = G.ROOT_PINS[cohort][:4]
    row = receipt_rows.get(cohort)
    if row is None or row.get('publisher_datafile') != fid or row.get('md5_pinned') != md5:
        IC.stop('download receipt does not cover this file', 23)
    path = G.root_path(cohort)
    if not path.is_file():
        IC.stop('root file missing', 23)
    data = path.read_bytes()
    import hashlib
    if len(data) != size or hashlib.md5(data).hexdigest() != md5 or hashlib.sha256(data).hexdigest() != row.get('sha256'):
        IC.stop('root file does not match its pins', 23)
    try:
        return G.strict_loads(data)
    except G.SourceError:
        IC.stop('root file refused by the strict reader', 24)
