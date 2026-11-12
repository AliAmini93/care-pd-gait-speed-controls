"""Fabricated-only regression tests of the portable residual kernels. No real data is read."""
from __future__ import annotations
import argparse
import os
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'care_pd'))
import abcd_privacy as AP
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import abcd_baseline as AB
import abcd_common as AC
import abcd_worlds as W
import common as C
SCHEMA = 'gait_abcd_synthetic_v1'
MIN_CLASS_WALKS = 15
MIN_CLASS_PERSONS = 4
SPEED_RHO_MIN = 0.95
S_TARGET_MIN = 0.4
W1_ORACLE_GAIN_MIN = 0.05
W3_ORACLE_GAIN_MAX = 0.03
W2_RHO_ABS_MIN = 0.5
W4_RELABEL_MIN = 0.3
W4_ORACLE_DROP_MIN = 0.1
HEADING_R_MAX = 0.3
FX_UNIT_RATIO = (0.008, 0.012)

def _oracle_fit_predict(X, y, tr, te):
    sc = StandardScaler().fit(X[tr])
    lr = LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000).fit(sc.transform(X[tr]), y[tr])
    return lr.predict(sc.transform(X[te]))

def _oracle_X(data, cols):
    return np.array([[[np.log(t['v']), t['log_arm'], t['sigma_cycle'], t['asym']][c] for c in cols] for t in data['truth']])

def oracle_f1(data, cols, labels_for_target=None):
    """Oracle trained on the three sources, scored on the fake target (truth variables, fixed LR)."""
    src, tgt = W.split(data)
    pred = _oracle_fit_predict(_oracle_X(data, cols), data['label'], src, tgt)
    yt = data['label'][tgt] if labels_for_target is None else labels_for_target
    return AC.macro_f1(yt, pred)

def oracle_f1_loeo(data, cols):
    """Mean leave-one-environment-out oracle macro-F1 over all four environments (more held-out persons)."""
    X, y, env = (_oracle_X(data, cols), data['label'], data['env'])
    out = []
    for e in range(len(W.ENVS)):
        tr, te = (np.flatnonzero(env != e), np.flatnonzero(env == e))
        out.append(AC.macro_f1(y[te], _oracle_fit_predict(X, y, tr, te)))
    return float(np.mean(out))

def s_target_f1(data):
    b = AC.synthetic_bundle(data)
    src, tgt = W.split(data)
    warned = [0]
    s = AB.s_fold(b['speed'][src][:, None], b['label'][src], b['site'][src], b['subject'][src], b['speed'][tgt][:, None], warned)
    return (AC.macro_f1(b['label'][tgt], s['proba'].argmax(axis=1)), s['name'], b)

def validate_worlds(seed_set):
    """Generator validator. Returns (checks dict, descriptive dict)."""
    reps = range(W.REPLICATES[seed_set])
    checks, desc = ({}, {'worlds': {}})
    t0 = time.monotonic()
    a = W.generate('W1', seed_set, 0)
    b_ = W.generate('W1', seed_set, 0)
    checks['V01_determinism'] = {'pass': W.digest(a) == W.digest(b_)}
    del b_
    digests, schema_ok, support_ok, support = ({}, True, True, {})
    speed_rho, s_f1, oracle, timebase_ok, heading_r = ({}, {}, {}, True, [])
    w2_rho, w4 = ([], [])
    for world in W.WORLDS + W.FALSIFIERS:
        for r in reps:
            d = a if world == 'W1' and r == 0 else W.generate(world, seed_set, r)
            digests[f'{world}_r{r}'] = W.digest(d)
            n = len(d['pose'])
            schema_ok &= n == len(W.ENVS) * W.PERSONS_PER_ENV * W.WALKS_PER_PERSON
            for p, tr, fps, e in zip(d['pose'], d['trans'], d['fps'], d['env']):
                schema_ok &= p.ndim == 2 and p.shape[1] == 72 and (tr.shape == (p.shape[0], 3)) and (p.shape[0] >= 8) and np.isfinite(p).all() and np.isfinite(tr).all() and (fps == W.FPS[e])
            schema_ok &= bool(np.isin(d['label'], (0, 1, 2)).all())
            for e in range(len(W.ENVS)):
                m = d['env'] == e
                schema_ok &= np.unique(d['person'][m]).size == W.PERSONS_PER_ENV
                walks = [int((d['label'][m] == k).sum()) for k in range(3)]
                persons = [int(np.unique(d['person'][m & (d['label'] == k)]).size) for k in range(3)]
                support[f'{world}_r{r}_{W.ENVS[e]}'] = {'walks_by_grade': walks, 'persons_by_grade': persons}
                support_ok &= min(walks) >= MIN_CLASS_WALKS and min(persons) >= MIN_CLASS_PERSONS
            timebase_ok &= all((abs(t['tau']) <= W.TIMEBASE + 1e-12 for t in d['truth']))
            if world in ('W1', 'W3'):
                b = AC.synthetic_bundle(d)
                for e in range(len(W.ENVS)):
                    m = np.flatnonzero(d['env'] == e)
                    truev = np.array([d['truth'][i]['v'] * W.MEAS_SCALE[e] for i in m])
                    speed_rho[f'{world}_r{r}_{W.ENVS[e]}'] = float(stats.spearmanr(b['speed'][m], truev).statistic)
                f1, name, _ = s_target_f1(d)
                s_f1[f'{world}_r{r}'] = {'macro_f1': f1, 'selected': name}
                oracle[f'{world}_r{r}'] = {'speed_only': oracle_f1_loeo(d, [0]), 'speed_plus_cues': oracle_f1_loeo(d, [0, 1, 2, 3]), 'target_only_speed': oracle_f1(d, [0]), 'target_only_plus_cues': oracle_f1(d, [0, 1, 2, 3])}
            if world == 'W1':
                for e in range(len(W.ENVS)):
                    m = np.flatnonzero(d['env'] == e)
                    hd = []
                    for i in m:
                        R0 = AC.RC.rotvec_to_matrix(d['pose'][i][0, 0:3])
                        hd.append(np.arctan2(R0[0, 2], R0[2, 2]))
                    hd = np.asarray(hd)
                    heading_r.append(float(np.hypot(np.cos(hd).mean(), np.sin(hd).mean())))
            if world == 'W2':
                for e in range(len(W.ENVS)):
                    m = np.flatnonzero(d['env'] == e)
                    jit = np.array([d['truth'][i]['jitter'] for i in m])
                    w2_rho.append((e, float(stats.spearmanr(jit, d['label'][m]).statistic)))
            if world == 'W4':
                src, tgt = W.split(d)
                src_lab = np.array([d['truth'][i]['source_threshold_label'] for i in tgt])
                relabel = float((src_lab != d['label'][tgt]).mean())
                f1_rubric = oracle_f1(d, [0, 1, 2, 3])
                f1_source_rubric = oracle_f1(d, [0, 1, 2, 3], labels_for_target=src_lab)
                w4.append({'relabel_share': relabel, 'oracle_f1_target_rubric': f1_rubric, 'oracle_f1_if_source_rubric': f1_source_rubric, 'drop': f1_source_rubric - f1_rubric})
            if world == 'FX_APERIODIC':
                checks.setdefault('V10a_aperiodic_flag', {'pass': True})
                checks['V10a_aperiodic_flag']['pass'] &= all((not t['periodic'] for t in d['truth']))
            if world == 'FX_UNIT':
                b = AC.synthetic_bundle(d)
                ref = AC.synthetic_bundle(W.generate('W1', seed_set, r))
                tg = d['env'] == W.TARGET_ENV
                ratio = float(np.median(b['speed'][tg]) / np.median(ref['speed'][tg]))
                checks.setdefault('V10b_unit_rescale', {'pass': True, 'ratio': []})
                checks['V10b_unit_rescale']['ratio'].append(ratio)
                checks['V10b_unit_rescale']['pass'] &= FX_UNIT_RATIO[0] <= ratio <= FX_UNIT_RATIO[1]
            del d
    checks['V02_schema'] = {'pass': bool(schema_ok)}
    checks['V02b_class_support'] = {'pass': bool(support_ok), 'min_walks': MIN_CLASS_WALKS, 'min_persons': MIN_CLASS_PERSONS}
    checks['V03_speed_validity'] = {'pass': bool(min(speed_rho.values()) >= SPEED_RHO_MIN), 'min_rho': min(speed_rho.values())}
    for world in ('W1', 'W3'):
        vals = [s_f1[f'{world}_r{r}']['macro_f1'] for r in reps]
        checks[f'V04_S_informative_{world}'] = {'pass': bool(np.mean(vals) >= S_TARGET_MIN), 'mean': float(np.mean(vals))}
    g1 = np.mean([oracle[f'W1_r{r}']['speed_plus_cues'] - oracle[f'W1_r{r}']['speed_only'] for r in reps])
    g3 = np.mean([oracle[f'W3_r{r}']['speed_plus_cues'] - oracle[f'W3_r{r}']['speed_only'] for r in reps])
    checks['V05_W1_oracle_incremental'] = {'pass': bool(g1 >= W1_ORACLE_GAIN_MIN), 'mean_gain': float(g1)}
    checks['V06_W3_oracle_null'] = {'pass': bool(g3 <= W3_ORACLE_GAIN_MAX), 'mean_gain': float(g3)}
    src_rho = [rho for e, rho in w2_rho if e in W.SOURCE_ENVS]
    tgt_rho = [rho for e, rho in w2_rho if e == W.TARGET_ENV]
    checks['V07_W2_spurious_flip'] = {'pass': bool(min(src_rho) >= W2_RHO_ABS_MIN and max(tgt_rho) <= -W2_RHO_ABS_MIN), 'min_source_rho': min(src_rho), 'max_target_rho': max(tgt_rho)}
    checks['V08_W4_rubric_shift'] = {'pass': bool(min((x['relabel_share'] for x in w4)) >= W4_RELABEL_MIN and np.mean([x['drop'] for x in w4]) >= W4_ORACLE_DROP_MIN), 'replicates': w4}
    checks['V09_heading_spread'] = {'pass': bool(max(heading_r) <= HEADING_R_MAX), 'max_resultant': max(heading_r)}
    checks['V11_timebase'] = {'pass': bool(timebase_ok), 'bound': W.TIMEBASE}
    desc.update({'digests': digests, 'class_support': support, 'speed_rho': speed_rho, 'S_target_macro_f1': s_f1, 'oracle_macro_f1': oracle, 'W2_jitter_label_rho_by_env': [{'env': W.ENVS[e], 'rho': r_} for e, r_ in w2_rho], 'seconds': round(time.monotonic() - t0, 1)})
    return (checks, desc)

def main():
    checks, _ = validate_worlds('dev')
    for name, result in checks.items():
        if not result['pass']:
            print('FAIL ' + name, flush=True)
            return 1
        print('PASS ' + name, flush=True)
    print('ABCD_WORLDS_TESTS_PASS n=' + str(len(checks)), flush=True)
    return 0
if __name__ == '__main__':
    sys.exit(main())
