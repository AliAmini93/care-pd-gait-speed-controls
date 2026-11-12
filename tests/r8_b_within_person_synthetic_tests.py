"""Fabricated-only tests for R8 card b_within_person (never reads real data; values are not evidence)."""
from __future__ import annotations
import json
import copy
import os
import sys
import tempfile
from pathlib import Path
import numpy as np
from scipy import stats
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'care_pd'))
sys.path.insert(1, str(Path(__file__).resolve().parent))
import common as C
import p0_core as P
import p1_core as Q
import r8_core as R
import release_context as RX
import r8_b_within_person as B
import r8_a_occasion_grain_synthetic_tests as AT
FAILS = []
MARK = AT.MARK

def check(cond, name):
    print(('PASS ' if cond else 'FAIL ') + name)
    if not cond:
        FAILS.append(name)

def raises_source(fn):
    try:
        fn()
    except C.SourceError:
        return True
    return False

def test_partial_spearman():
    rng = np.random.default_rng(1)
    n = 200
    speed = rng.normal(size=n)
    grade = np.clip(np.round(1 - speed + rng.normal(0, 0.3, n)), 0, 2)
    leaks = []
    for sd in range(40):
        rr = np.random.default_rng(100 + sd)
        sp = rr.normal(size=n)
        gr = np.clip(np.round(1 - sp + rr.normal(0, 0.3, n)), 0, 2)
        leaks.append(B.partial_spearman(gr, sp, 2 * sp + rr.normal(0, 0.1, n)))
    leaks = np.asarray([x for x in leaks if x is not None])
    check(abs(leaks.mean()) < 0.06 and leaks.size >= 35, f'a near-deterministic function of speed keeps only a small mean leak after adjustment (mean {leaks.mean():+.3f}, sd {leaks.std():.3f})')
    hidden = rng.normal(size=n)
    grade2 = np.clip(np.round(1 - speed + hidden + rng.normal(0, 0.2, n)), 0, 2)
    feat_hidden = hidden + rng.normal(0, 0.2, n)
    r2 = B.partial_spearman(grade2, speed, feat_hidden)
    check(r2 is not None and r2 > 0.4, f'a feature carrying grade information beyond speed survives adjustment ({r2:.3f})')
    check(B.partial_spearman([0, 1, 2], [1, 2, 3], [1, 2, 3]) is None, 'fewer than 4 walks is undefined')
    check(B.partial_spearman([1, 1, 1, 1], [1, 2, 3, 4], [1, 3, 2, 4]) is None, 'a constant grade is undefined')
    check(B.partial_spearman([0, 1, 0, 1], [1, 1, 1, 1], [1, 3, 2, 4]) is None, 'a constant speed is undefined')
    check(B.partial_spearman([0, 1, 0, 1], [1, 2, 3, 4], [5, 5, 5, 5]) is None, 'a constant feature is undefined')
    check(B.partial_spearman([0, 1, 2, 2], [4, 3, 2, 1], [4, 3, 2, 1]) is None, 'grade fully explained by speed gives an undefined value, not a number')
    st = B.sign_test([0.2, 0.1, -0.3, None, 0.0])
    check((st['n_defined'], st['n_positive'], st['n_negative'], st['n_zero']) == (4, 2, 1, 1) and st['sign_test_two_sided_p'] == 1.0 and (abs(st['mean']) < 1e-12) and (abs(st['median'] - 0.05) < 1e-12), 'sign test counts, drops None, exact binomial on positives vs negatives')
    noise = rng.normal(size=n)
    r3 = B.partial_spearman(grade2, speed, noise)
    check(r3 is not None and abs(r3) < 0.2, 'pure noise stays near zero')
    draws = B.boot_mean([[0.1, 0.2, 0.3], [0.5]], 50, 3)
    check(draws.shape == (50,) and np.isfinite(draws).all() and (draws.min() >= 0.1) and (draws.max() <= 0.5), 'stratified person bootstrap resamples within groups')
    check(np.isnan(B.boot_mean([[]], 5, 1)).all(), 'bootstrap of no persons is undefined')

def plant_signal(objs, site, feature_col, strength, rng):
    """Within each person, add to knee-rotation amplitude a term that tracks the grade (beyond speed)."""
    for pid, walks in objs[site].items():
        for rec in walks.values():
            lab = rec['UPDRS_GAIT']
            pose = rec['pose'].astype(np.float64)
            T = pose.shape[0]
            tt = np.arange(T) / rec['fps']
            pose[:, feature_col] += strength * lab * np.sin(2 * np.pi * tt)
            rec['pose'] = pose.astype(np.float32)
    return objs

def make_within_person_variation(objs, rng):
    """Give persons mixed grades within each non-BMCLab site by reassigning labels per walk and
    regenerating forward speed from the new label (fabricated speed world)."""
    for site in ('PD-GaM', 'T-SDU-PD', '3DGait'):
        for pid, walks in objs[site].items():
            keys = list(walks)
            for i, k in enumerate(keys):
                rec = walks[k]
                lab = rec['UPDRS_GAIT']
                if lab == 3:
                    continue
                new = int((lab + i % 3) % 3)
                rec['UPDRS_GAIT'] = new
                T = rec['trans'].shape[0]
                tt = np.arange(T) / rec['fps']
                v = max(0.05, 1.4 - 0.4 * new + rng.normal(0, 0.12))
                trans = rec['trans'].astype(np.float64)
                trans[:, 2] = v * tt
                rec['trans'] = trans.astype(np.float32)
    return objs

def synth_table(mode, seed):
    rng = np.random.default_rng(seed)
    objs = AT.synth_objs(mode, seed=seed)
    objs = make_within_person_variation(objs, rng)
    if mode == 'planted':
        objs = plant_signal(objs, 'PD-GaM', 3 * 1, 0.8, rng)
        objs = plant_signal(objs, 'T-SDU-PD', 3 * 1, 0.8, rng)
        objs = plant_signal(objs, 'BMCLab', 3 * 1, 0.8, rng)
    t = Q.build_table(objs)
    return (objs, t)

def run_analyze(t, med_site, n_boot=40, seed=5):
    """The frozen analysis entry (real-data gates included) on fabricated data."""
    body = B.analyze(t, med_site, n_boot=n_boot, seed=seed)
    return {'feature_names': body['feature_names'], 'per_site': body['unit_W_within_person_partial']['per_site'], 'pooled': body['unit_W_within_person_partial']['pooled'], 'criterion': body['criterion'], 'unit_T': body['unit_T_3dgait_two_walk_descriptive'], 'unit_P': body['unit_P_bmclab_paired_cells']}

def test_pipeline_null_and_planted():
    objs, t = synth_table('speed', 51)
    med_site = B.collect_medication(objs['BMCLab'])
    X, names = B.feature_matrix(t)
    check(X.shape == (t['label'].size, 24) and names[-1] == 'noise_control' and (names[0] == Q.KIN_NAMES[0]), '24 named features incl. the noise control')
    res = run_analyze(t, med_site)
    per = res['per_site']
    check(per['PD-GaM']['n_persons_used'] > 10 and per['T-SDU-PD']['n_persons_used'] > 5, 'fabricated mixed-grade persons are used in PD-GaM and T-SDU-PD')
    check(per['PD-GaM']['speed_grade_spearman']['n_negative'] > per['PD-GaM']['speed_grade_spearman']['n_positive'], 'fabricated speed signal: within-person speed-grade correlation is mostly negative')
    check(res['criterion']['reading'] == 'NO_WITHIN_PERSON_SIGNAL_BEYOND_SPEED' and res['criterion']['negative_control_clean'], f"speed-only fabricated world: no feature survives ({res['criterion']['reading']}, survivors {res['criterion']['survivors']})")
    text = R.render(res)
    check(MARK not in text, 'no fabricated identifier reaches the output')
    check(len(text.encode()) < 262144 * 0.6, f'output well under the summary cap ({len(text)} bytes)')
    res2 = run_analyze(t, med_site)
    check(R.render(res) == R.render(res2), 'deterministic for a fixed seed')
    up = res['unit_P']
    check(up['n_two_state_people'] == 19 and sum(up['delta_grade_counts_off_minus_on'].values()) == 19, 'BMCLab paired cells: 19 persons, grade-difference counts sum to 19')
    check(up['n_discordant'] == 8, 'BMCLab paired cells: 8 discordant persons in the fabricated structure')
    ut = res['unit_T']
    ci3 = C.COHORTS.index('3DGait')
    n_elig = len(np.unique(t['subject'][np.isin(t['label'], (0, 1, 2)) & (t['cohort'] == ci3)]))
    check(ut['n_persons_with_two_walks'] + ut['n_persons_other_walk_count'] == n_elig, '3DGait eligible persons are all counted')
    check(all((res['pooled'][n]['speed_likeness_within_person']['n_defined'] > 0 for n in names if n != 'duration')), 'speed-likeness is reported for every feature (fabricated walks share one duration, so that one is undefined)')
    check(res['criterion']['per_feature']['S_alt']['speed_proxy_suspect'], 'S_alt is flagged as a speed proxy within person')
    check(not res['criterion']['per_feature']['S_alt']['survives'] and (not res['criterion']['per_feature']['duration']['survives']), 'speed-like references can never survive')
    objs_p, t_p = synth_table('planted', 52)
    med_p = B.collect_medication(objs_p['BMCLab'])
    res_p = run_analyze(t_p, med_p)
    c = res_p['criterion']
    check('amp_hip' in c['survivors'] or 'angspeed_hip' in c['survivors'] or len(c['survivors']) > 0, f"planted within-person signal is detected ({c['reading']}, survivors {c['survivors']})")
    check(not c['per_feature']['noise_control']['survives'], 'noise control never survives in the planted world')
    check(c['reading'] == 'WITHIN_PERSON_CANDIDATE', 'planted world yields the candidate reading')
    fp = res_p['unit_P']['features']
    planted_names = [n for n in names if n.startswith('amp_hip') or n.startswith('angspeed_hip')]
    vals = [fp[n]['spearman_delta_grade_delta_feature']['point'] for n in planted_names]
    check(any((v is not None and abs(v) > 0.3 for v in vals)), f'BMCLab paired-cell analysis sees the planted feature change ({[round(v, 2) if v is not None else None for v in vals]})')
    return (objs, t, med_site)

def test_gates(objs, t, med_site):
    got = P.within_subject_control(t, n_boot=10)
    p0_match = {'status': 'P0_COMPLETE', 'within_subject_control': got}
    check(bool(RX.check_within_parity(t, p0_match)), 'locally computed synthetic P0 passes data-derived parity')
    mutations = [
        ('top-level count', lambda p: p['within_subject_control'].update(n_subjects=got['n_subjects'] + 1)),
        ('cohort count', lambda p: p['within_subject_control']['per_cohort'][C.COHORTS[0]].update(n_subjects=1)),
        ('missing cohort', lambda p: p['within_subject_control']['per_cohort'].pop(C.COHORTS[0])),
        ('wrong count type', lambda p: p['within_subject_control'].update(n_subjects=str(got['n_subjects']))),
        ('boolean count', lambda p: p['within_subject_control'].update(n_undefined=False)),
        ('missing status', lambda p: p.pop('status')),
        ('wrong status', lambda p: p.update(status='STOP')),
        ('undefined mismatch', lambda p: p['within_subject_control'].update(n_undefined=got['n_undefined'] + 1)),
        ('missing deterministic field', lambda p: p['within_subject_control'].pop('supports_negative')),
        ('missing cohort mean', lambda p: p['within_subject_control']['per_cohort'][C.COHORTS[0]].pop('mean_rho')),
        ('nonfinite mean', lambda p: p['within_subject_control'].update(mean_rho=float('nan'))),
        ('malformed interval', lambda p: p['within_subject_control'].update(mean_rho_ci95=[0.0])),
    ]
    for label, mutate in mutations:
        bad_p0 = copy.deepcopy(p0_match)
        mutate(bad_p0)
        check(raises_source(lambda: RX.check_within_parity(t, bad_p0)), 'data-derived parity refuses ' + label)
    sparse = dict(t, speed=np.ones_like(t['speed']))
    sparse_p0 = {'status': 'P0_COMPLETE', 'within_subject_control': P.within_subject_control(sparse, n_boot=10)}
    check(RX.check_within_parity(sparse, sparse_p0)['n_subjects'] == 0, 'all-undefined synthetic sparse schema passes')
    sparse_p0['within_subject_control']['n_undefined'] += 1
    check(raises_source(lambda: RX.check_within_parity(sparse, sparse_p0)), 'all-undefined synthetic mismatch fails closed')
    import run_followup as F
    body = F.compute('r8b', objs, t, p0_match, None)
    check(body['status'] == B.STATUS and body['p0_within_subject_parity']['per_cohort'] == got['per_cohort'], 'portable R8-B wrapper completes on fabricated inputs')
    t_bad = dict(t, label=t['label'].copy())
    ci = C.COHORTS.index('BMCLab')
    m = np.flatnonzero((t_bad['cohort'] == ci) & np.isin(t_bad['label'], (0, 1, 2)))
    t_bad['label'][m] = 1
    med_const = list(med_site)
    check(raises_source(lambda: B.analyze(t_bad, med_const, n_boot=5, seed=1)), 'analyze enforces the unit-W person-count gate (BMCLab must contribute 8 persons)')
    bad = list(med_site)
    bad[0] = 'unknown'
    X, names = B.feature_matrix(t)
    check(raises_source(lambda: B.unit_p(t, X, names, bad, 5, 1)), 'unit P stops on an unrecognised medication value')
    check(raises_source(lambda: B.unit_p(t, X, names, med_site[:-1], 5, 1)), 'unit P stops on medication row misalignment')

def main():
    test_partial_spearman()
    objs, t, med = test_pipeline_null_and_planted()
    test_gates(objs, t, med)
    if FAILS:
        print(f'R8_B_SYNTHETIC_FAIL {len(FAILS)}: ' + '; '.join(FAILS))
        return 1
    print('R8_B_SYNTHETIC_PASS all fabricated checks passed (no real data read)')
    return 0
if __name__ == '__main__':
    sys.exit(main())
